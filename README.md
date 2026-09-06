> # ⚠️ STATUS: NOT USABLE FOR REAL WORK — READ THIS FIRST
>
> **GLM-5.3-Flash on vLLM is broken for agentic use, and this repo cannot fix
> it.** Do not adopt this stack for coding agents, tool use, or anything that
> needs reproducible output. The remaining defects are in **vLLM itself** and
> have to be fixed upstream. Until they are, use a different model for that
> work.
>
> ### What is broken — measured on this rig, 2026-09-05
>
> | Symptom | Evidence |
> |---|---|
> | **Tool calling collapses at long context.** The model writes the answer as prose instead of calling the tool, or emits a one-line preamble and stops with `finish_reason: stop` and no tool call. | **3/3** valid tool calls at 229 prompt tokens; **0/3** at 12,288. Forcing `tool_choice` does not rescue it: **1/3** valid at 18,792 tokens, and 2 of 3 burned the **entire 16,000-token budget** emitting empty arguments. |
> | **Greedy decoding is not reproducible above ~2,048 prompt tokens.** | 5 byte-identical requests → 5 different completions at 2,373 / 4,718 / 9,705 / 14,782 prompt tokens (`temperature=0`, fixed seed). |
> | **MTP (speculative decoding) corrupts structured output.** | Must stay disabled — `glm53_mtp_tokens: 0`. See [PATCHES.md](PATCHES.md) §1. |
>
> **Why this matters in practice:** a coding-agent request — system prompt, tool
> schemas, open files — is several thousand tokens. **Every realistic agentic
> request therefore lands in the broken regime.** Short single-turn chat below
> ~2,000 tokens works correctly and is reproducible.
>
> ### What we fixed, and what we did not
>
> We root-caused and fixed **one** of the defects: nondeterministic MoE expert
> selection, where the router selected experts with
> `torch.topk(..., sorted=False)`. Submitted upstream as
> **[vllm-project/vllm#55514](https://github.com/vllm-project/vllm/pull/55514)**
> and carried here as `container/patches/vsh-moe-router-deterministic-topk.patch`
> ([PATCHES.md §14](PATCHES.md)). That makes greedy decoding reproducible
> **below 2,048 prompt tokens only.**
>
> The rest is **not ours to fix**:
>
> - **The DSA sparse-attention indexer's top-k above `index_topk`** — tracked in
>   [vllm-project/vllm#54521](https://github.com/vllm-project/vllm/issues/54521).
>   Note that the in-flight fix
>   [#55122](https://github.com/vllm-project/vllm/pull/55122) repairs the CUDA
>   `persistent_topk` path, which **ROCm never calls**
>   (`sparse_attn_indexer.py` gates it on `current_platform.is_cuda()`); on this
>   stack the op in play is `top_k_per_row_prefill`. So #55122 landing will
>   **not** fix ROCm.
> - **The long-context tool-call collapse** — not yet root-caused. It is *not*
>   caused by the nondeterminism: it persists unchanged with the router fix
>   live, and its failures are budget runaways rather than output variation.
>
> Working notes, including every exclusion and the retractions:
> **[PENDINGWORK.md](PENDINGWORK.md)**.

# vllm-strix-halo — GLM-5.3-Flash (and DeepSeek-V4-Flash) on 2× AMD Strix Halo, TP=2 over Thunderbolt

Serve **GLM-5.3-Flash** (and **DeepSeek-V4-Flash**) with vLLM across **two AMD
Strix Halo (gfx1151) boxes** — one 8060S iGPU per box, tensor-parallel (TP=2),
with the inter-GPU all-reduce carried over a **Thunderbolt-4/USB4** cable
between them. The shipped default (`transport: hybrid`) runs RCCL over IP
sockets on the cable and keeps a custom RDMA fast-path (`tbv_ar2`,
`usb4_rdma0`) for the small decode collectives; sockets measured faster than
the RoCE path at every message size, so `transport: tcp` — no kernel modules
at all — is equally valid ([PATCHES.md §10](PATCHES.md)).

GLM-5.3-Flash support is **upstream vLLM** since 2026-09-03
([#53906](https://github.com/vllm-project/vllm/pull/53906)). This repo rebuilds
vLLM at a post-merge commit on kyuz0's proven **ROCm 10.0 gfx1151** base image
(no prebuilt gfx1151 image contains the merge yet), adds the Thunderbolt RDMA
fabric pieces, and ships the host orchestration that starts/stops/drives the
2-box cluster. Which patches are carried from the DeepSeek build and why:
**[PATCHES.md](PATCHES.md)**.

This project builds on and reuses the RDMA work of
[`AlexKGwyn/ds4-vllm`](https://github.com/AlexKGwyn/ds4-vllm) — the model work
is dropped (GLM is upstream), the fabric work is kept.

## Hardware & prerequisites

- **2× AMD Strix Halo (gfx1151)** boxes, ~128 GB unified memory each.
- A **Thunderbolt-4/USB4 cable** between them (the RDMA rail rides the cable).
- Linux with kernel headers for the running kernel, `podman`, `toolbox`,
  `rdma-core`, `git`, build toolchain; **Secure Boot disabled** (the tbv
  modules are unsigned).
- The model weights on **both** boxes (see below).

## Quick start

```bash
git clone https://github.com/davidcanar/vllm-strix-halo ~/vllm-strix-halo

# 0. weights, on BOTH boxes (~191 GB GLM / ~156 GB DS4, resumable)
scripts/download-models.sh glm53     # or: ds4, all

# 1. RDMA kernel modules, on BOTH boxes, then reboot both together
#    OPTIONAL: skip this and set `transport: tcp` in ~/vsh-config.yaml -- it
#    measures the same single-stream (PATCHES.md §10) and needs no Secure Boot
#    change, no unsigned modules, no coordinated reboot.
tbv/build-modules.sh && sudo tbv/install-modules.sh

# 2. build the serving image (box1), copy to box2 (podman save | podman load),
#    create the toolbox "vllm-glm" on both boxes
container/build.sh

# 3. site config + host scripts
cp host/vsh-config.yaml ~/vsh-config.yaml   # edit IPs / paths / ports
host/deploy.sh

# 4. launch (box1) — full 2-box bringup, OpenAI API when done
./vllm-strix-halo.sh start            # glm53 by default (API :1235)
./vllm-strix-halo.sh status
./vllm-strix-halo.sh logs
./vllm-strix-halo.sh stop

# DeepSeek-V4-Flash (needs the deployed ds4-vllm stack; see AGENTS.md)
./vllm-strix-halo.sh ds4 start
```

Full ordered runbook with gates and gotchas: **[AGENTS.md](AGENTS.md)**.

## What's inside

| path | what |
|---|---|
| `vllm-strix-halo.sh` | the launcher: `[glm53\|ds4] [start\|stop\|status\|logs]` |
| `scripts/download-models.sh` | weight downloads for both models, both boxes |
| `container/` | Dockerfile + build.sh: vLLM ≥ GLM merge on the ROCm 10 gfx1151 base, usb4_rdma provider, tbv_ar2 natives, `vsh-rdma-allreduce.patch` |
| `tbv/` | Thunderbolt RDMA kernel-module kit (vendored from ds4-vllm, GPL-2.0 side) |
| `host/` | cluster env/config/restart/down/serve scripts, systemd unit, deploy.sh |
| `PATCHES.md` | the ds4-vllm → GLM patch review |
| `PENDINGWORK.md` | **open defects, what was ruled out, and where to resume** |
| `AGENTS.md` | the ordered end-to-end runbook |

## Models

| profile | model | weights | API port | quantization |
|---|---|---|---|---|
| `glm53` | GLM-5.3-Flash | `wtdcode/GLM-5.3-Flash-AWQ-W4A16` (~191 GB) | 1235 | compressed-tensors W4A16, bf16 KV; MTP speculative decoding **should be disabled** (`glm53_mtp_tokens: 0`) — it corrupts structured output on gfx1151; see [Known correctness issues](#known-correctness-issues) |
| `ds4` | DeepSeek-V4-Flash | `deepseek-ai/DeepSeek-V4-Flash-0731` (~156 GB) | 1234 | fp8 KV + DSpark MTP (ds4-vllm image) |

> The official `zai-org/GLM-5.3-Flash` checkpoint is FP8 ≈ 335 GB — it cannot
> fit 2×128 GB UMA or the reference worker's disk. AWQ W4A16 is the format that
> fits; see PATCHES.md §3.

## Known correctness issues

**Read this before trusting any output from this stack.** Three defects were
found on 2026-09-04, after the performance work below. Full investigation,
measurements and reproducers: **[PENDINGWORK.md](PENDINGWORK.md)**.

| # | defect | status | mitigation |
|---|---|---|---|
| 1 | **MTP corrupts structured output.** Requests die at exactly 12 output tokens (3 MTP steps at k=3) with 13-character tool-call arguments; ~4k-token prompts return empty responses. The upstream recipe says MTP is unsupported on ROCm. | ours to avoid | **`glm53_mtp_tokens: 0`** — costs ~3x decode throughput |
| 2 | **Greedy decoding is not reproducible.** 5 identical `temperature 0` requests → 5 different completions, at every length down to 244 prompt tokens. Some runs degenerate into repetition loops; retrieval returns near-tie wrong tokens (`velvet-harbor` for `velvet-harpoon`). | upstream, open — [vllm#54521](https://github.com/vllm-project/vllm/issues/54521) ([our data](https://github.com/vllm-project/vllm/issues/54521#issuecomment-5545047644)) | none available; `VLLM_BATCH_INVARIANT` requires NVIDIA CC ≥ 9.0 |
| 3 | **Tool calling collapses above ~10k prompt tokens.** The model stops emitting `</tool_call>`, so the parser buffers forever and the client sees HTTP 200 with nothing after the preamble. Coding harnesses stall silently. | likely a consequence of #2 | repeat the tool-call format at the *end* of the prompt (validated at 12k) |

Defect 2 means **this rig should not be used for anything requiring
reproducible output**, and evaluation numbers from it carry run-to-run variance
that has nothing to do with sampling.

## Performance

> ⚠️ The decode figures below were measured with **MTP enabled**, i.e. with
> defect 1 active. The **timing** is still valid — ms/step and tok/s do not
> depend on which token is emitted — but with MTP disabled for correctness you
> lose the 2.2–4.0 tokens/step speculative gain, so real decode throughput is
> roughly **3x lower** than the tok/s quoted here. The prefill and TTFT
> figures are unaffected: those changes (chunk size, `NCCL_PROTO`, transport)
> do not touch speculative decoding.

Reference rig: 2× Ryzen AI Max+ 395 / 128 GB, single stream, temperature 0,
MTP = 3 draft tokens, `max_ctx: 131072`, shipped `transport: hybrid`.

Two numbers matter, and different changes moved each:

- **decode → ms/step.** Tokens/s at a fixed step time is set by MTP
  acceptance, which swings 40–100% with how predictable the text is (see the
  table below), so step time is the reproducible figure.
- **prefill → tok/s** on *uncacheable* prompts (nonce-prefixed so the
  prefix cache never hits), warm kernels — plus TTFT.

### Where it started, where it is

| | bring-up | now | |
|---|---:|---:|---|
| decode step | 359 ms | 200–231 ms | **~1.6×** |
| prefill | 151 tok/s | 284–334 tok/s | **~+100%** |
| TTFT, 2 841-token prompt | 18.05 s | 8.7 s | **−52%** |
| cold 128 K prefill (extrapolated from 32 K) | 13.9 min | ~7.1 min | |

Four independent changes got there, each measured on its own:

| change | knob | effect | why |
|---|---|---|---|
| tuned fused-MoE tile configs | `host/moe-configs/` | decode step 359 → 225 ms | the stock int4 path hardcodes `BLOCK_SIZE_K=32`; 38 → 95 GB/s ([§6](PATCHES.md)) |
| prefill chunk 512 → 4096 | `glm53_max_batched` | prefill 151 → 196 tok/s | the "NOT larger" warning was over-cautious — the big indexer buffer doesn't scale with this knob ([§8](PATCHES.md)) |
| unpin `NCCL_PROTO` (was `LL`) | `VSH_NCCL_PROTO` | prefill 196 → 277 tok/s | LL exists for tiny collectives, but RCCL only ever sees the 29 MB prefill ones ([§9](PATCHES.md)) |
| RCCL over sockets, not the RoCE rail | `transport` | prefill 271 → 305 tok/s, TTFT −0.8 s | sockets beat `usb4_rdma0` at **every** message size ([§10](PATCHES.md)) |

### Measured now

Uncacheable prompts, token counts straight from `usage.prompt_tokens`:

| prompt | prefill | TTFT | decode step | tok/step | acceptance | decode |
|---:|---:|---:|---:|---:|---:|---:|
| 523 | 284 tok/s | 1.8 s | 200.6 ms | 4.00 | 100% | 19.9 tok/s |
| 2 084 | 334 tok/s | 6.2 s | 221.3 ms | 4.00 | 99% | 18.1 tok/s |
| 8 190 | 316 tok/s | 25.9 s | 215.7 ms | 2.22 | 41% | 10.3 tok/s |
| 32 830 | 306 tok/s | 107.4 s | 230.8 ms | 3.87 | 96% | 16.8 tok/s |

Both curves are flat: **prefill does not degrade with context** (0.5 K → 32 K
holds ~300 tok/s — the DSA sparse attention is doing its job, there is no
quadratic term) and **decode step does not either**, which is the signature of
a context-independent MoE bottleneck. What moves decode tok/s is MTP
acceptance alone. Three repeats of one ordinary instruction prompt, same
server, show the spread: 96.9 / 65.5 / 85.7% acceptance → 20.0 / 14.3 / 17.3
tok/s at a near-identical 195–206 ms step. So quote step time; treat any
single tok/s reading as a sample from a 10–20 tok/s band.

### Negative results — don't re-run these

- **MoE tile and occupancy tuning is exhausted.** A second sweep over
  `waves_per_eu` and `SPLIT_K` gained 1.04–1.18× *in the kernel* and nothing
  end to end. M = 4 sits at ~95 GB/s while the model's own BF16 GEMVs reach
  ~214 GB/s on this GPU; closing that needs a different kernel, not different
  parameters ([§6](PATCHES.md)).
- **CUDA graphs are blocked.** Rank 0 captures (11.43 GiB), rank 1 dies in a
  Triton launch hook with `SystemError` — and 11.43 GiB/rank is unaffordable
  against ~18.6 GiB free regardless. `glm53_enforce_eager: 1` stays
  ([§11](PATCHES.md)).
- **The RoCE rail is not the win the repo name implies.** `transport: tcp`
  measures the same as `hybrid` within noise, so a fresh rig can skip the tbv
  kernel-module build entirely. Single-stream only — RDMA may still matter at
  concurrency ([§10](PATCHES.md)).

### What's left

The largest remaining decode cost is the **weights the AWQ checkpoint leaves
in BF16**: `self_attn.*` — which includes all 34 KDA layers — is ~6.1 GB/rank
of reads every step, ~48 ms, **24% of decode**, and would be ~7 ms at int4.
The MTP block `layers.45.*` is another ~9 ms/step and ~7 GB/rank of memory
that the KV pool would rather have. Both need a re-quantized checkpoint, not a
serving change ([§12](PATCHES.md)).

### Bring-up baseline (historical)

The original bring-up numbers, kept because they are the only measurement of
MTP's own contribution. Whole-request tok/s including prefill, so they are not
comparable to the step times above, and the generation length was not
recorded:

| context | MTP off | MTP on | MTP + tbv_ar2 |
|---|---|---|---|
| 512 | ~6.7 tok/s | ~7.6 tok/s | ~7.9 tok/s |
| 4.5k | ~2.4 tok/s | ~5.5 tok/s | ~5.6 tok/s |

MTP needed a gfx1151 patch — aiter's asm sparse-decode kernel aborts on GLM's
rope-free MLA ([§1.5](PATCHES.md)) — and accepts 40–100% of draft tokens
depending on the prompt. tbv_ar2, the custom decode all-reduce over the
Thunderbolt rail, turned out to work fine on the ROCm 10 stack once MTP was
fixed (the "crash" was a misattribution, [§5.0](PATCHES.md)); it is on by
default but is neutral now that RCCL runs on sockets ([§10](PATCHES.md)).
Treat any figure quoted elsewhere as unverified.

## License & attribution

Original work here is **Apache-2.0** ([LICENSE](LICENSE)). The kernel modules
and their patch series are GPL-2.0 (hellas-ai/thunderbolt-ibverbs,
westeri/thunderbolt); the RDMA natives and tbv kit derive from
AlexKGwyn/ds4-vllm (Apache-2.0); rdma-core is dual BSD/GPL. Third-party
sources are fetched at pinned revisions at build time, not redistributed. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
