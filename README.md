> # STATUS: re-validated on the 2026-09-27 merge — materially better, two caveats remain
>
> **The 2026-09-05 "not usable" verdict is obsolete.** This repo now pins
> vLLM at a 2026-09-27 commit carrying the GLM-5.3-Flash correctness wave
> (sparse-indexer topk backend selection [#58594], kpool corruption with
> speculative decoding [#58454, the AMD path], SparseIndexerTopk dispatcher
> [#57546], FlashKDA prefill, ...) and runs the fabric on the OdinLink
> driver. Re-measured on this rig, 2026-09-27:
>
> | Was broken (2026-09-05) | Now (2026-09-27) |
> |---|---|
> | Greedy: 5 identical requests → 5 different completions at every length | **Byte-identical (1/5 distinct) at single-chunk prompts** (≤4,096 tokens) via `vsh-indexer-persistent-topk-rocm.patch` (deterministic `persistent_topk` was compiled into HIP builds all along — only a python `is_cuda` gate kept it from ROCm). **Multi-chunk prefill still diverges**: the prefill path hardcodes `top_k_per_row_prefill` with no backend dispatch — upstream work. |
> | Tool calling: 0/3 valid at 12,288 tokens; budget runaways (16K tokens of empty args) | **3/3 valid at short context** (also with MTP on). At 12K-token single-message contexts the model is now **coherent and grounded** (it correctly reads and summarizes the padding — the corruption is gone) but still often answers in prose instead of calling the tool. |
> | MTP corrupts structured output; must stay off | **MTP validated ON** (`glm53_mtp_tokens: 3`): clean tool calls, no 12-token/13-char-arg corruption, **~1.9× decode** (10.3 vs 5.5 tok/s). The kpool spec-decode fix (#58454) holds on this rig. |
>
> **Practical read:** short-context agentic use (system prompt + schemas +
> a few files) is now in the *working* regime and reproducible. Very long
> single-message contexts (>4K tokens of prompt) remain non-reproducible and
> unreliable for tool routing — prefill-side top-k determinism is the
> remaining upstream blocker.
>
> History of the 2026-09-05 investigation: [PENDINGWORK.md](PENDINGWORK.md).
> The 2026-09-27 measurements and the two init-time fixes this rig needed
> (the WNA16-conversion GTT reservation OOM, the persistent-topk unlock):
> [PATCHES.md §15](PATCHES.md).

# vllm-strix-halo — GLM-5.3-Flash (and DeepSeek-V4-Flash) on 2× AMD Strix Halo, TP=2 over Thunderbolt

Serve **GLM-5.3-Flash** (and **DeepSeek-V4-Flash**) with vLLM across **two AMD
Strix Halo (gfx1151) boxes** — one 8060S iGPU per box, tensor-parallel (TP=2),
with the inter-GPU fabric carried over a **Thunderbolt-4/USB4** cable between
them via the **OdinLink driver** (`odl_tb5`): RCCL rides the OdinLink net
plugin for the big prefill-sized collectives, the `odl_ar2` native carries the
small decode all-reduces over odl streams, and `odl_mq` (fail-open) moves the
EngineCore↔worker control plane off the per-step TCP round trips. The default
is `transport: odl`; `transport: tcp` (plain sockets, no kernel modules)
remains the fallback.

GLM-5.3-Flash support is **upstream vLLM** since 2026-09-03
([#53906](https://github.com/vllm-project/vllm/pull/53906)), and the
**late-September correctness wave is in**: sparse-indexer topk backend
selection ([#58594](https://github.com/vllm-project/vllm/pull/58594)), kpool
corruption with speculative decoding ([#58454](https://github.com/vllm-project/vllm/pull/58454)
— the AMD path), the SparseIndexerTopk dispatcher (#57546), and more. This
repo rebuilds vLLM at a pinned **2026-09-27** commit on kyuz0's proven
**ROCm 10.0 gfx1151** base image, adds the OdinLink fabric userspace, and
ships the host orchestration that starts/stops/drives the 2-box cluster.
Which patches are carried and why: **[PATCHES.md](PATCHES.md)**.

This project builds on and reuses the fabric work of
[`AlexKGwyn/ds4-vllm`](https://github.com/AlexKGwyn/ds4-vllm) — the model work
is dropped (GLM is upstream), the fabric work is kept.

## Hardware & prerequisites

- **2× AMD Strix Halo (gfx1151)** boxes, ~128 GB unified memory each.
- **Exactly one** Thunderbolt-4/USB4 cable between them (OdinLink demultiplexes
  peers by route; a second cable breaks it).
- Linux (stock kernel) with kernel headers, `podman`, `toolbox`, `git`, build
  toolchain; **Secure Boot disabled** (the `odl_tb5` module is unsigned unless
  a MOK is enrolled — the install path signs it when one exists).
- The model weights on **both** boxes (see below).

## Quick start

```bash
git clone https://github.com/davidcanar/vllm-strix-halo ~/vllm-strix-halo

# 0. weights, on BOTH boxes (~191 GB GLM / ~156 GB DS4, resumable)
scripts/download-models.sh glm53     # or: ds4, all

# 1. OdinLink driver, on BOTH boxes (stock kernel; one cable between them)
#    OPTIONAL: skip this and set `transport: tcp` in ~/vsh-config.yaml -- no
#    kernel modules at all, but you lose the odl_ar2 decode fast path and the
#    net plugin.
odinlink/build-odinlink.sh && sudo odinlink/install-odinlink.sh
#    (migrating from the old tbv stack? sudo odinlink/uninstall-tbv.sh --apply
#     first, then reboot both boxes together)

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
| `container/` | Dockerfile + build.sh: vLLM main pin on the ROCm 10 gfx1151 base, OdinLink userspace (net plugin + `odl_ar2`), the vLLM patch set |
| `odinlink/` | OdinLink driver kit (vendored from ds4-vllm, GPL-2.0 side): `odl_tb5.ko` build/install, `odl_ar2` sources, uninstall-tbv migration |
| `host/` | cluster env/config/restart/down/serve scripts, systemd unit, deploy.sh |
| `PATCHES.md` | the patch review: what is carried, from where, and why |
| `PENDINGWORK.md` | the 2026-09 correctness investigation record |
| `AGENTS.md` | the ordered end-to-end runbook |

## Models

| profile | model | weights | API port | quantization |
|---|---|---|---|---|
| `glm53` | GLM-5.3-Flash | `wtdcode/GLM-5.3-Flash-AWQ-W4A16` (~191 GB) | 1235 | compressed-tensors W4A16, bf16 KV; MTP speculative decoding defaults OFF (`glm53_mtp_tokens: 0`) — it corrupted structured output on the Sep-3 pin; upstream #58454 (kpool corruption with spec decode) is in the Sep-27 pin and MTP is being re-validated |
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

## Performance (measured on the 2026-09-03 pin — re-measurement on the Sep-27 pin pending)

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
