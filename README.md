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
| `AGENTS.md` | the ordered end-to-end runbook |

## Models

| profile | model | weights | API port | quantization |
|---|---|---|---|---|
| `glm53` | GLM-5.3-Flash | `wtdcode/GLM-5.3-Flash-AWQ-W4A16` (~191 GB) | 1235 | compressed-tensors W4A16, bf16 KV; MTP speculative decoding enabled (2.2-4.0 tokens/step measured, 40-100% draft acceptance depending on the prompt — PATCHES.md §1.5) |
| `ds4` | DeepSeek-V4-Flash | `deepseek-ai/DeepSeek-V4-Flash-0731` (~156 GB) | 1234 | fp8 KV + DSpark MTP (ds4-vllm image) |

> The official `zai-org/GLM-5.3-Flash` checkpoint is FP8 ≈ 335 GB — it cannot
> fit 2×128 GB UMA or the reference worker's disk. AWQ W4A16 is the format that
> fits; see PATCHES.md §3.

## Performance

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
