# vllm-strix-halo — GLM-5.3-Flash (and DeepSeek-V4-Flash) on 2× AMD Strix Halo, TP=2 over Thunderbolt RDMA

Serve **GLM-5.3-Flash** (and **DeepSeek-V4-Flash**) with vLLM across **two AMD
Strix Halo (gfx1151) boxes** — one 8060S iGPU per box, tensor-parallel (TP=2),
with the inter-GPU all-reduce carried over a **Thunderbolt-4/USB4 RoCE-RDMA**
link (`usb4_rdma0`).

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
| `glm53` | GLM-5.3-Flash | `wtdcode/GLM-5.3-Flash-AWQ-W4A16` (~191 GB) | 1235 | compressed-tensors W4A16, bf16 KV; MTP speculative decoding enabled (~92% acceptance, ~2.3× at long ctx — PATCHES.md §1.5) |
| `ds4` | DeepSeek-V4-Flash | `deepseek-ai/DeepSeek-V4-Flash-0731` (~156 GB) | 1234 | fp8 KV + DSpark MTP (ds4-vllm image) |

> The official `zai-org/GLM-5.3-Flash` checkpoint is FP8 ≈ 335 GB — it cannot
> fit 2×128 GB UMA or the reference worker's disk. AWQ W4A16 is the format that
> fits; see PATCHES.md §3.

## Performance note

Validated on the reference rig (2× Ryzen AI Max+ 395 / 128 GB each), single
stream, temperature 0, TP all-reduce over the Thunderbolt RoCE rail (RCCL
`Using network IB` on `usb4_rdma0`, 40 Gbps), MTP = 3 draft tokens:

| context | MTP off | MTP on |
|---|---|---|
| 512 | ~6.7 tok/s | ~7.6 tok/s |
| 4.5k | ~2.4 tok/s | ~5.5 tok/s |

MTP needed a gfx1151 patch (aiter's asm sparse-decode kernel aborts on GLM's
rope-free MLA; PATCHES.md §1.5) and accepts ~92% of draft tokens. The
per-step fp8→fnuz conversion on the sparse-MQA path is the known next
optimization (PATCHES.md §1.3). Treat any figure quoted elsewhere as
unverified.

## License & attribution

Original work here is **Apache-2.0** ([LICENSE](LICENSE)). The kernel modules
and their patch series are GPL-2.0 (hellas-ai/thunderbolt-ibverbs,
westeri/thunderbolt); the RDMA natives and tbv kit derive from
AlexKGwyn/ds4-vllm (Apache-2.0); rdma-core is dual BSD/GPL. Third-party
sources are fetched at pinned revisions at build time, not redistributed. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
