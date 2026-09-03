# THIRD_PARTY_NOTICES.md

This project builds on the following third-party works. Sources are fetched at
pinned revisions at build time rather than redistributed, except where noted.

## vLLM
- Project: https://github.com/vllm-project/vllm
- License: Apache-2.0
- This repo rebuilds vLLM from source at a pinned commit and carries one
  derived patch (`container/patches/vsh-rdma-allreduce.patch`), licensed like
  the code it modifies.

## AlexKGwyn/ds4-vllm (RDMA natives, tbv kit, orchestration patterns)
- Project: https://github.com/AlexKGwyn/ds4-vllm
- License: Apache-2.0 (original work)
- `container/native/tbv_ar.c`, `container/native/tbv_ar2.hip`,
  `container/rootfs/.../tbv_ar.py`, `container/rootfs/.../tbv_ar2.py` and the
  `tbv/` kernel-module kit are vendored from that project (with `DS4_*` env
  renames and the v57→v59 rdma-core ABI adaptation). The all-reduce hook is a
  rebased, renamed derivative of its `DS4_TBV_AR2` hook. The host scripts
  follow its orchestration design.

## hellas-ai/thunderbolt-ibverbs
- Project: https://github.com/hellas-ai/thunderbolt-ibverbs
- The kernel RDMA provider (`thunderbolt_ibverbs`), the westeri-thunderbolt
  core/net patch series, and the userspace provider patches applied to
  rdma-core at build time. Kernel code is GPL-2.0; check the upstream repo for
  the exact terms of each part. Pinned at commit
  `76ba39b630a70accb72f19388eefe48844b50eb8` (same pin as ds4-vllm).

## westeri/thunderbolt (Linux Thunderbolt core + net)
- Project: https://git.kernel.org/pub/scm/linux/kernel/git/westeri/thunderbolt.git
- License: GPL-2.0 (Linux kernel code). Pinned at commit `503c5ae1...` by
  `tbv/build-modules.sh`.

## rdma-core (linux-rdma)
- Project: https://github.com/linux-rdma/rdma-core
- License: dual BSD-2-Clause / GPL-2.0. The `usb4_rdma` userspace provider is
  built from rdma-core v59.0 plus the thunderbolt-ibverbs provider patches.

## Base container image
- `docker.io/kyuz0/vllm-therock-gfx1151:rocm10.0.0-torch2.11.0-vllm0.28.0`
  (kyuz0's TheRock ROCm 10.0 / PyTorch 2.11 gfx1151 build). Repo:
  https://github.com/kyuz0/vllm-therock-gfx1151 (check its own notices for the
  ROCm/torch component licenses).

## Model weights (downloaded by scripts/download-models.sh, never redistributed)
- `deepseek-ai/DeepSeek-V4-Flash-0731` — DeepSeek model license; see the
  Hugging Face repo card (https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731).
- `wtdcode/GLM-5.3-Flash-AWQ-W4A16` — community AWQ quantization of
  `zai-org/GLM-5.3-Flash` (Zhipu AI). The upstream model is subject to Zhipu's
  license terms and the "GLM" trademark; see
  https://huggingface.co/zai-org/GLM-5.3-Flash and the quant repo's card
  (https://huggingface.co/wtdcode/GLM-5.3-Flash-AWQ-W4A16) before commercial use.

## Trademarks
AMD, ROCm, Ryzen AI Max, Radeon, Thunderbolt, and USB4 are trademarks of their
respective owners. This project is not affiliated with or endorsed by them.
