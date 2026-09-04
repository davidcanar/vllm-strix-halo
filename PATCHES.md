# PATCHES.md — what this repo patches, and why (review of `AlexKGwyn/ds4-vllm`)

This is the review you asked for: *which patches from the DeepSeek-V4-Flash
Strix Halo build are required to run **GLM-5.3-Flash** on the same 2-box
gfx1151 rig over RDMA?*

**Short answer:** almost none of the model work — GLM-5.3-Flash support is now
**upstream vLLM** ([`vllm-project/vllm#53906`](https://github.com/vllm-project/vllm/pull/53906),
merged 2026-09-03, `vllm/models/glm5next/`, architecture
`Glm5NextForCausalLM` / `Glm5NextForConditionalGeneration`, AMD Triton ops
included). What *is* needed is a small slice of the **platform/infrastructure**
work: the Thunderbolt RDMA fabric pieces and one model-agnostic all-reduce
hook. This repo ships exactly that slice and rebuilds vLLM at a post-merge
commit on the ROCm 10 gfx1151 base.

---

## 1. The ds4-vllm patch inventory, classified for GLM-5.3-Flash

`AlexKGwyn/ds4-vllm` = vLLM commit `470229c` + 31 modified files + 12 new
files, layered on `kyuz0/vllm-therock-gfx1151` (ROCm 7.14).

| ds4-vllm patch area | Needed for GLM-5.3-Flash here? | Why |
|---|---|---|
| `vllm/models/deepseek_v4/*` (model, rocm, dspark_mtp, attention, MLA ops) | ❌ **No** | DeepSeek-V4-specific. GLM-5.3 has its own upstream model code. |
| Sparse indexer / top-k kernels (`ds4_topk.py`, `ds4_tl_indexer.py`, `rocm_aiter_mla_sparse.py`, …) | ❌ **No** | DS4 indexer internals. GLM-5.3's kpool indexer is upstream (`SparseAttnIndexerKpool`, Triton). |
| MoE/GEMM tuning (`opt_flags.py`, MXFP4 `matmul_ogs` knobs, `DS4_MOE_*`, A8W8 config) | ❌ **No** | MXFP4/MoE decode tuning written for DS4's expert layout. GLM-5.3 runs AWQ W4A16 here — different kernels entirely. |
| Disk KV tier (`fs_lru`, `distributed.py`, offload batch bounds) | ❌ **No** | Valuable work, but a DS4-specific tier. vLLM upstream has its own offloading stack; we start without a disk tier (see §5). |
| OpenAI reasoning/tool fixes (`deepseek_v4_encoding.py`, structured-output `</think>` boundary) | ❌ **No** | DS4 tokenizer/parser work. GLM-5.3 uses the standard chat path. |
| **Thunderbolt RDMA kernel modules** (`tbv/`: patched `thunderbolt`, `thunderbolt_net`, `thunderbolt_ibverbs`, `nhi_throttle`) | ✅ **Yes — verbatim** | This is the fabric itself. It is model-agnostic. Already built & loaded on the reference rig; vendored here (`tbv/`) so a fresh pair of boxes can be rebuilt. GPL-2.0 (kernel side) — see THIRD_PARTY_NOTICES.md. |
| **`usb4_rdma` libibverbs provider built into the image** | ✅ **Yes — adapted** | Without it `ibv_devices` is empty inside the container and RCCL silently falls back to TCP. ds4-vllm built it against rdma-core v57 (its base's ABI); this repo ships the same proven pair — provider rdmav57 + libibverbs 1.14.58 — because the ROCm 10 base's rdmav59-era libibverbs gets EINVAL from the thunderbolt kernel driver's uverbs dispatch (verified with RCCL falling back to `Using network Socket` until the swap, then `Using network IB`). |
| **`DS4_TBV_AR2` custom all-reduce hook** (`cuda_communicator.py`) + `tbv_ar2.hip` native | ✅ **Yes — rebased, renamed `VSH_TBV_AR2`, enabled by default** | Model-agnostic, env-gated, fail-open. Carries the small decode all-reduce (~150 µs/op measured on this stack at 8 KiB) while prefill-sized collectives go over RCCL-over-IB on the same rail. The hook lives in vLLM's *distributed* layer, not in any model file — the DS4 model patches (`deepseek_v4/amd/model.py`) are **not** part of it. Rebased against vLLM main and shipped as `container/patches/vsh-rdma-allreduce.patch`. Validated serving end-to-end (PATCHES.md §5.0 — the earlier "crashes on ROCm 10" note was a misattribution). |
| `tbv_ar` v1 native (`tbv_ar.c`) | ⚠️ Vendored, not wired | DS4's v1 path was superseded by v2 and its hook is not carried. The native builds for experimentation; only v2 is hooked. |
| RCCL CQ warm-up (DS4 `deepseek_v4/amd/model.py` +67) | ⚠️ **Candidate, not applied** | DS4 pre-creates RCCL's RDMA completion queue before the ~80 GiB weight load to dodge an ENOMEM crash on ROCm 7.14. If the failure reproduces on the ROCm 10 stack we will port it (it's ~15 lines, model-local). Not applied preemptively — see §5. |
| ROCr rebuild + `rocr-force-block-indefinite-active-wait.patch` | ❌ **No** | ROCm 7.14-era idle-CPU fix. ROCm 10 supports `HSA_ENABLE_INTERRUPT=1` natively (set in `vsh-cluster-env.sh`) — the rebuild is unnecessary here. |
| `rocm.py` `get_device_name` fallback (amdsmi-mock) | ❌ **No** | Test-harness convenience; not needed to serve. |
| `breakable_cudagraph.py` stream-sync fix | ❌ **No (for now)** | We serve `--enforce-eager` (same as DS4). If cudagraphs are enabled later, revisit. |
| Scheduler / KV / MTP tweaks (`kv_cache_utils.py`, `llm_base_proposer.py`, …) | ❌ **No** | DS4-specific layouts and DSpark drafter. GLM-5.3's KV grouping and its MTP (`glm5_next_mtp`) are upstream. |

**Net:** this repo patches **five** vLLM files (the all-reduce hook, the
TileLang-MHC gate, the aiter support gate, the fp8 fnuz casts, and the
MTP rope-free Triton routing) and replaces three
RDMA userspace pieces in the image (provider, libibverbs, tools). Everything
else is upstream vLLM + configuration.

The non-DS4 patches were found during the reference-rig bring-up (all
gfx1151-specific, all upstream-bug-class):

1. **TileLang MHC gate** (`vsh-mhc-no-tilelang-gfx1151.patch`): upstream
   enables TileLang `mhc` fused kernels on ROCm for everything except gfx942,
   but the compiled kernels crash natively on gfx1151 (both TP ranks die on
   the first forward). The patch routes gfx1151 to the torch/triton
   fallbacks upstream already maintains for gfx942.
2. **AITER support gate** (`vsh-aiter-gfx1151-gate.patch`): upstream gates
   aiter to `get_cdna_version() > 2`, hiding kyuz0's gfx1151 aiter build
   (which ships the sparse-attention indexer op) and making the glm5next
   sparse indexer raise "Sparse attention indexer ROCm path is only
   supported on AITER" no matter the env. The patch admits gfx1151 when
   aiter is present.
3. **fp8 format** (`vsh-fp8-fnuz-mqa.patch` + the aiter
   `pa_mqa_logits.py` overlay): this stack's `current_platform.fp8_dtype()`
   is `fp8e4nv` (NVIDIA format), but triton's `tl.dot` on AMD only accepts
   `fp8e4m3fnuz` — the sparse-MQA logits kernels die with "Unsupported lhs
   dtype fp8e4nv". q is cast to fnuz at the vLLM call sites and the aiter
   wrapper converts the packed K payload.
4. **RDMA userspace ABI**: the base image's rdmav59-era libibverbs marshals
   `query_device` in a way the thunderbolt kernel driver rejects (EINVAL) —
   RCCL silently fell back to TCP. The image now ships the DS4-proven pair:
   provider `rdmav57` + libibverbs `1.14.58` (+ matching `ibv_*` tools),
   and RCCL logs `Using network IB` (`usb4_rdma0`, RoCE, 40 Gbps).
5. **MTP rope-free routing** (`vsh-mtp-ropefree-triton-sparse.patch`): GLM-5.3
   is a *rope-free* MLA model (`qk_rope_head_dim = 0`, head 512 ==
   `kv_lora_rank`), and aiter's asm sparse-decode kernel table has **no
   heuristic kernel** for that layout — `aiter.mla.mla_decode_fwd` fails
   `get_heuristic_kernel_mla()` and calls `abort()` (SIGABRT, no traceback).
   Upstream vLLM knows aiter can't serve rope-free and routes prefill/plain
   decode to its Triton ragged kernel — but its gate
   (`_use_rocm_sparse_triton`: `plain_decode` / `max_query_len == 1`) sends
   every **MTP-shaped batch** (draft steps flattened to next_n single-token
   rows; the multi-token verify query) into the aiter path, killing the
   worker on the first speculative batch. The patch keeps the upstream
   conditions first and, on gfx1151, falls back to the Triton ragged kernel
   for rope-free BF16 batches of any shape — safe because the backend's own
   metadata builder already flattens *every* query token to its own ragged
   row, which is exactly the input that kernel serves for prefill and plain
   decode. Result: MTP runs end-to-end (~92% draft acceptance, ~2.3× at
   4.5k ctx vs MTP-off; see §5.4).

Config-only bring-up findings (no patch): the vision-encoder cache profiling
kills the worker on gfx1151 → `--skip-mm-profiling`; the aiter env must be
baked into the image (vLLM snapshots it at import, before the driver→worker
env RPC lands); `VLLM_ROCM_USE_AITER_MOE=0` (aiter MoE kernels don't support
gfx1151); aiter needs the gfx1151 GEMM/BATCHED_GEMM tuned configs (cloned
from the gfx1201 RDNA4 tunes into the image).

## 2. Why the image rebuilds vLLM at all

The GLM merge landed **2026-09-03** (`98ed0856`). The newest kyuz0 gfx1151
images at the time of writing (`rocm10.0.0-torch2.11.0-vllm0.28.0`,
2026-08-30; `latest`, 2026-08-31) predate it — and `v0.29.0rc2` was tagged
seven hours *before* the merge. So there is no prebuilt gfx1151 image with
GLM-5.3 support; `container/Dockerfile` rebuilds vLLM from source at a pinned
post-merge commit (`8bf3963` by default) **on top of** kyuz0's ROCm 10 base,
keeping the base's gfx1151-built torch 2.11 / triton 3.8 / aiter:

```
kyuz0/vllm-therock-gfx1151:rocm10.0.0-torch2.11.0-vllm0.28.0
  ├─ pip uninstall vllm (0.28, pre-merge)
  ├─ pip install vllm@8bf3963  --no-build-isolation --no-deps   (PYTORCH_ROCM_ARCH=gfx1151)
  ├─ vLLM runtime deps from its own pinned requirements (minus torch/triton/aiter/amdsmi)
  ├─ usb4_rdma provider (rdma-core v59.0 + hellas-ai patches)
  ├─ tbv_ar2 / tbv_ar natives (hipcc gfx1151)
  └─ vsh-rdma-allreduce.patch  (cuda_communicator.py)
```

vLLM's official ROCm build pins torch 2.12 + ROCm 7.14; the base's torch
2.11 + ROCm 10.0 gfx1151 build is the proven Strix Halo toolchain and is kept
untouched (`--no-build-isolation --no-deps`).

## 3. Why AWQ W4A16 (the weight constraint)

| checkpoint | size | fits 2×128 GB UMA (TP=2)? | fits both boxes' disks? |
|---|---|---|---|
| `zai-org/GLM-5.3-Flash` (FP8) | ~335 GB | ❌ (167 GB/box > GPU budget) | ❌ worker has 230 GB free |
| `zai-org/GLM-5.3-Flash-BF16` | ~643 GB | ❌ | ❌ |
| `wtdcode/GLM-5.3-Flash-AWQ-W4A16` (compressed-tensors, group 128) | ~191 GB | ✅ ~95 GB/box + KV | ✅ |
| GGUF Q4 (llama.cpp path) | ~178 GB | n/a (llama.cpp) | already on the worker |

GLM-5.3-Flash is a ~300 B-class MoE (46 layers × 288 experts + shared
experts, MLA + kpool sparse indexer + Mamba/KDA hybrid, MTP). Serving it on
Strix Halo requires 4-bit weights; the AWQ W4A16 checkpoint is served with
`--quantization compressed-tensors`, and upstream vLLM carries a dedicated
gfx1151 W4A16 kernel path (`vllm/model_executor/kernels/linear/
mixed_precision/rdna_hybrid_w4a16.py`: Triton prefill + HIP skinny decode).

## 4. What the reference rig already provides (not rebuilt)

Both boxes already run the patched Thunderbolt kernel stack and the RoCE
rail is `ACTIVE` (`usb4_rdma0`, GID 1 = the 10.0.2.x IPs, `thunderbolt0`
UP). On a fresh pair of boxes the vendored `tbv/` kit rebuilds it:
`tbv/build-modules.sh && sudo tbv/install-modules.sh`, then reboot both
together. See AGENTS.md §RDMA.

## 5. Known gaps / deliberate omissions (watch list)

0. **tbv_ar2 decode fast-path — WORKS on ROCm 10; the "crash" was a
   misattribution.** The bring-up-era note ("worker dies natively with
   VSH_TBV_AR2=1 during the first collective") did not survive re-testing:
   a full journal audit found `VSH_TBV_AR2=1` was **never actually enabled**
   on this stack (the deployed `~/vsh-config.yaml` predated the
   `glm53_tbv_ar2` key, and `vsh-cluster-restart.sh` never passed it through
   ENVPASS/serve env either). Every warmup crash in the window — including
   the one originally blamed on tbv_ar2 — carries the aiter abort signature
   of §1.5 (MTP was on by default then). After the MTP fix, re-testing
   showed: isolated 2-rank selftest over the real rail (both containers)
   passes 25/25 exact-sum rounds at ~152 µs/op (8 KiB decode size); serving
   with `glm53_tbv_ar2: 1` + MTP runs clean (API 200, greedy quality
   correct, ~90% draft acceptance as the all-reduce correctness canary,
   ~7.9 tok/s @512 / ~5.6 @4.5k). `glm53_tbv_ar2: 1` is now the default;
   prefill-sized collectives still ride RCCL-over-IB on the same rail.
1. **MLA attention backend on gfx1151.** Upstream `glm5next` rides the shared
   MLA infrastructure (DeepSeek-V3.2-era `MLAModules`, `deep_gemm` page-size
   tables). DS4 needed a heavy rewrite of `rocm_aiter_mla_sparse.py` for its
   indexer; GLM-5.3's Triton kpool path is different code and we serve with
   `VLLM_ROCM_USE_AITER=1` (config knob `glm53_aiter`) because the sparse
   indexer's ROCm path requires aiter. AITER MoE kernels do not support
   gfx1151, so `VLLM_ROCM_USE_AITER_MOE=0` keeps MoE on Triton.
2. **No disk KV tier.** DS4's `fs_lru` tier is DS4-specific and not carried.
   `glm53_max_ctx` starts at 32 K for the ~4 GiB KV pin; raise as memory allows.
3. **No cudagraphs.** `--enforce-eager` (matches the validated DS4 profile).
4. **MTP — FIXED** (`vsh-mtp-ropefree-triton-sparse.patch`, §1.5). The
   original crash: with `glm53_mtp_tokens > 0` the worker SIGABRTed on the
   first speculative batch (no traceback; last worker log was aiter's
   `module_mla_metadata` import, then the JIT build of `module_mla_asm`,
   whose `get_heuristic_kernel_mla` aborts for GLM's rope-free BF16 layout).
   Isolated repro: calling `aiter.mla.mla_decode_fwd` with the rope-free
   512-dim MQA shape aborts with `asm_mla.cu:193 cannot get heuristic
   kernel!`. The patch routes MTP-shaped batches to the Triton ragged
   kernel; validated end-to-end — engine warmup survives, greedy quality is
   correct, ~92% draft-token acceptance (197/213), and single-stream
   throughput at 4.5k ctx goes **~2.4 → ~5.5 tok/s (~2.3×)** (~6.7 → ~7.6
   tok/s at 512 ctx). `glm53_mtp_tokens: 3` is now the validated default
   everywhere.
5. **RCCL CQ warm-up** not pre-applied (see §1). Symptom to watch: EngineCore
   SIGSEGV during weight load on a memory-tight box.
6. **NVFP4/EXL3 4-bit formats** seen in the wild for GLM-5.3 are NVIDIA-only;
   AWQ/compressed-tensors is the ROCm-compatible format used here.

## 6. Attribution

RDMA natives and the `tbv/` kernel kit come from
[`AlexKGwyn/ds4-vllm`](https://github.com/AlexKGwyn/ds4-vllm) (Apache-2.0 for
the original work; kernel modules GPL-2.0 via hellas-ai/thunderbolt-ibverbs).
The all-reduce hook is a rebased, renamed derivative of the same project's
`DS4_TBV_AR2` hook. See THIRD_PARTY_NOTICES.md.
