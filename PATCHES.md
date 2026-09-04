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
   The reference rig runs `glm53_max_ctx: 131072` with an 8 GiB KV pin, which
   measures 526,083 tokens = **4.01x concurrency at 128 K** and leaves ~20 GiB
   free per box. Note the pool does *not* scale linearly with context: the KDA
   recurrent state is a large per-*request* cost, so it amortises over far more
   blocks at 128 K (57 attention blocks per request vs 15 at 32 K) — 8 GiB at
   128 K buys better concurrency than 4 GiB did at 32 K. Sizing `max_ctx` past
   this is bounded by **prefill time, not memory**: prefill is a flat
   156 tok/s (measured, linear — the DSA sparse attention has no quadratic
   term), so 128 K is ~14 min TTFT cold and 256 K would be ~28 min. Raising
   `max_model_len` costs almost nothing at decode: the indexer buffers scale
   with it, but the step only moved 208 -> 224 ms at 512 ctx and was unchanged
   at 4 K/8 K.
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

## 6. Tuned fused-MoE tile configs (gfx1151) — the decode bottleneck

`host/moe-configs/E=288,N=1024,device_name=AMD_Radeon_8060S,dtype=int4_w4a16.json`

§1 classified ds4-vllm's "MoE/GEMM tuning" as **not needed**, on the grounds
that its MXFP4 tuning was written for DS4's expert layout and GLM runs AWQ
W4A16 — "different kernels entirely". That reasoning is right and the
conclusion was wrong: the DS4 configs do not transfer, but GLM still needs its
**own** MoE tuning, and not having it was costing ~1.6x on decode.

**What the profiler showed.** Torch traces from both ranks of a live decode
(short context, MTP=3, so a 4-token verify batch) put the step at ~360 ms with
the GPU ~90% busy, and `fused_moe_kernel_gptq_awq` at ~230 ms of it — 64% of
the step. Everything else (dense BF16 GEMVs ~22 ms, the tbv_ar2 all-reduces
~26 ms, sparse MLA + kpool indexer ~15 ms, KDA, mHC) is small by comparison.
Both ranks measured within 0.1 ms of each other, so this is not a TP imbalance.

**Why it was slow.** For `int4_w4a16` vLLM never consults the normal tile
heuristic for N/K: `get_moe_wna16_block_config()` hardcodes
`BLOCK_SIZE_N=64, BLOCK_SIZE_K=32` (32/64 at batch 1). `BLOCK_SIZE_K=32`
loads **16 bytes** of packed int4 weight per row per K-step, which cannot
saturate the memory system. Measured on the GLM decode shape: **38 GB/s**.

For calibration, the model's own dense BF16 projections (KDA + MLA, via
`wvSplitK`) move ~4.7 GB in ~22 ms on the same GPU = **~214 GB/s**. So the
hardware was delivering roofline; only the MoE kernel was not.

This is also most of the GLM-vs-DS4 gap on this rig. DS4 is block-FP8, which
takes the *other* branch of `get_default_config()` and gets
`BLOCK_SIZE_N=128, BLOCK_SIZE_K=128` — 4x the K tile — for free.

**Why a config file and not a code patch.** The alternatives are closed on
gfx1151: Marlin MoE is `return False` for ROCm
(`check_moe_marlin_supports_config`), the Humming backend is CUDA-only, and
vLLM's small-M `moe_wna16` CUDA kernel cannot be ported because
`csrc/moe/moe_wna16_utils.h` is inline PTX (`lop3.b32`, `prmt.b32`). Triton is
the only MoE backend here, so the lever is its tile shape. vLLM already looks
for a per-GPU config and logs the miss at startup:

```
Using default MoE config. Performance might be sub-optimal! Config file not found at
  .../configs/E=288,N=1024,device_name=AMD_Radeon_8060S,dtype=int4_w4a16.json
```

`vsh-cluster-env.sh` sets `VLLM_TUNED_CONFIG_FOLDER=$HOME/vsh-moe-configs`,
which vLLM checks *before* its built-in `configs/`, so nothing in
site-packages is modified and the file survives container rebuilds.
`host/deploy.sh` installs it on both boxes and verifies the checksums match —
TP runs in lockstep, so an untuned rank would set the pace.

**Measured, on the isolated kernel (E=288, K=4096, N=1024/rank, top-8, g128):**

| tokens M | stock | tuned | speedup | tile |
|---:|---:|---:|---:|---|
| 1 | 1.18 ms | 0.70 ms | 1.69x | BM=32 BN=32 BK=128 w4 s2 |
| 2 | 4.15 ms | 1.21 ms | 3.44x | BM=32 BN=32 BK=64 w2 s2 |
| 4 (MTP verify) | 5.43 ms | 2.04 ms | 2.67x | BM=32 BN=32 BK=64 w2 s2 |
| 8 | 6.84 ms | 3.43 ms | 2.00x | BM=16 BN=32 BK=64 w2 s2 |
| 16-512 (prefill) | 9.5-27.5 ms | 5.9-20.3 ms | 1.35-1.66x | BM=16 BN=32 BK=64 w2 s2 |

A narrow `BLOCK_SIZE_N=32` matters as much as the wider K: the BN=64/128 +
BK=128 shapes that look natural for a GEMM are **3-6x slower** here.

**Measured, end to end** (streamed, so TTFT is excluded from the decode
figure; matched at the same 47.4% MTP acceptance / 2.52 tokens per step):

| | before | after |
|---|---:|---:|
| decode step (512 / 4k / 8k ctx) | 359 / 359 / 360 ms | 213 / 225 / 230 ms |
| decode throughput @ 4k ctx | 7.02 tok/s | 11.18 tok/s |

Decode step time is flat across 512-8192 context, which is what you expect
when the step is dominated by a context-independent MoE.

**Numerics are unchanged**: the tuned tiles produce bit-identical output to
the stock ones (max abs diff 0.0 at M=1/4/8), because `BLOCK_SIZE_K` only
chunks a sequential fp32 accumulation. Greedy text does still vary run to run
on this cluster, but that predates this change — the same server queried twice
also diverges.

### Round 2: the knobs the first sweep missed — and where this stops

Round 1 swept `BLOCK_SIZE_M/N/K`, `num_warps`, `num_stages`. It never touched
`waves_per_eu` (a ROCm occupancy hint; vLLM's own shipped 8060S int4 config
uses it) or `SPLIT_K` (splits the K = 4096 reduction across blocks). A second
sweep over those, on top of each round-1 winner:

| M | round 1 | round 2 | gain | added |
|---:|---:|---:|---:|---|
| 1 | 0.741 ms (70 GB/s) | 0.630 ms | **1.18x** | `waves_per_eu=1 SPLIT_K=4 GROUP_SIZE_M=4` |
| 4 | 2.177 ms (95 GB/s) | 2.086 ms | 1.04x | `SPLIT_K=4` |
| 8 | 3.457 ms (120 GB/s) | 3.309 ms | 1.04x | `waves_per_eu=1 SPLIT_K=2 GROUP_SIZE_M=4` |

Still bit-identical to round 1 and still deterministic (max abs diff 0.0 at
M = 1/4/8, repeated runs `torch.equal`), so these are free. Applied for
M = 1/4/8 only; 16-512 keep the round-1 tiles because round 2 did not test the
prefill shapes.

**But it does not show up end to end.** M = 4 is the shape MTP decode actually
uses, and 4% of an 84 ms/step MoE is ~3 ms of a ~225 ms step — under the
±10 ms run-to-run spread. Measured after applying: 231.5 / 229.1 / 231.7
ms/step at 4 K, against 225.5 before. That is noise, not a regression, and not
a win either.

The useful conclusion is the negative one: **tile and occupancy tuning is
exhausted.** M = 4 sits at ~95 GB/s while the model's own dense BF16 GEMVs
reach ~214 GB/s on this GPU, and no combination of the seven knobs Triton
exposes closes that. The remaining ~2x needs a different kernel — a hand
int4 grouped-GEMV for the `M*topk` rows-spread-over-experts shape — not
different parameters for this one. Do not re-run the sweep.

### Re-tuning

`host/moe-configs/` is specific to (num_experts, intermediate-per-rank, GPU,
quant dtype). A different TP size, a different checkpoint, or a different GPU
needs a fresh sweep of `BLOCK_SIZE_M/N/K`, `num_warps`, `num_stages` against
`fused_experts_op(..., use_int4_w4a16=True)`, keeping every entry's `SPLIT_K`
(the Triton kernel takes it as a required argument, so a config missing it
raises `TypeError` at the first MoE call).

## 7. Tool calling and reasoning separation (`glm53_tool_parsing`)

Coding harnesses need OpenAI-style `tool_calls` rather than raw text, which
means three flags:

```
--enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm47
```

**Why `glm47` for a 5.3 model.** GLM-5.3's `chat_template.jinja` emits the
GLM-4.5/4.7 shapes verbatim —

```
<think>...</think>
<tool_call>fn<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>
```

— and `vllm/parser/glm47_moe.py` keys off exactly those markers
(`TOOL_CALL_START`, `ARG_KEY_START`, `THINK_START`, …). vLLM registers the
pair under **both** `glm45` and `glm47`, and they resolve to the same classes
(`Glm47MoeModelToolParser` / `Glm47MoeParserReasoningAdapter`), so mixing the
two names works but is pointless; this repo uses `glm47` for both because that
is the implementation name and the parser's own
`structural_tag_model = "glm_4_7"`. There is no `glm5*` parser in this build.

**Thinking is always on.** The chat template opens the assistant turn with
`<|assistant|><think>` (line 256) and defaults `Reasoning Effort: Max`, so
generation starts *inside* a think block. The parser handles this: with no
`thinking` / `enable_thinking` chat kwarg it defaults `thinking_enabled=True`
and starts in `ParserState.REASONING`.

**Gotcha — the field is `reasoning`, not `reasoning_content`.** On this build
the chat response carries the think block in `message.reasoning`, with the
token count under `usage.completion_tokens_details.reasoning_tokens`. A client
that looks for `reasoning_content` will silently see nothing. Message keys are
`annotations, audio, content, function_call, reasoning, refusal, role`.

Validated end to end on the rig: non-streaming tool call returns
`finish_reason: tool_calls` with valid-JSON arguments; the streaming path emits
the name and argument deltas and reassembles to valid JSON; a reasoning-heavy
prompt puts 1,579 chars in `reasoning` and a clean 604-char answer in
`content` with no `<think>` leakage. Decode is unaffected (206.8 / 233.7
ms/step at 512 / 4k, unchanged from §6).

## 8. Prefill chunk size (`glm53_max_batched`)

Once the MoE was fixed (§6), decode stopped being the user-visible cost.
Prefill is a flat ~151 tok/s regardless of prompt length (linear, no quadratic
term -- the DSA sparse attention works), so on a 20 K-context agent turn
prefill is ~76% of the wall time and decode ~24%.

`--max-num-batched-tokens` was pinned at 512 with the note "NOT larger: the
sparse-attention indexer workspace scales with batch x context". That was
over-cautious. The large indexer allocation is the *prefill* workspace,
`get_max_prefill_buffer_size() = max_model_len * 40` entries x 132 bytes,
which does not depend on this knob at all. What does scale with it is
`topk_indices_buffer` (`max_num_batched_tokens x 2176 x 4` bytes = 4.5 MB at
512, 36 MB at 4096) -- immaterial against ~19 GiB free. vLLM was in fact
already asking for the opposite:

```
max_num_scheduled_tokens is set to 512 based on the speculative decoding
settings. This may lead to suboptimal performance. Consider increasing
max_num_batched_tokens to accommodate the additional draft token slots...
```

Measured at `max_ctx: 131072`, uncacheable prompts (nonce-prefixed so the
2304-token prefix-cache blocks never hit), warm -- the first call at any new
chunk shape pays Triton JIT and reads ~80-107 tok/s, which is not the
steady-state number:

| prompt | 512 | 2048 | 4096 |
|---:|---:|---:|---:|
| 1.8 K | 150 | 205 | 206 |
| 7 K | 152 | 189 | 195 |
| 14 K | 151 | 190 | 197 |
| 21 K | 157 | 188 | 196 |
| 34 K | - | - | 195 |

TTFT for a 4 K prompt: **18.05 -> 14.79 -> 13.60 s**. Projected cold 128 K
prefill: 13.9 -> 11.8 -> ~10 min. Decode is unaffected (225-240 ms/step
across all three, inside run-to-run noise). Memory cost of 512 -> 4096 is
0.6 GiB (19.6 -> 19.0 GiB free per box).

Returns flatten hard after 2048 (+3-5% for 2048 -> 4096), so ~200 tok/s looks
like the next wall; the remaining per-chunk cost is the ~87 latency-bound
all-reduces, which is §5.0s territory rather than this knobs.

## 9. RCCL protocol: unpin NCCL_PROTO (prefill all-reduce)

After §8 raised the chunk, a prefill profile showed the collectives had become
the dominant cost, not the MoE:

| prefill kernel (2 chunks, 2304 + 3599 tok) | self CUDA | share |
|---|---:|---:|
| `vllm::all_reduce` / `ncclDevKernel_Generic_4` | 16.09 s | **51%** |
| `fused_moe_kernel_gptq_awq` | 6.54 s | 21% |
| `_sparse_attn_prefill_ragged_kernel` | 1.97 s | 6% |
| `aten::mm` | 1.59 s | 5% |

224 collectives at **71.5 ms each**, moving ~29 MB per call: **0.41 GB/s on a
40 Gbps (~5 GB/s) rail**, about 8% of link.

The cause was `NCCL_PROTO=LL` in `vsh-cluster-env.sh`. LL spends 8 bytes of
flag per 8 bytes of payload and exists for tiny latency-bound collectives.
That pin was reasonable when RCCL carried the decode all-reduces — but since
tbv_ar2 took over everything <= 1 MiB (§5.0), **RCCL only ever sees the big
prefill-sized collectives**, which is precisely where LL is worst. The pin had
outlived its reason.

Fix: stop pinning it and let RCCL choose per message size (`NCCL_ALGO=Ring`
stays; 2 ranks). Set `VSH_NCCL_PROTO=LL` to restore the old behaviour. Safe in
all cases: if tbv_ar2 ever fails open, the RCCL selection picks LL for the
small sizes anyway.

Measured prefill (uncacheable nonce prompts, warm):

| prompt | 512 chunk + LL | 4096 chunk + LL | 4096 chunk + auto |
|---:|---:|---:|---:|
| 1.8 K | 150 | 206 | **277** |
| 7 K | 152 | 195 | **276** |
| 14 K | 151 | 197 | 247 |
| 21 K | 157 | 196 | **280** |
| 34 K | – | 195 | **277** |

TTFT for a 4 K prompt: 18.05 → 13.60 → **9.18 s**. For 512 tokens:
2.11 → **1.34 s**. Projected cold 128 K prefill: 13.9 → 9.8 → **7.8 min**.
Decode is unchanged (216–239 ms/step). §8 + §9 together take prefill from
~151 to ~277 tok/s, **+83%**.

The original plan here was to raise `TBV2_MAX_BYTES` (1 MiB) so tbv_ar2 could
carry prefill collectives too. That is now much less attractive: the tensors
are ~29 MB, over even the 16 MiB the doorbell can encode (`nbytes` occupies
24 bits), so it would need chunking *and* a grid-stride rewrite of
`tbv2_wait_add_kernel`, which is launched `dim3(1), dim3(1024)` — a single
workgroup. Re-profile before spending that effort.

## 10. RCCL transport: sockets beat the RoCE rail at every size

This one cuts against the premise of the repo, so here is the measurement.

The netdev negotiates **20 Gbps**, not the 40 Gbps quoted elsewhere in these
docs (`ethtool thunderbolt0` → `Speed: 20000Mb/s`), so the ceiling is ~2.5
GB/s. Raw `ib_write_bw -d usb4_rdma0 -x 1 -q 4` over the rail managed
**3.85 Gb/s = 0.48 GB/s**, i.e. the RoCE driver delivers under a fifth of the
link.

A standalone 2-rank RCCL all-reduce over the same rail, run inside the serving
containers with the real cluster env (`ar_bench`, bf16, algbw = busbw at
n = 2):

| message | IB (`usb4_rdma0`) | sockets (`thunderbolt0`) |
|---:|---:|---:|
| 16 KB | 0.40 ms | **0.11 ms** |
| 64 KB | 0.42 ms | **0.18 ms** |
| 256 KB | 0.51 ms | **0.29 ms** |
| 1 MB | 0.83 GB/s | **1.28 GB/s** |
| 8 MB | 0.93 GB/s | **1.46 GB/s** |
| 32 MB | 0.97 GB/s | **1.48 GB/s** |
| 64 MB | 0.47 GB/s | **1.48 GB/s** |

Sockets win at **every** size — 3.6x on 16 KB latency, ~1.5x on bandwidth —
and stay flat at 64 MB where the IB path collapses. There is no crossover to
trade off.

End to end, three server configurations (single stream, tuned MoE, 4096 chunk,
`NCCL_PROTO` auto). Prefill is the mean of five uncacheable warm prompts;
decode is ms/step, which is the acceptance-independent figure:

| transport | prefill tok/s | decode @512 | decode @4k | TTFT @4k |
|---|---:|---:|---:|---:|
| `rdma` — IB + tbv_ar2 | 271 | 216.4 | 238.8 | 9.18 s |
| `hybrid` — sockets + tbv_ar2 | 305 | 213.1 | 227.7 | **8.37 s** |
| `tcp` — sockets only | **314** | 218.7 | 227.3 | **8.37 s** |

Two conclusions:

1. **Disabling IB for RCCL is a clear win** (+13% prefill, −0.8 s TTFT at 4 K,
   decode a touch better). `NCCL_IB_DISABLE=1` is the whole change.
2. **tbv_ar2 is neutral once RCCL is on sockets.** `hybrid` and `tcp` are
   inside run-to-run noise of each other on both axes. tbv_ar2 was worth ~150
   µs/op against *RCCL-over-IB* (§5.0); against RCCL-over-sockets, which does
   16 KB in 0.11 ms, it no longer has an edge to add.

`transport: hybrid` is the shipped default: it takes the socket win while
keeping the RDMA rail in the decode path, so the tbv stack is not wasted and
§5.0 stays exercised. `transport: tcp` measures the same within noise, which
means **a fresh rig can skip the tbv kernel-module build entirely** — no
Secure Boot changes, no coordinated reboots, no matched module sets — and give
up nothing measurable. That is a large reduction in bring-up risk.

Caveats worth respecting before ripping anything out: all of this is
**single-stream**; RDMA may still matter at concurrency, where many small
collectives overlap and CPU-side socket handling could become the limit. The
decode differences between the three are within noise, so only the prefill
number is firmly established. And `ib_write_bw` measures the driver, not the
fabric — a better RoCE driver could still change the picture.

## 11. CUDA graphs — attempted, blocked (`glm53_enforce_eager`)

`--enforce-eager` has been on since bring-up because it matched the validated
DS4 profile (§5.3). With the MoE fixed the decode step is ~200 ms and a
decode-only profile shows ~32 ms/step (≈16%) outside the model-execute marker
— launch and CPU work that CUDA graphs would be expected to recover. So it was
worth an attempt. It does not work yet, and the failure is specific enough to
be worth recording.

Three gates, in the order they appear:

1. **`max_num_seqs`.** Capture refuses to start:
   `ValueError: max_num_seqs (1024) exceeds available Mamba cache blocks (293).
   Each decode sequence requires one Mamba cache block, so CUDA graph capture
   cannot proceed.` The default 1024 was meaningless here anyway (§5.2 gives
   4.01x concurrency at 128 K), so `glm53_max_seqs: 256` is now set
   independently of this experiment — it is free (decode 201.6/225.5 ms at
   512/4k, prefill 310-340 tok/s, i.e. unchanged or marginally better).

2. **Not torch-compiled.** vLLM logs
   `torch.compile is turned on, but the model ... does not support it`, and
   then piecewise capture refuses:
   `piecewise CUDA graphs (cudagraph_mode=FULL_AND_PIECEWISE) unavailable,
   model is not torch-compiled and breakable CUDA graph is off. Set
   VLLM_USE_BREAKABLE_CUDAGRAPH=1 or cudagraph_mode=NONE/FULL.`
   `VLLM_USE_BREAKABLE_CUDAGRAPH=1` is the right answer — it is exactly what
   the `@eager_break_during_capture` decorators on the kpool indexer and the
   KDA `_forward` exist for.

3. **Capture then breaks on rank 1.** With breakable capture on, rank 0
   succeeds — `Graph capturing finished in 177 secs, took 11.43 GiB` — and
   rank 1 dies inside a Triton launch hook:
   `SystemError: <built-in method __reversed__ of list object> returned a
   result with an exception set`, raised from `triton/knobs.py:432` under
   `triton/backends/amd/driver.py`. Engine core init then fails. This is the
   bug class §1 flagged when it listed DS4's `breakable_cudagraph.py`
   stream-sync fix as a candidate we had not ported.

Even if (3) were fixed, **11.43 GiB of graph memory per rank** is not
affordable against the ~18.6 GiB free that the 8 GiB KV pin leaves — it would
have to be traded against KV, i.e. against context.

So: `glm53_enforce_eager: 1` stays the default, `glm53_max_seqs: 256` is kept
on its own merits, and the knob is in place for whoever retries this after
porting the DS4 stream-sync fix. Set `glm53_enforce_eager: 0` to reproduce.

## 12. What is left: the unquantized BF16 weights

The AWQ checkpoint quantizes the routed experts and nothing else. Its ignore
list is `lm_head`, `embed_tokens`, `visual.*`, **`self_attn.*`**,
`shared_experts.*`, `mlp.gate`, `_hc.*`, `eh_proj`, `layers.(0|1|2).mlp.*` and
**`layers.45.*`**. Two of those are expensive at decode.

**Attention + KDA projections (`self_attn.*`) — ~48 ms/step, 24% of decode.**
Note `self_attn` covers the KDA layers too: `Glm5NextDecoderLayer` assigns
`self.self_attn = Glm5NextLinearAttention(...)` for the 34 linear-attention
layers, and `kda.py` forces `quant_config=None` for them anyway ("KDA
projections remain BF16 because fp8 checkpoints omit their scales"). So per
rank each step reads roughly:

| | per layer | layers | total |
|---|---:|---:|---:|
| KDA `in_proj_qkvbfg_a` [12576, 4096] | 103 MB | 34 | 3.5 GB |
| KDA `o_proj` [4096, 4096] | 34 MB | 34 | 1.1 GB |
| MLA `q_b` / `kv_b` / `o_proj` / `fused_qkv_a` | ~126 MB | 11 | 1.4 GB |

~6.1 GB/rank of BF16, every step. At the ~214 GB/s these `wvSplitK` skinny
GEMMs actually achieve that is ~29 ms, and 48 ms is measured — so the kernels
are within ~1.7x of roofline and there is little left in the *kernel*. The
prize is in the *format*: at int4 the same weights would be ~1.5 GB and ~7 ms.
**Potentially ~40 ms/step, ~20% of decode.** Needs a re-quantized checkpoint
with `self_attn` included, and attention is the most quality-sensitive place to
put 4 bits, so it needs real calibration and an eval — not a blind RTN pass.

**MTP block (`layers.45.*`) — ~9 ms/step plus ~7 GB/rank.** Confirmed
unquantized: the safetensors index has 889 tensors under `layers.45.` and
**zero** `weight_packed`. Its 288 experts in BF16 are ~14.5 GB (~7 GB/rank),
which is why the KV pin and `max_model_len` are as tight as they are. Freeing
that is worth more as *memory* (a much larger KV pool, or headroom for §12's
attention quantization) than as the 9 ms/step of drafter MoE.

This one is lower risk than the attention weights: layer 45 only produces
*draft* tokens, and the target model verifies every one. If a cheap RTN int4
pass degrades the drafts, MTP acceptance drops and throughput regresses, but
**output correctness is unaffected** — a safe thing to try and revert. The work
is emitting `weight_packed`/`weight_scale` in compressed-tensors
pack-quantized layout for those tensors and dropping `layers.45.*` from the
ignore list.

Either way this is checkpoint work, not serving work. The cheapest route is
asking `wtdcode` for a variant that quantizes layer 45 (and optionally the MLA
projections) rather than re-deriving one from the 643 GB BF16 original.
## 13. Attribution

RDMA natives and the `tbv/` kernel kit come from
[`AlexKGwyn/ds4-vllm`](https://github.com/AlexKGwyn/ds4-vllm) (Apache-2.0 for
the original work; kernel modules GPL-2.0 via hellas-ai/thunderbolt-ibverbs).
The all-reduce hook is a rebased, renamed derivative of the same project's
`DS4_TBV_AR2` hook. See THIRD_PARTY_NOTICES.md.
