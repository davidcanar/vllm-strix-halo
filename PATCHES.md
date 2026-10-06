# PATCHES.md — what this repo patches, and why (review of `AlexKGwyn/ds4-vllm`)

> ## ⚠️ Correctness warnings (2026-09-04)
>
> Three defects were found after the performance work below was written, and
> some of the claims in this file were wrong. Read
> **[PENDINGWORK.md](PENDINGWORK.md)** before trusting any quality claim here.
>
> 1. **MTP corrupts structured output on gfx1151.** `glm53_mtp_tokens: 0` is
>    now the recommended setting, not `3`. See §1 item 4.
> 2. **Greedy decoding is not reproducible on this rig** at `temperature 0` —
>    5 identical requests give 5 different completions, at every prompt length
>    down to 244 tokens. Reported upstream:
>    [vllm#54521](https://github.com/vllm-project/vllm/issues/54521)
>    ([our data](https://github.com/vllm-project/vllm/issues/54521#issuecomment-5545047644)).
> 3. **Tool calling collapses above ~10k prompt tokens** — the model stops
>    emitting `</tool_call>`, so agent harnesses stall silently on HTTP 200.
>
> The *timing* results below (prefill tok/s, ms/step, TTFT) are unaffected —
> they do not depend on which token is emitted. The *quality* claims are.

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
   and RCCL logs `Using network IB` (`usb4_rdma0`, RoCE, 20 Gbps negotiated
   — §10; and RCCL no longer uses it by default).
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
   tok/s at 512 ctx).

   > ⚠️ **RETRACTED: MTP is not a safe default.** "Greedy quality is correct"
   > above was measured only as readable prose, which is exactly what this
   > failure mode preserves. With MTP on, **structured** output corrupts:
   > requests die at exactly 12 output tokens (3 MTP steps at k=3) emitting
   > 13-character tool-call arguments, and prompts as small as ~4k tokens
   > return empty responses. Disabling MTP removed that failure mode
   > entirely — 10/10 valid tool calls where 8/10 had failed. The upstream
   > recipe for this model states MTP is **not supported on ROCm**
   > (<https://recipes.vllm.ai/zai-org/GLM-5.3-Flash>), and we run it via the
   > gfx1151 patch above anyway.
   >
   > **Use `glm53_mtp_tokens: 0`** until this is fixed, and accept ~3x slower
   > decode. Related upstream: spec decoding altering greedy output
   > ([#54928](https://github.com/vllm-project/vllm/issues/54928),
   > [#53436](https://github.com/vllm-project/vllm/issues/53436)) and MTP
   > repetition collapse until `max_tokens`
   > ([#55357](https://github.com/vllm-project/vllm/issues/55357)).
   > Full detail in [PENDINGWORK.md](PENDINGWORK.md).
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
| decode step (~155 / ~1.3 K / ~2.6 K tok prompts) | 359 / 359 / 360 ms | 213 / 225 / 230 ms |
| decode throughput (~1.3 K tok prompt) | 7.02 tok/s | 11.18 tok/s |

Decode step time is flat across context - 523 to 32 830 tokens in the
re-measurement below - which is what you expect when the step is dominated by
a context-independent MoE.

> **Context labels.** The decode figures in this section came from a harness
> that sized prompts as `round(target / 230)` paragraphs when a paragraph is
> in fact 71 tokens, so its "512 / 4k / 8k" prompts were really ~155 / ~1.3 K /
> ~2.6 K tokens. The before/after comparisons are unaffected (same prompts on
> both sides) but the labels were. Re-measured at true token counts, decode
> step time is flat over a much wider range than originally claimed:
> **200.6 / 221.3 / 215.7 / 230.8 ms** at **523 / 2 084 / 8 190 / 32 830**
> tokens, with prefill holding 284-334 tok/s across the same span.

**Numerics are unchanged**: the tuned tiles produce bit-identical output to
the stock ones (max abs diff 0.0 at M=1/4/8), because `BLOCK_SIZE_K` only
chunks a sequential fp32 accumulation.

> ⚠️ **CORRECTION.** This section originally went on to say that greedy text
> "does vary run to run on this cluster, but that predates this change", and
> used that to wave the divergence away. The observation was right and the
> conclusion was wrong: **it is a real correctness bug**, not ambient noise.
> Five byte-identical `temperature 0` requests return five different
> completions, at every prompt length tested down to 244 tokens, and some runs
> degenerate into repetition loops. Reported upstream at
> [vllm#54521](https://github.com/vllm-project/vllm/issues/54521); our
> gfx1151 data is
> [here](https://github.com/vllm-project/vllm/issues/54521#issuecomment-5545047644).
>
> The tuned tiles are **not** the cause, and that is now properly established
> rather than assumed. The original check used `E=32` and two repeats; redone
> at the real `E=288, K=4096, N=1024, topk=8` with **20 identical calls** per
> config, every call is bitwise identical (1 distinct result, max drift 0.0)
> for M=1/4/8 under both `SPLIT_K=1` and `SPLIT_K=4`. `SPLIT_K` partitions the
> K reduction, so it was the obvious suspect; it is excluded.

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
ms/step at ~1.3 K tok, against 225.5 before. That is noise, not a regression,
and not a win either.

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
ms/step at ~155 / ~1.3 K tok prompts, unchanged from §6).

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

TTFT for a 2 841-token prompt: **18.05 -> 14.79 -> 13.60 s**. Projected cold 128 K
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
20 Gbps (~2.5 GB/s) rail**, about 16% of link.

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

TTFT for a 2 841-token prompt: 18.05 → 13.60 → **9.18 s**. For a short
(~150-token) prompt: 2.11 → **1.34 s** (a real 523-token prompt measures
1.84 s cold). Projected cold 128 K prefill: 13.9 → 9.8 → **7.8 min**.
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

The netdev negotiates **20 Gbps**, not the 40 Gbps this repo originally
assumed (`ethtool thunderbolt0` → `Speed: 20000Mb/s`), so the ceiling is ~2.5
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

| transport | prefill tok/s | decode, ~155 tok | decode, ~1.3 K tok | TTFT, 2 841 tok |
|---|---:|---:|---:|---:|
| `rdma` — IB + tbv_ar2 | 271 | 216.4 | 238.8 | 9.18 s |
| `hybrid` — sockets + tbv_ar2 | 305 | 213.1 | 227.7 | **8.37 s** |
| `tcp` — sockets only | **314** | 218.7 | 227.3 | **8.37 s** |

Two conclusions:

1. **Disabling IB for RCCL is a clear win** (+13% prefill, −0.8 s TTFT on a
   2 841-token prompt, decode a touch better). `NCCL_IB_DISABLE=1` is the whole change.
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
   ~155/~1.3 K tok, prefill 310-340 tok/s, i.e. unchanged or marginally better).

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

### 2026-10-02 retry — replay-time root cause found; still not shippable

Reproduced with the knob (`glm53_enforce_eager: 0`) under reduced pressure
(ctx 32 K, KV 8 GiB, `--cudagraph-capture-sizes 1..32` — the cap lives in
`vsh-manual-serve.sh` and only applies with eager off). Findings, in order:

1. **The 2026-09-28 "wedge" had two stacked causes.** With eager off the box
   runs the JIT-warmup/compile phase for the first time (eager disables it),
   which pushed host RAM to 0 + swap during the full 35-size capture — a
   thrash that mimics a deadlock. Separately, the large end of the capture
   list deadlocked rank 1 inside a KDA Triton launch (py-spy: stuck in
   `triton/backends/amd/driver.py` launching `layer_norm_gated_fwd`) while
   rank 0 spun in the MoE all-reduce. With ctx 32 K / 7 capture sizes both
   ranks capture cleanly (~2.7–3.3 GiB of graph memory) and the API comes up.

2. **The first real request then deadlocks at replay — and it is NOT the MTP
   drafter** (the old attribution was wrong). py-spy on rank 0 shows it
   spinning in `prepare_chunk_indices`
   (`vllm/third_party/flash_linear_attention/ops/index.py:28`), whose
   `.tolist()` is a device→host sync *inside the KDA eager-break segment at
   replay* (the file's own comment: "This will be fixed by
   vllm-project/vllm/pull/51540" — still open, and main still has the sync).
   With MTP on, every decode step routes the non-spec rows through
   `chunk_kda_with_fused_gate` → `prepare_chunk_indices` → a per-step sync
   that deadlocks against the peer rank's in-flight TP collective.

3. **With MTP off, graphs work end to end and compute correctly**: the 12 K
   needle probe PASSES under graphed decode; steady decode is 8.8 t/s
   (engine stat) vs ~7.6 eager spec-off — the ~+15 % the §11 profile
   predicted (32 ms launch overhead of a ~200 ms step), not more. But it
   loses to production eager+MTP (~13–14 t/s), so it is a net regression.

**Verdict**: production stays `glm53_enforce_eager: 1`. Graphs become worth
revisiting only when the #51540-class sync removal is extended to the
MTP-decode chunked-KDA path (or upstream ships it; the PR is scoped to
prefill only), or if a future engine ends up running without MTP anyway.

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
### 2026-10-02 profile — the requant gate, measured (`torch` profiler, MTP k=3)

Fresh 96-step decode window (191.6 ms/step; traces in `~/glmprof/` on both
boxes, `glm53_profiler_dir` knob in `vsh-config.yaml`):

| kernel class | ms/step | share |
|---|---:|---:|
| routed experts `fused_moe_kernel_gptq_awq` (int4) | 42.7 | 22% |
| **bf16 GEMMs (`wvSplitK_hf_sml` family + hipBLASLt `Cijk_*`)** | **27.7** | **14.5%** |
| MLA sparse attention core | 6.6 | 3.5% |
| aiter fp8 batched GEMM (indexer scoring) | 5.7 | 3.0% |
| fused index/mask triton (`arange/eq/bitwise_not`) | 5.7 | 3.0% |
| launch-path overhead (`hipPointerGetAttribute` 8.7 GPU-serialized + `hipModuleLaunchKernel`) | ~12 | ~6% |
| collectives (`odl2_wait_add` + nccl all-gather) | 3.7 | 1.9% |
| mhc fused ops | 2.1 | 1.1% |
| KDA recurrent core | 0.6 | 0.3% |

Cross-check: the bf16 GEMM time (27.7 ms) moving ~6.1 GB/rank implies
~220 GB/s — **the bf16 kernels are at roofline** (matches the §12 analysis
above; the kernels are not pathological, the *format* is). Consequences:

- **fp8 W8A8 on `self_attn.*`** (the antirez/ds4 analogue) halves those
  bytes: ceiling ≈ −14 ms/step ≈ **+8% decode** (14.6 → ~15.8 t/s). Below
  the >20–25% gate; not worth the surgery on speed alone.
- **int4 on `self_attn.*`** stays the bigger prize (§12's ~+20%) but is the
  highest quality risk and needs calibration + eval, not blind RTN.
- **The disk constraint**: any full-checkpoint requant writes ~172 GB and
  box1 has 113 GB free. Feasible only after freeing ~60 GB (dangling podman
  images ≈ 45 GB + the official-nightly images) or by building the variant
  on box2 (199 GB free) first. A `layers.45`-only patch checkpoint (symlink
  unchanged shards + new layer-45 shards, ~4 GB) fits today and is the
  low-risk memory lever (frees ~7 GB/rank for KV), not a decode lever.

**Verdict: stand down on the requant-for-speed.** The remaining honest
levers are launch overhead (~6%, needs the #51540-class graphs fix) and the
already-int4 MoE. Keep the `glm53_profiler_dir` knob — it is inert until
`POST /start_profile` and made this measurement a 5-minute job.

### 2026-10-02 fp8 requant — BUILT but blocked at kernel selection on gfx1151

The antirez-analogue experiment was carried out end to end up to the point
where the silicon says no:

- `requant-fp8attn.py` builds `~/models/GLM-5.3-Flash-AWQ-W4A16-fp8attn` as a
  **shard-patch checkpoint** (symlinked original shards + 3 new shards,
  ~6.2 GB: 12.30 GiB of bf16 `self_attn.*` 2-D linear weights + `lm_head`
  requantized to fp8 e4m3 per-output-channel + scales; the indexer
  projections, conv1d weights and norms stay bf16; `layers.45` stays
  ignored). config.json gains a fp8 W8A8 config_group placed FIRST
  (first-match order beats the `Linear` catch-all; the `RoutedExperts`
  helper keeps the experts on W4A16). Re-runs in ~18 s.
- `kda.py` got an env-gated patch (`VSH_KDA_QUANT`, default off): the
  Glm5NextLinearAttention constructor nulls `vllm_config.quant_config`
  ("fp8 checkpoints omit their scales") — with our checkpoint they exist.
- **Boot fails at kernel selection** for every ScaledMM candidate:
  aiter hipbmm requires CDNA3+ (`get_cdna_version() > 2`; gfx1151 is RDNA),
  the two aiter a8w8 kernels require a *tuned configuration* per (N, K)
  (`a8w8_tuned_gemm.csv` has gfx942 rows only), `ROCmFP8ScaledMMLinearKernel`
  requires CDNA3+/RDNA4, and the three torch `_scaled_mm` variants are
  unsupported on this platform. Enabling
  `VLLM_ROCM_USE_AITER_LINEAR[_HIPBMM]=1` does not help (the CDNA gate).
- The un-blocker is upstream: PRs #59531/#59479 ("gfx1151 W8A8 wvSplitK
  blockscale skinny path") are building exactly this fp8 skinny-GEMM path.
  Once one lands (or a seeded `a8w8_tuned_gemm.csv` provides our shapes),
  the fp8 checkpoint is ready to serve; ceiling remains the measured ~+8%.

The alternative that reuses *proven* kernels today is **W4A16 for
`self_attn.*`** — the exact pack-quantized scheme the routed experts already
use (same wna16 kernels, same loading path, §12's ~+20% estimate) at the
cost of 4-bit attention precision, which the A/B probes would have to
validate.

Env plumbing added for the attempt and kept, default-off: `VSH_KDA_QUANT`,
`VLLM_ROCM_USE_AITER_LINEAR`, `VLLM_ROCM_USE_AITER_LINEAR_HIPBMM` in
`vsh-cluster-restart.sh` ENVPASS (worker-visible at ray start).

## 14. Deterministic MoE-router top-k (`vsh-moe-router-deterministic-topk.patch`)

The stock Python router selects experts with

```python
use_sorted = envs.VLLM_BATCH_INVARIANT          # False by default
topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=use_sorted)[1]
```

which is **not reproducible on this hardware**, in two separate ways. Measured
on a real 740x288 fp32 score tensor dumped from a live layer-21 forward,
20 identical calls:

| arm | distinct raw / 20 | distinct selected sets / 20 |
|---|---:|---:|
| `sort(descending=True, stable=True)[:k]` | **1** | **1** |
| `topk(sorted=False)` (stock default) | **20** | 2 |
| `topk(sorted=True)` | 2 | 2 |

1. `sorted=False` returns the correct top-k **set** in a **different order on
   740/740 rows on every call**. The gathered routing weights are permuted to
   match, so the downstream k-term summation order changes and the MoE output
   differs run to run.
2. `sorted=True` still flips the selected **set** on exact fp32 ties. The real
   tensor has k-boundary ties at rows 40 and 549 (experts 12/202 and 243/250 —
   *different* logits and *different* biases whose sums round to the same
   float), and the varying row is exactly row 40.
   `torch.use_deterministic_algorithms(True)` does **not** help: no raise, no
   warning, still flips.

The patch selects with a stable descending sort, giving a total order of
**value descending, then expert index ascending**. That is not a new
convention: it is what vLLM's own fused CUDA kernel already does
(`moeTopKFuncs.cuh` packs `65535 - idx` into the comparison key; the
multi-group path uses `WarpSelect<..., is_stable=true>`), and what
`test_grouped_topk_single_group_stable_ties` asserts. Only the Python fallback
— which is the path taken on ROCm — fails to match it.

**The boundary is 256 experts, exactly.** `sorted=False` permits any order,
and above 256 columns `torch.topk` switches to a multi-pass path whose order
also varies per call. Measured on the bare op, 20 calls of
`torch.topk(x, k=8, sorted=False)` on `64 x E`: E of 128/250/255/**256** give
1 distinct result and sorted output; E of **257**/258/260/264/272/288/512/
1024/2048 give **20** distinct and unsorted. Same split for any `k >= 4`. So
this bites **models with more than 256 routed experts** - GLM-5.3 has 288,
while DeepSeek-V3/R1's 256 sits just inside the safe side, which is why the
upstream fallback has survived this long unnoticed.

**Effect, measured end to end.** The determinism sweep goes from **5 distinct
of 5 at every prompt length** to **1 of 5** for prompts below `index_topk`, and
all 45 layers become bit-identical across forwards on both TP ranks at 740
tokens. Quality and tool calling are unaffected; prefill stays inside the
284-334 tok/s baseline (340/328/317 at 2K/8K/32K measured after the patch).

**It is not a complete cure, and the boundary is sharp.** Above
`index_topk = 2048` a *separate* defect remains: the DSA sparse-attention
indexer's own top-k. Confirmed with a six-point tap inside layer 3, both ranks,
in one boot — at 719 prompt tokens all six points are deterministic, at 2497
the divergence enters at the attention output (`self.self_attn(...)`, 6 of 6
distinct) with its input bit-identical, and everything downstream cascades.
That one is upstream's [#54521](https://github.com/vllm-project/vllm/issues/54521),
with a fix in flight at
[#55122](https://github.com/vllm-project/vllm/pull/55122) — see
[PENDINGWORK.md](PENDINGWORK.md) §12.

**Cost:** none — slightly faster. 71.6 us vs 97.8 us for the selection on the
real tensor, about 1.1 ms saved per 42-layer prefill, ~2.4 MiB transient. A
full sort is O(E log E) against topk's O(E log k), so the trade could invert at
much larger expert counts.

**Behavioural delta:** confined to exact ties, 1 row in 740 on the real
tensors. Ascending-index is always one of the outcomes `torch.topk` already
produced, so no new selection is introduced — the patch only pins which one.

**Known cross-path inconsistency, pre-existing.** AITER's
`biased_grouped_topk` (reached when `VLLM_ROCM_USE_AITER_MOE=1`, which this
repo sets to 0) is deterministic but its tie-break is opaque and demonstrably
not ascending-index. This patch does not create that divergence, but it does
not unify it either.

`fused_topk_bias_router.py` carries the same `VLLM_BATCH_INVARIANT` gating and
was patched identically in the validated build, but is **not** included in the
shipped patch: it is unreachable for GLM-5.3 (a valid grouping config routes to
`grouped_topk`), it has no behavioural coverage on this model, and it contains
a further unanalysed `topk` site.

Upstream submission is prepared in `upstream/` (`PR-BODY.md` and
`test_grouped_topk_determinism.py`); the bug is not a duplicate of anything
open as of 2026-09-05.

## 15. The 2026-09-27 merge: vLLM pin bump + OdinLink fabric

The vLLM pin moved `8bf39632` (2026-09-03) → `73859fec` (2026-09-27), and the
fabric moved from the tbv/thunderbolt-ibverbs RoCE stack to **OdinLink**
(`odl_tb5`), tracking [AlexKGwyn/ds4-vllm](https://github.com/AlexKGwyn/ds4-vllm)@`6c0550f`.
Base image: `rocm10.0.0-torch2.11.0-vllm0.28.0` → `...-vllm0.30.0` (same ROCm 10
/ torch 2.11 / triton 3.8 toolchain — only the vLLM-inside changed).

### 15.1 Why the pin bump

The 2026-09-05 correctness verdict (README banner) identified defects that are
upstream's. Between 2026-09-18 and 2026-09-26 upstream landed exactly that
class of fixes; the pin includes all of them:

| upstream | what it fixes |
|---|---|
| #58594 | sparse indexer attn topk **backend selection** — the DSA top-k nondeterminism family (#54521) that broke greedy reproducibility above ~2K prompt tokens on ROCm |
| #58454 | **kpool corruption with speculative decoding** (touches the AMD `glm5next/amd/ops/kpool_compress.py`) — the suspected cause of MTP corrupting structured output |
| #58704 | SM90 sparse MLA index_kpool mismatch → corruption via unread query token (NVIDIA path, same family) |
| #57546 | GLM kpool indexer top-k routed through the shared `SparseIndexerTopk` dispatcher |
| #57464-era | ROCm: alias `SparseAttnIndexerKpool.forward_cuda` → `forward_native` (GLM-5.3-Flash boot crash) |
| #58061 | GLM dense MLP layers on the sequence-parallel shard (TP correctness) |
| #57327, #57701, #57162, #58450 | cooperative top-k for small decode batches; −3 GiB indexer workspace; FlashKDA chunked prefill (1.7–3.8×); metadata-op perf (1.6–4.8×) |

Patch-set consequences:

- **`vsh-mtp-ropefree-triton-sparse.patch` dropped — superseded upstream.**
  `_use_rocm_sparse_triton` now returns True for every rope-free BF16 batch
  (multi-token speculative verification rows included); the gfx1151 fallback
  the patch forced is the default behaviour.
- **`vsh-rdma-allreduce.patch` replaced** by `vsh-odl-ar2-allreduce.patch`
  (the OdinLink decode all-reduce hook, `DS4_ODL_AR2=1`) and
  `vsh-odl-mq-dataplane.patch` (the `odl_mq` control-plane data plane in
  `shm_broadcast`, fail-open to zmq), both ported verbatim in intent from
  ds4-vllm's `vllm-upstream.patch` and re-anchored on the new pin. ds4's
  `communication_op.py` eager-break wrapper is deliberately NOT ported: it
  exists to keep the custom AR out of CUDA-graph captures, this stack runs
  `--enforce-eager`, and the wrapper depends on the fork's
  `breakable_cudagraph` machinery which is not upstream.
- **Carried unchanged** (apply clean on the new pin):
  `vsh-moe-router-deterministic-topk` (#55514 is still open upstream),
  `vsh-aiter-gfx1151-gate`, `vsh-fp8-fnuz-mqa` (+ the `pa_mqa_logits` aiter
  overlay; vLLM now probes both aiter layouts, so the overlay only affects
  the old-layout aiter the base ships — harmless either way),
  `vsh-mhc-no-tilelang-gfx1151` (verify whether the upstream MHC TileLang
  migration makes this gate unnecessary — kept for now, it is two lines).

### 15.2 What the OdinLink fabric replaces

The tbv stack was a *patched thunderbolt core* + `thunderbolt_ibverbs` RoCE
driver with the stock `thunderbolt` blacklisted, RCCL speaking IB verbs over
`usb4_rdma0`, and `DS4_TBV_AR2`/tbv_ar2 all-reduces over ibverbs QPs. OdinLink
(`odl_tb5.ko`, from Geramy/OdinLink-Five at `4534f585` + the vendored
`odinlink/odinlink-local.patch`) runs on the **stock kernel**: a
`tb_protocol_handler` on the unmodified thunderbolt core exposing
`/dev/odl_tb5_*` with a stream API. In-image userspace (built by the
Dockerfile's `odinlink-build` stage, same pin + patch as the host driver):

- `libodl_tb5.so.0` — the stream API library
- `librccl_net_odl_tb5.so` — the RCCL net plugin (`NCCL_NET_PLUGIN`),
  host-pointer based (no GPUDirect on gfx1151 anyway), one connection =
  one stream multiplexed over the single TB link. `NCCL_MAX_NCHANNELS=4`
  to keep the per-message syscall count sane.
- `libodl_ar2.so` (from `odl_ar2.hip`, gfx1151) + `odl_ar2.py` — the 2-rank
  decode all-reduce: D2H into a pinned slot → doorbell kernel → CPU progress
  thread does one `STREAM_SEND` ioctl per round (≤64 KiB latency path) →
  wait+add kernel spins on the pinned flag and adds the recv slot (zero-copy
  on UMA). Rendezvous: rank0 → rank1 TCP listener (`ODL2_RANK1_IP`, default
  the worker IP; tbnet stays up for it).
- `odl_mq.py` + the `shm_broadcast` hook — MessageQueue remote data plane
  over odl streams (EngineCore→worker broadcast + response queue), removing
  the per-step zmq-over-TCP round trips. Ungated by design in ds4, fail-open
  to zmq on any init failure; the odl transport profile engages it
  implicitly by shipping the device + lib into the container.

The old `provider-build` stage (rdma-core v57 + the hellas-ai usb4_rdma
provider + v58 libibverbs) and the tbv natives are gone from the image; the
host-side `tbv/` kit is replaced by the vendored `odinlink/` driver kit
(including `uninstall-tbv.sh` for migrating a rig that ran the old stack).
RCCL's libibverbs dependency is simply unused now (`NCCL_IB_DISABLE=1`).

### 15.3 Transport profiles

`vsh-cluster-env.odl.sh` (default, `transport: odl`) — RCCL over the net
plugin, `DS4_ODL_AR2=1`, odl_mq data plane, control bootstrap still on
`thunderbolt0`. `vsh-cluster-env.tcp.sh` — sockets fallback, `DS4_ODL_AR2=0`.
The old `rdma`/`hybrid` profiles are deleted with the tbv stack they
configured.

**§10's "sockets beat the RoCE rail" measurement is hereby historic**: it
compared RCCL-over-ibverbs vs sockets on the tbv stack. The odl net plugin is
a different transport (no verbs marshalling, stream framing over DMA rings)
and has not been A/B'd end-to-end on this rig yet — the odl_ar2 decode path
and odl_mq control plane are the primary wins regardless of which way RCCL's
prefill collectives go.

### 15.4 Bring-up on the new pin: two fixes this rig needed, and the 2026-09-27 validation

**Fix 1 — `vsh-wna16-layerwise-empty-cache.patch` (the init OOM).** First
serve on the new pin OOM-killed the worker mid-load at every config (TP and
PP, dummy and real weights). Measured at death: torch `allocated=85.5 GiB`
but `reserved=120.3 GiB` and `gpu_active=119.7 GiB` — the TRITON WNA16 MoE
weight conversion (`convert_to_wna16_moe_kernel_format`: per-layer
`transpose().contiguous()` chains) leaves each layer's staging segments
reserved-but-free in the caching allocator, and on gfx1151 UMA reserved GTT
bills against system RAM. 42 layers × ~0.85 GiB of dead reservation plus the
transients exceeded the 124 GiB budget and the kernel OOM killer shot the
worker (which also strands the GTT: a killed-but-lingering vLLM process pins
its BOs via /dev/kfd until `fuser -k /dev/kfd` — the reason consecutive
failed attempts died earlier and earlier). A per-layer `torch.cuda.empty_cache()`
at the next layer's `process_weights_after_loading` bounds reserved to
allocated + one layer; the engine then inits at `alloc=reserved=82.4 GiB`
and serves. Load-time only.

**Fix 2 — `vsh-indexer-persistent-topk-rocm.patch` (decode determinism).**
The deterministic `persistent_topk` (#55122) is compiled and REGISTERED in
HIP builds (`torch.ops._C.persistent_topk` resolves in this image), but the
`SparseIndexerTopk` python dispatch gates every deterministic backend on
`is_cuda`; `auto` resolves ROCm to the nondeterministic `per_row` chain —
the same class of top-k divergence as #54521. Relaxing the persistent gate
(`topk_tokens` 512/1024/2048 fits GLM's `index_topk` 2048) makes decode
deterministic.

**Validation, 2026-09-27** (TP=2, odl transport, odl_ar2 ranks ready,
odl_mq writer/reader up, 128K ctx, 8 GiB KV = 634,762 tokens / 4.84x
concurrency — both better than the Sep-3 pin's 526,083 / 4.01x thanks to
#57701's −3 GiB indexer workspace):

| test | Sep-3 pin | Sep-27 pin + patches |
|---|---|---|
| greedy, 5 identical reps, ~1K-token prompt | 5 distinct | **1 distinct** |
| greedy, 5 reps, ≥8K-token prompt | 5 distinct | 5 distinct (multi-chunk prefill path still `top_k_per_row_prefill`, hardcoded, no backend dispatch — remaining upstream gap) |
| tool call, short context | 3/3 | **3/3** (also with MTP on) |
| tool call, 12K-token single-message context | 0/3, budget runaways | 0/3, but **coherent + grounded** responses (model reads/summarizes the padding correctly; corruption gone, routing still off) |
| MTP | corrupts structured output | **clean tool calls, ~1.9x decode** (10.3 vs 5.5 tok/s); `glm53_mtp_tokens: 3` is the shipped default again |
| prefill | ~284–334 tok/s @2–8K | 289 tok/s @8.3K (parity) |

Also fixed in passing: the launcher's `odl_up()` check false-negatived under
`set -o pipefail` (`lsmod | grep -q` → SIGPIPE 141); it now greps
`/proc/modules` directly. And `container/build.sh`'s post-build checks never
actually ran (`podman run python -` without `-i` reads EOF; the import also
needs `--device /dev/kfd --device /dev/dri` for the ROCm platform probe) —
both fixed.

### 15.5 The 2026-09-28 efficiency pass (MiaAI 2xDGX-Spark ideas, applied to this rig)

Four items from [MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks):

1. **Single-user serving profile — APPLIED.** `glm53_max_batched: 8192` (was 4096),
   `glm53_max_seqs: 32` (was 256; also frees Mamba cache blocks and would permit
   CUDA-graph capture). Measured: prefill is at parity (~270-290 tok/s @8K) —
   this GPU is compute-bound at prefill, the chunk size was not the limiter.
2. **CUDA graphs — TESTED, PARKED with evidence.** The old rank-1 Triton
   capture bug is GONE on this pin: both ranks capture 7/7 decode graphs +
   the MTP speculator (61 s, 13.6 GiB/rank) with `glm53_enforce_eager: 0`.
   But the first real request wedges the worker (EngineCore dies on the
   shm-broadcast dequeue timeout; no worker-side traceback) — full-graph
   replay is incompatible with the drafter's between-draft-steps attention
   metadata rebuild on this hybrid backend. And the trade loses anyway:
   graphs-without-MTP (~+10%) < MTP-without-graphs (+90%). Revisit only if
   piecewise capture composes with the speculator upstream.
3. **DFlash2 drafter — 80% PORTED, blocked at KV-group aliasing.** The
   upstream `method: dflash` + `incoai/GLM-5.3-Flash-DFlash2` (5-layer
   Qwen3-based drafter, declares target aux layers [5,14,24,33,42]) now
   reaches: EAGLE3-interface port (`vsh-glm-eagle3-aux-states.patch`:
   `EagleModelMixin` on `Glm5NextModel` with mHC-aware aux stashing via
   `hc_post`+`hc_contract`, `SupportsEagle3` on both wrapper classes),
   drafter loads, aux layers engage (`Using Eagle3 auxiliary layers from
   config: (6,15,25,34,43)`). BLOCKER: the drafter's SlidingWindow KV
   layers cannot join GLM's MLA/Mamba/kpool pool —
   `NotImplementedError: ... page size is not divisible ... cannot be
   padded` in `unify_kv_cache_spec_page_size`. MiaAI solves this with an
   872-line `patch_glm5_drafter_group.py` (partition the drafter specs into
   their own group + alias them onto MLA tensors at disjoint block ids in
   `_glm5_next_tensor_layout`); our pin's layout code has diverged, so that
   patch does not apply as-is. The aux-states patch is inert on the MTP
   path (aux layers are only set by dflash/dspark/eagle3 speculators), so
   it ships enabled. Knob: `glm53_spec_method: dflash|glm5_next_mtp`.
4. **Dense/KDA FP8 — SCOPED, NOT STARTED.** MiaAI's patch is 5 lines of
   "stop nulling quant_config for KDA/MLA constructors" — because their EXL3
   method then quantizes those projections (Marlin, CUDA-only). Our AWQ
   checkpoint carries those projections in BF16, so the equivalent needs a
   small model patch (keep quant config / attach an fp8 method at those
   linears) and ROCm fp8 GEMM plumbing for them. Expected payoff mirrors
   theirs (+10-19% decode) since our own §12 measured the bf16 attention+
   KDA reads at ~24% of every decode step. This is the next concrete
   efficiency project.

Also this pass: `deploy-new-stack.sh` no longer clobbers the live
`~/vsh-config.yaml`; `one-launch.sh` on box1 wraps the full
teardown+fuser-k/kfd+single-launch dance that failed launches need.

### 15.6 The DFlash2 drafter: fully ported, working, at parity (2026-09-28, second session)

The remaining KV blocker from §15.5 is solved. `vsh-dflash-drafter-kv-group.patch`
(6 anchored edits to `kv_cache_utils.py`, adapted from MiaAI's
`patch_glm5_drafter_group.py` to this pin's rewritten layout code):

1. `_get_kv_cache_groups_glm5_next` partitions the drafter's exact-type
   `SlidingWindowSpec` layers out of the MLA-uniformity check and appends them
   as one extra group (LAST, so existing group ids stay stable): exact-fit
   blocks when the drafter's page divides the MLA page, else a padded
   slot-share (`block_size=64, page_size_padded=mla_page`).
2. `_glm5_next_tensor_layout` recognizes the drafter group (9th tuple element).
3. The three consumers: bytes-per-block unchanged (the drafter adds no block
   bytes), the config builder aliases drafter layer i onto MLA tensor i's
   offset (the same disjoint-block-id mechanism the mamba groups use), and the
   memory-usage estimator accounts the drafter's block demand.

**Result: the DFlash2 drafter boots and serves.** Outputs are byte-identical
to MTP on probes ("The capital of France is Paris", same usage; clean
`get_weather({"location": "Paris"})` tool calls). KV pool cost of the drafter:
1,105,488 tokens at k=3 (vs 1,183,680 with MTP) and 1,000,204 at k=7.

**But it is parity, not the MiaAI 2.6×**: k=3 gives 10.6 tok/s vs MTP's 10.4
(acceptance ~50%, mean accepted length 2.50); k=7 *drops* acceptance to ~24%
(draft-depth collapse); replacing the `hc_contract` aux reconstruction with
DS4's `.mean(dim=1)` convention halves throughput (5.3 tok/s) — the drafter's
aux-state conditioning is convention-sensitive, and this checkpoint (trained
by incoai against some specific GLM aux integration) matches `hc_contract`
best but not perfectly. The remaining acceptance gap lives in exactly which
hidden-state transform the drafter was trained against; candidates to try
next: MiaAI's image's own GLM aux code (not public in their overlay), the
un-reconstructed mHC stream, or layer-index off-by-one variants.

Shipped state: `glm53_spec_method: glm5_next_mtp` (validated default) with
`dflash` one knob away (`glm53_spec_method: dflash`, `glm53_mtp_tokens` = k).
Both drafter paths exercise the same eagle3/KV machinery, so flipping between
them is a restart-only change.

### 15.7 Dense/KDA FP8 on gfx1151: a platform gap, precisely mapped (parked)

The MiaAI trick does not transfer today — every fp8 GEMM route on gfx1151 is
broken at a different layer, in order of discovery:

1. `is_fp8_fnuz()` (platforms/rocm.py) is `\"gfx94\" in arch` — MI300-only.
   gfx1151 therefore gets `fp8_dtype() = float8_e4m3fn` (the NVIDIA dtype),
   which **aiter's pybind dtype map rejects outright** (this build's map has
   no fp8 entries at all).
2. hipBLASLt (via aiter `hipb_mm`/`hipbsolgemm`, which JIT-compiles fine on
   gfx1151) has **zero fp8 GEMM algo solutions** on RDNA 3.5 — rowwise and
   tensorwise both (`hipblasLtMatmulAlgoGetHeuristic` returns 0).
3. vLLM's `ROCmFP8ScaledMMLinearKernel` gates on CDNA3+/RDNA4 honestly.
4. The **aiter Triton `gemm_a8w8_blockscale` RUNS** with fnuz tensors (no
   pybind wall) — the one viable route — but its scale semantics need
   reconciliation (a quick fnuz emulation gave rel≈658) and per-shape tuning
   for the KDA/MLA projection shapes (13 GB/s untuned at M=4 — the old rig's
   tuned A8W8-blockscale configs cover different (N,K)).

The online-quantization framework itself (`OnlineQuantizationConfig`,
`--quantization fp8_per_block`, meta-device load + quantize-during-load) is
fully wired at this pin and is the right attachment point once a GEMM exists:
pass it as the per-layer `quant_config` for the KDA/MLA projections instead of
the `quant_config = None` nulling in `kda.py:191` / `model.py`. Expected
payoff when landed: +10–19% decode (our §12: the bf16 KDA/MLA reads are ~24%
of each decode step). Also worth an upstream issue: `is_fp8_fnuz` excluding
RDNA 3.5 while aiter (the AMD library) only speaks fnuz.

### 15.8 Long-context diagnostics: what actually breaks, and a correction (2026-09-28)

**Correction to §15.5/§14.** The `vsh-indexer-persistent-topk-rocm.patch` never
engaged: (a) `persistent_topk` is an `#ifndef USE_ROCM` stub that raises
`persistent_topk is not supported on ROCm` (topk.cu:294) — no HIP build of the
kernel exists in this image; and (b) the AMD/ROCm GLM model
(`glm5next/amd/sparse_indexer.py`) bypasses the shared `SparseIndexerTopk`
dispatcher entirely, so the unlock could not have been selected anyway. The
observed determinism at short prompts is the **trivial select-all path**: when
every row's visible pool count ≤ `select_k` (512 pools = 2,048 tokens), no
selection happens at all. The patch has been dropped from the patch set.

**Diagnostic results** (chunk 8192, `index_topk` 2048, kpool 4, MTP on and off):

- Greedy divergence turns on at **> 2,048 prompt tokens** — i.e. the first row
  that must actually *select* pools — not at the chunk boundary (diverges at
  2,932 tokens = 1 chunk; MTP-independent).
- Tool calling works 3/3 up to ~7K tokens (1 chunk, real selection active) and
  fails 0/3 from ~9K (2 chunks) — a different threshold from the divergence.
- **Forced `tool_choice` at 12K produces well-formed calls with the wrong
  argument** — `{"location": "Harbor District"}` (an entity from the padding)
  instead of "Paris" (the final line of the prompt). This is a *retrieval*
  failure: the model cannot attend to the prompt's final tokens once prefill
  spans multiple chunks. The `index_kpool_always_select_tail` guarantee (the
  expand-pools-and-append-tail path) is the prime suspect.

**Two fix attempts, both reverted.** A deterministic stable-sort replacement
((value desc, index asc), the MoE-router convention) for the prefill pool
top-k — verified engaging via debug probe (`rows=6528 W=1632 k=512
det=True`, ray worker logs) — did **not** remove the divergence. Extending it
to the decode-side `top_k_per_row_decode` *regressed* 2.5K-token tool calls
0/3 (subtle tail-pool semantics in the decode path) and still did not fix
divergence; both were reverted to the stock kernels and the 2.5K behavior
re-verified at 3/3.

**Net narrowing of both defects:**
- *Greedy nondeterminism*: selection kernels are now exonerated as the primary
  source; with deterministic selections the outputs still vary — the noise is
  in the kernels' arithmetic itself (the Triton MQA-logits fp8 accumulation,
  the aiter sparse-MLA attention, or the mHC/KDA reductions). That is upstream
  kernel-level determinism work, not patchable from Python.
- *Long-context tool calling*: retrieval of the prompt's final tokens breaks
  at ≥2 prefill chunks. Next targets: `kpool_ops.expand_pools_and_append_tail`
  chunk-boundary behavior, the decode `dec_pos`/`dec_slot` tail indexing, and
  the ROCm sibling of upstream #58704 ("index_kpool mismatch → corruption via
  unread query token", fixed for SM90 only). Both are open upstream issues
  worth filing with this rig's reproducers.

## 16. Attribution

OdinLink userspace integration (`odl_ar2`, `odl_mq`, the RCCL net plugin
build, the vLLM hooks they hang on) and the `odinlink/` driver kit come from
[`AlexKGwyn/ds4-vllm`](https://github.com/AlexKGwyn/ds4-vllm) (Apache-2.0 for
the original work; the OdinLink-Five driver and kernel module are GPL via
Geramy/OdinLink-Five). The all-reduce/MessageQueue hooks are re-anchored
derivatives of that project's patch set. The previous tbv-era attribution
(rdma-core provider work via hellas-ai/thunderbolt-ibverbs) remains accurate
for the pre-2026-09-27 history of this repo. See THIRD_PARTY_NOTICES.md.

## 17. Adaptive draft length, wired for async scheduling (`vsh-adaptive-k.py`)

Ported from MiaAI-Lab's `overlay/patch_adaptive_k.py` (see the review in
`.analysis/`), then fixed: **the v1 port could not fire on this build.**

**Why it could not fire.** v1 trimmed `request.spec_token_ids` inside
`Scheduler.update_draft_token_ids`. This build resolves async scheduling **ON**
for spec decode, and `v1/engine/core.py:252` guards that call with
`not self.async_scheduling` — so the function is never reached, and the trim was
dead code. In async mode the *verify length* is fixed at **schedule time** by
`SchedulerOutput.num_spec_tokens_to_schedule`:
`AsyncScheduler._update_after_schedule` sizes `_spec_token_placeholders` from it,
and the worker hands the same number to the drafter
(`gpu_model_runner.py:4972` → `drafter.propose(num_speculative_tokens=…)`).
That cap inside `schedule()` is therefore the only lever that works here; the
sync-path trim is kept for the `async_scheduling=False` case.

**v2 adds** (a) the cap as the primary path, (b) per-step telemetry
(`VSH_ADAPTIVE_K_DEBUG=1`: the k that was *scheduled* vs the k actually
*verified*, the per-request choices and the accepted-token histogram), and
(c) a live override file (`VSH_ADAPTIVE_K_JSON`, `{"force": n}` / `{"margin": x}`)
so a **k-sweep needs one boot, not four**.

**Measured k-curve** (prose, temp 0, single request, 2.4K prompt, alternating
A/B/A/B, each window exactly one request; `force` from the override file):

| force | drafts | draft tokens | tokens/draft | accepted/step | steps/s | decode tok/s |
|---:|---:|---:|---:|---:|---:|---:|
| 3 | 129 | 387 | 3.00 | 1.98 | 4.41 | 8.92 |
| 2 | 128 | 256 | 2.00 | 1.88 | 4.41 | 8.54 |
| 3 | 113 | 339 | 3.00 | 1.85 | 4.45 | 8.26 |
| 2 | 121 | 242 | 2.00 | 1.87 | 4.45 | 8.35 |

**The override provably controls the verify batch** (tokens/draft is exactly
3.00 / 2.00), which is what v1 could not demonstrate.

**Verdict: adaptive K is throughput-neutral-to-negative on this rig, and the
reason is structural.** The step time is *independent of the verify length*
(225–227 ms at k=1, 2 and 3 — the step is CPU/launch-bound, see §13's profile:
~50% GPU-idle), so accepted tokens are nearly free and trimming only gives them
away. MiaAI's +13–21% comes from prose cells where *their* acceptance is 0.34;
ours is 0.85–0.99.

**k=4 was tried and does not help either** (boot with `glm53_mtp_tokens: 4`,
then forced down through the override in the same boot):

| force | tokens/draft | accepted/step | steps/s | decode tok/s |
|---:|---:|---:|---:|---:|
| 4 | 4.00 | 2.02 | 4.16 | 8.38 |
| 3 | 3.00 | 2.00 | 4.26 | 8.57 |
| 2 | 2.00 | 2.00 | 4.31 | 8.70 |
| 4 | 4.00 | 2.07 | 4.18 | 8.70 |

Per-position acceptance is ≈0.68 / 0.44 / 0.12 / 0.02, so **accepted tokens per
step saturate at ~2.0 for every k** — the MTP drafter's useful depth is two
drafts, and the step rate falls ~3 % from k=2 to k=4. The draft-length axis is
therefore exhausted on this rig in both directions.

**Verdict:** leave `VSH_ADAPTIVE_K=off` in production. The mechanism is
installed, env-gated and instrumented (`vsh-adaptive-k.py`), and the live
override makes a k-sweep a one-boot job — but there is no regime on this
workload where it beats a static k=3. Its value is the *receipts*: it is what
proved the scheduler-side verify length is the live lever and that our step cost
is draft-length-independent, which re-targets the work at the step time itself
(§13's launch/CPU overhead) rather than at the drafter.

## 18. Prefix-cache hits were being truncated by a mis-flagged EAGLE set (`vsh-glm53-apc-align.py`)

**Symptom.** Repeated prompts showed 0 % cache benefit, and the engine logged
`Prefix cache hit rate: 0.0%` on boots where identical prompts were replayed.

**Root cause (verified live).** `v1/core/kv_cache_utils.py:2364-2365` returns
**early** for GLM-5-Next (`elif glm5_groups := _get_kv_cache_groups_glm5_next(…):
return glm5_groups`), so `_annotate_eagle_groups()` (called later, at 2419) is
dead code and **no group ever gets `is_eagle_group = True`**. Our spec method
normalizes to `mtp` (`config/speculative.py:1135-1139`), so `use_eagle()` is
True and `kv_cache_coordinator.py:108-110` — "conservatively fall back to flag
all groups" — flagged **all five** groups. Every lookup then paid the EAGLE
last-block pop, i.e. one scheduler page (**2304 tokens**) off the hit.

**Fix.** Resolve the set to groups holding **only** drafter layers (layer-name
markers, or an exact `SlidingWindowSpec` group; `KpoolTailSpec` subclasses it and
is excluded by type identity). The MTP drafter layer merges into group 0 with the
target MLA layers, so the resolved set is **empty** and the drop stays off —
which is what upstream means by "the group that holds the drafter", not "every
group". The write-side protection is untouched (the last
`num_reprefillable_tokens = num_prefill_lookahead - 1` tokens are never cached,
`kv_cache_coordinator.py:347`), and the lookahead guard at 120-131 is re-asserted
explicitly because an empty set silently bypasses it. Env-gated with
`VSH_GLM53_APC_ALIGN=1` (default off = byte-identical behaviour).

**Boot receipt** (both lines absent before):

```
[vsh-glm53-apc-align] 5 groups, no annotation: eagle_group_ids=[] (upstream fallback would be all 5)
[vsh-glm53-apc-align] eagle=[] scheduler_block=2304 align=2304 partial_hash_hits=False retention=0
                      groups=[('MLAAttentionSpec', (0,), 'FullAttentionManager', False),
                              ('MambaSpec', (2, 3, 4), 'MambaManager', False)]
```

**Measured** (identical prompts, same `cache_salt`, `force=3` so the draft length
is held constant):

| probe | before fix | after fix |
|---|---|---|
| 13 877-token prompt, repeat | 11 520 cached (5 pages) | **13 824 cached (6 pages)** |
| 2 362-token prompt, repeat | **0 cached** | **2 304 cached** |
| TTFT, 13.9K prompt, cached | 8.92 s | **2.05 s** |
| TTFT, 13.9K prompt, cold | — | 53.57 s (259 tok/s prefill) |

One page is exactly the EAGLE pop; a prompt whose length is between one and two
scheduler pages used to get **nothing**. Spec acceptance is unchanged
(0.85–0.99 accepted/draft, per-position 0.67/0.31/0.12 before and after), which
is the gate this patch had to pass — the drop exists to keep draft KV written
with a lookahead out of reused blocks, and that is still covered by the
write-side trim.

**One more page hides behind the retention policy.** The fix above restores the
EAGLE page, but an identical repeat could still miss once before it hit
(observed: cold miss, repeat miss, third request hit). That is
`CacheConfig.prefix_cache_retention_interval`, whose default is **0** = "retain
only semantic checkpoints (the latest replay boundary and shared-prefix
junctions)" — the *sparsest* setting, and it applies to the sliding-window and
Mamba (KDA) groups, which sit inside the hybrid `min()`. See §20.

## 19. SHM reader busy-spin window (`vsh-shm-spin.py`)

`SpinCondition.wait()` busy-loops on `sched_yield()` for `busy_loop_s` seconds
after every read before falling back to the zmq poll; the reader is built with
the stock default of **1 s**, so the EngineCore↔worker control-plane reader
never idles while a request is in flight. Tracks MiaAI's `busy_loop_s 1 → 0.016`
(+0.95 % decode, −85 % EngineCore CPU on their kit). Env-gated:
`VSH_SHM_BUSY_LOOP_S` (unset = stock 1.0).

**Measured here: no gain — mildly negative.** Step rate is the sensitive
metric (it is remarkably stable across boots): **4.41 / 4.41 / 4.45 / 4.45
steps/s** at the stock 1.0 s window, versus **4.16 / 4.26 / 4.31 / 4.18
steps/s** at 0.016 s. Decode tok/s moved with it (8.92/8.54/8.26/8.35 →
8.38/8.57/8.70/8.70), and the difference is inside run-to-run spread, so treat
it as null. Unlike MiaAI's kit, our control plane already rides `odl_mq` (§15),
which is why there is less CPU left to win. Reverted to the stock window; the
knob stays available (`VSH_SHM_BUSY_LOOP_S=0.016`) for anyone chasing
host-side CPU again.

## 20. Prefix-cache retention interval (`VSH_GLM53_APC_RETENTION`)

`CacheConfig.prefix_cache_retention_interval` defaults to **0**, documented as
"retain only semantic checkpoints, including the latest replay boundary and
shared-prefix junctions" — the sparsest of the three settings (`0` / `N` /
`None` = dense) and, per its own docstring, it "sparsifies sliding-window and
Mamba (linear-attention) checkpoints" only. GLM-5.3's hybrid `min()` includes
three padded KDA/Mamba groups, so a hit is tied to a Mamba checkpoint that, at
retention 0, is only guaranteed at the previous request's replay boundary.

Measured on a 13.9K prompt, identical requests back to back:

| request | retention 0 (stock) | retention 2304 |
|---|---|---|
| 1st (cold) | 0 hits | 0 hits |
| **2nd (first repeat)** | **0 hits**, TTFT 53.1 s | **11 520 hits**, TTFT 9.62 s |
| 3rd | 13 824 hits, TTFT 0.94 s | 13 824 hits, TTFT 0.95 s |

So the first repeat went from "worse than cold" to five of the six pages. It is
**decode-neutral** (8.54 tok/s, 4.45 steps/s, 1.91 tokens/step — identical to
the stock-retention baseline) because it only changes which checkpoints are
retained, not what is computed.

The flag is stock `EngineArgs` (`engine/arg_utils.py:576`), so **no vLLM patch
is needed** — only the env-gated launcher plumbing
(`vsh-apc-retention.py`, `VSH_GLM53_APC_RETENTION=2304`; the value must be a
multiple of the scheduler block size, 2304 here). Unset = no flag = stock 0.

## 21. The `hipPointerGetAttribute` hunt: root-caused, measured, and stood down

**Where it comes from (proven, not inferred).** gdb on the live TP0 worker
(`sudo gdb -p <worker> -ex "set solib-search-path <container-rootfs>/..."`,
breakpoint on the symbol, 8/8 identical backtraces):

```
hipPointerGetAttribute
  <- extractPointer()      triton/backends/amd/driver.c:715   (compiled into
  <- launchKernel()        ~/.triton/cache/<hash>/hip_utils...so)
  <- <python> vLLM op -> ray worker
```

`extractPointer` resolves every **pointer argument of every Triton kernel
launch** with `hipPointerGetAttribute(..., HIP_POINTER_ATTRIBUTE_DEVICE_POINTER,
ptr)`. The driver fetches that symbol through `hipGetProcAddress`, so an
`LD_PRELOAD` interposer cannot see it — only the source can be patched.

**What it actually costs: ~0.5 µs.** Measured three independent ways:

| method | result |
|---|---|
| direct `ctypes` call on a real device pointer, 20 000 iterations | **0.50 µs/call** |
| the profiler's own API table (`hipPointerGetAttribute` 113.4 ms / 137 335 calls) | **0.83 µs/call** |
| the profiler *summary* table ("Self CUDA" 831.6 ms / 137 335 calls) | 6.06 µs/call ← **wrong** |

So the 8.7 ms/step figure quoted in §13's profile table (and repeated in the
2026-10-02 review) is **tracing overhead, not execution time**: ~1430 calls/step
× 0.5 µs ≈ **0.7 ms/step, 0.3 %**. Every `hip*` row in that summary table is
inflated the same way (each API call pays a profiler callback); the *kernel*
rows are device timestamps and remain trustworthy.

**A second correction from the same investigation.** Re-analysing the raw trace
(`scripts/trace_gap_analyze.py`, 96-step window) shows the GPU **79 % busy**
(9.07 s of merged kernel time in an 11.51 s span) with **no gap longer than
2 ms**. The "GPU ~50 % idle" reading in the review came from comparing the
profiler's *self* CUDA total (9.21 s) against an annotation CUDA total that
double-counts nested regions (18.40 s) — not against wall time. Live decode is
reproducible at **2.06 accepted tokens per 224 ms step = 9.2 tok/s**, confirmed
engine-side (`generation_tokens_total` delta over the same window: 9.2 tok/s),
so the client measurement was not understating anything.

**Outcome.** The memoisation patch (`container/patches/vsh-triton-ptr-cache.py`,
env `VSH_TRITON_PTR_CACHE=1`) is written, applied to box1's container, and
validated bit-exact (`max_abs_err=0.00e+00` with the flag on and off) — but the
honest verdict is **do not enable it**: 0.3 % is inside run-to-run noise and the
change touches every kernel launch. It stays in the tree as a documented,
reversible experiment.

**Where that leaves the step time.** With the API rows debunked, the remaining
non-kernel time in a 224 ms step is: the mamba/KDA `prepare_chunk_indices()
.tolist()` sync (a true device→host round trip per step, upstream #51540) and
the per-step host work that the trace shows as ~17 % API occupancy. That is what
to attack next — not the launch path, and not the pointer lookups.

## 22. Host syncs, measured to the end: the rig is GPU-bound, not CPU-bound

Section 21 killed the `hipPointerGetAttribute` figure. This section finishes the
job on the two remaining suspects -- the KDA chunk-index round trip and the
per-layer `nonzero` readbacks -- and closes the "the host is the bottleneck"
line of reasoning.

**The KDA chunk-index sync** (`third_party/flash_linear_attention/ops/index.py`,
`cdiv(lens, chunk_size).tolist()`, upstream #51540) is real, but it is on the
*chunked* path only: `models/glm5next/common/kda.py` takes it when
`attn_metadata_narrowed.num_prefills > 0`. Pure-decode steps run the recurrent
kernels (`fused_recurrent_kda` for spec rows, `causal_conv1d_update` +
recurrent for plain decode) and never call it. It fires **once per
prefill-bearing step**, not per decode step -- a profiled prefill shows the
chunked kernels (`chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter/intra`,
`chunk_gated_delta_rule_fwd_kernel_h_blockdim64`, 68 launches = 34 KDA layers
x 2) with the sync ahead of them. The generic GDN backend already passes a CPU
mirror (`v1/attention/backends/gdn_attn.py:201`), so the upstream fix is
"hand it the host-side query lengths"; the GLM path passes the device tensor.

**What the sync and its friends actually cost.** A profiled 6 972-token prefill
(TTFT 25.9 s, 270 tok/s):

| host API | calls | total | worst single |
|---|---:|---:|---:|
| `hipMemcpyWithStream` (the 4/8-byte count readbacks) | 1 490 | 24.1 s | 1.459 s |
| `aten::nonzero` (median 0.088 ms) | 228 | 23.4 s | 1.459 s |
| `aten::to` / `_to_copy` | 3 132 / 1 082 | 0.49 s | 0.41 s |
| `hipPointerGetAttribute` | 137 335 (decode window) | 0.11 s | -- |

**And here is the number that settles it: during that same window the GPU was
99.2 % busy** -- 25.87 s of merged kernel time in a 26.09 s span, 0.8 % idle, no
gap longer than 2 ms. The host spending 24 s inside synchronous copies is the
host *waiting for a saturated GPU*, not the GPU waiting for the host. The same
analysis on a decode window shows 79 % busy. A busy-GPU loop with a
synchronous D2H readback per layer is normal; nothing is being wasted.

So: **do not chase host syncs on this rig.** The kernel time is memory traffic
(MoE at ~87 % of achieved bandwidth, bf16 GEMMs at roofline), so the only
levers left are the ones that reduce *bytes per token*: the weight format
(§12/§13), the number of verified rows (k), and the drafter. That is also why
k=1/2/3/4 all land within 3 % of each other (§17): the weights are read once
per step regardless of how many rows are verified.

**Kept for reuse:** `container/patches/vsh-sync-instr.py`
(`VSH_SYNC_INSTR=1` times the chunk-index copy and logs every 20th miss; off by
default -- note the log gate means a short run prints nothing) and the trace
analysers `scripts/trace_gap_analyze.py` (GPU busy vs idle, gap attribution),
`scripts/trace_dtoh.py` (device->host copy inventory) and
`scripts/trace_sync.py` (blocking-op worst cases). Those three scripts are what
produced the corrections in sections 21 and 22, and they are the first thing to
run before believing any host-side number from the profiler summary.

## 23. Should we requant the BF16 tensors (the "Q8-attention" idea)? Measured: ~+6 %, and the tuned MoE config was stale

**The question.** antirez's PR-1024 measuring stick requantized a KDA/attention
checkpoint's BF16 tensors to Q8_0 and got decode 11.8 -> 19.0 t/s and prefill
69/72/52 -> 146/140/122, because "on the official GGUF those BF16 tensors were
more than half of the decode step". Our AWQ checkpoint has the same tensor class
in BF16 (`self_attn.*` incl. KDA, the MLA projections and `lm_head`; only the
routed experts are int4). So: what would it buy here?

**What the tensors cost, measured three ways.**

1. *Byte inventory* (per rank, per decode step, M=4): KDA projections
   34 x (12576x4096 + 4096x4096) x 2 B = **4.64 GB**, MLA projections
   11 x (2048x4096 + 4096x4096) x 2 B = **0.55 GB**, `lm_head` 1.27 GB ->
   **6.47 GB per rank per step**.
2. *Kernel bandwidth at those exact shapes* (`scripts/bf16_bw.py`, M=1..8):
   the big tensors run at **192-204 GB/s** (KDA in-proj 0.51 ms, `lm_head`
   6.2 ms) -- i.e. DRAM-bound, at the same rate the whole rig sustains. The
   small ones (`o_proj` 367 GB/s, MLA 531-571 GB/s) are L2-inflated in a
   microbenchmark and will not do that in-model, where every layer's weights are
   read once per step.
3. *Profile share*: bf16 dense GEMM = **27.7 ms of a 191.6 ms decode step
   (29 % of GPU time, 14.5 % of the step)** and only **7.9 % of a prefill
   window**.

**So the answer is the byte-ratio one, not the pathological one.** Our BF16 path
is *not* the C engine's: it is at roofline. Halving the bytes (fp8) gives
3.24 GB -> ~15 ms and saves ~14 ms of a 224 ms step = **+6.3 %**; quartering
them (int4) gives 1.62 GB but runs on the Triton wna16 path at ~95 GB/s
(measured, see the MoE sweep), which is also **~+6 %**. Prefill gains ~4 %
(7.9 % share, halved). That is the whole prize, and it is computable without
touching the checkpoint -- which is the useful part: the half-day of requant
surgery (plus the real precision risk on the KDA gates, which the checkpoint
authors left BF16 deliberately) buys about a sixth of what DFlash is worth.

**Do not use `lm_head` as the probe.** It is 1.27 GB = 6.2 ms/step = 2.8 % of the
step, inside the +-4 % run-to-run spread measured on this rig today. A probe has
to move at least the KDA class (4.64 GB, 9.6 %) to be readable.

**Side find: the tuned MoE tile config had been silently unused since the
Sep-27 pin bump.** The runtime builds the config filename as
`E=..,N=..,device_name=..[,dtype=..].json` and now asks for the name *without*
the dtype suffix, while the file deployed in September (and committed under
`host/moe-configs/`) carries `,dtype=int4_w4a16`. Every boot since Sep-27 logged
"Using default MoE config ... not found at ...8060S.json". Supplying the plain
name (both boxes; the original file is left in place) makes it load --
`fused_moe.py:1150 Using configuration from .../E=288,N=1024,device_name=AMD_Radeon_8060S.json`
-- and the measured effect is **nil**: decode 8.66/8.94 tok/s and prefill
268/261 tok/s, against 8.54/8.92 and 264-269/253-259 before. The +59 % decode
gain recorded in section 6 was against a stock configuration that the Sep-27
pin no longer ships, so the September sweep is stale rather than lost. Keep the
correctly named file (it is free and it is what any future sweep will overwrite)
but do not count it as a win.

## 24. DFlash2 k=7 vs MTP k=3, by workload: the drafter is fine, prose is just prose

Section 15.6 closed the DFlash2 port at "parity, not the MiaAI 2.6x" and blamed
the auxiliary-state convention, on the strength of a **k=7 acceptance of ~24 %**
that looked like draft-depth collapse (the signature of a causal mask inside the
draft block -- the failure the MiaAI kit warns about for `TRITON_ATTN`). This
re-measures it per draft position, on two workloads, with the same prompt and
the same client. Both boots are otherwise identical (APC fix on, retention 2304,
MTP/DFlash as noted, k from `vsh-config.yaml`).

| workload | drafter | drafts | draft tok | accepted/step | tok/step | decode tok/s |
|---|---|---:|---:|---:|---:|---:|
| structured (JSON) | MTP k=3 | 29 | 87 | 1.62-1.66 | **2.62-2.66** | 12.43 |
| structured (JSON) | **DFlash2 k=7** | 48 | 336 | **1.94** | **2.94** | **18.24** |
| prose | MTP k=3 | 123 | 369 | 0.85 | 1.85 | 8.41 |
| prose | DFlash2 k=7 | 111 | 777 | 0.88-1.24 | 1.88-2.24 | 7.63-9.18 |

**Per-position acceptance is the diagnostic, and it acquits the drafter:**

```
structured, DFlash k=7 : 30  21  14   9   8   7   4     (of 48 drafts)
prose,      DFlash k=7 : 64  23  10   1   0   0   0     (of 111 drafts)
```

Positions 4-6 accept on structured output (8/7/4) -- a drafter with a causal
mask inside its block could not do that. On prose the same positions accept
zero: the block-parallel drafter has nothing to latch onto six tokens deep in
unpredictable text, which is a property of the workload, not of the mask. So
15.6's "draft-depth collapse" was a prose-only artefact, and the auxiliary
states are exonerated.

**What it means for the rig.** DFlash2 k=7 is the better drafter exactly where
an agent workload spends its tokens -- structured/tool-call output: **2.94 vs
2.62 tokens/step (+12 %) and 18.2 vs 12.4 tok/s (+47 %) on the same prompt**.
On prose it is a wash (1.9-2.2 vs 1.9-2.1 tokens/step) and it costs ~10 % more
per step plus 15 % of the KV pool (1,000,204 vs 1,183,680 tokens).

Caveats before making it the default: the two structured runs generated
different lengths (141 vs 72 tokens), the decode-rate gap is larger than the
tokens/step gap (165 vs 210 ms/step, unexplained -- likely fewer distinct
experts routed on the predictable JSON), and n=1 boot per arm. The next step is
a matched-length, >=5-repetition A/B on real tool-call traffic, plus a k sweep
for DFlash on prose (k=4-5 may keep the structured win without paying for seven
drafts on prose). The adaptive-k machinery from section 17 is the natural place
to switch depth per phase once that A/B lands.

Switch: `glm53_spec_method: dflash` + `glm53_mtp_tokens: 7`. Production is left
on MTP k=3, the validated default.

## 25. DFlash2 vs MTP, matched A/B: the single-shot win was an artefact

Section 24 proposed DFlash2 k=7 as the default on the strength of a single JSON
probe (18.2 vs 12.4 tok/s, +47 %) and listed its caveats. This is the matched A/B
that was needed to settle it: same prompts, same `max_tokens=200`, 5 reps per
cell, medians, both arms booted back to back, and a step metric taken from the
engine's own draft counter (HTTP chunks undercount steps -- one step can deliver
several accepted tokens in one chunk, which is what inflated the earlier
number).

| workload | MTP k=3 | DFlash2 k=3 | DFlash2 k=7 |
|---|---|---|---|
| JSON output | 2.37 tok/step, 209 ms, **11.62 tok/s** | 2.20, 209 ms, 10.22 | 2.59, 239 ms, 9.83 |
| tool calls | 2.53 tok/step, 226 ms, 11.34 | 2.66, 210 ms, **12.65** | 2.92, 253 ms, 11.71 |
| prose | 2.01 tok/step, 220 ms, **9.14** | 1.92, 213 ms, 8.98 | 2.06, 243 ms, 8.37 |

**Verdict: do not switch the default.** MTP k=3 wins JSON by 12-14 % and prose by
2-9 %; DFlash k=3 wins the tool-call cell by 11.6 % but that cell's ranges
overlap (MTP 10.16-13.77, DFlash 11.47-13.86 over n=5) and its JSON cost is the
larger effect. The k=7 arm loses everywhere except the tool cell because the
8-row verify batch costs 13-23 % more per step (209 -> 239 ms, 226 -> 253,
220 -> 243) -- the step is *not* k-independent past k=4, which is where the
earlier "+47 %" went.

Per-position acceptance still stands as measured in section 24 (the drafter is
healthy, positions 4-6 accept on structured output); what fails is the economics:
the extra accepted tokens do not pay for the wider verify batch, and DFlash also
costs KV (1,000,204 tokens at k=7 and 1,105,488 at k=3 against 1,183,680 for
MTP).

If someone wants to chase the tool-call cell anyway, the honest next step is a
larger sample (>=20 reps) on real harness traffic, not a config flip: a single
specialised cell at n=5 is not a default. A DFlash k=3 default would also need
the JSON regression explained, and the only mechanism that captures both is the
per-phase adaptive depth from section 17.

**Production stays on MTP k=3.** Tooling added: `scripts/vsh_ab.py` (matched
A/B with engine-side step accounting, tool-call gate optional).

## 26. The degenerate tool-call mode (SUPERSEDED by section 27: it was the rig)

Section 25's fail-open fix stopped the silent loss; this is what the salvaged
text revealed about *why* it happens, and every configuration tested.

**Symptom.** Roughly 30-40 % of turns produce a usable tool call. The rest end
with `finish_reason=stop`, no tool call, and the model's attempt delivered as
content (after the fix) or dropped (before). The reasoning text is always
coherent; the *format* falls apart. Samples collected on the opencode prompt
(10 tools, 7055 prompt tokens), short user request:

    <tool_call>write</arg_key><arg_value>content</arg_key><arg_value>hello\n</arg_value>
    <tool_call>bash$$\n<parameter=name>bash</parameter>\n<command>echo hello ...
    <tool_call>writeXML\u200f=null_file\rplaceholderscript>
    write("/tmp/snake-test/hello.txt", "hello")
    bash -c 'echo hello > hello.txt' cwd=/tmp/snake-test
    ```bash\necho "hello" > /tmp/snake-test/hello.txt\n```
    <tool_call>write# Write hello.txt\n\nvoid main() {\n    print("hello");\n}
    <tool_call>Write借助内容/file</think>文件已创建：hello.txt 包含单词 "hello"

i.e. missing opening `<arg_key>` tags, invented dialects (Python call, C
function, shell one-liner, markdown fence, `<parameter=name>` XML), mixed
scripts, and false success claims. Occasionally a clean call.

**What it is not** (each measured, same request, ~5 repetitions per arm):

| hypothesis | test | result |
|---|---|---|
| output budget truncates the call | captured opencode request | `max_tokens=32000`, `stop=None` -- not it |
| the reasoning span eats it | `chat_template_kwargs {thinking: false}` | same rate (2/6) |
| the tool count overloads it | 2 tools vs 10 tools | 0/4 vs 2/4 -- not monotonic, not the lever |
| spec-decode depth corrupts it | live `{"force": 1}` (verified 1.00 draft tok/step) | **worse**: 1/6 usable |
| thinking effort | `reasoning_effort` low / high | 1/5 / 2/5 |
| the prompt never states the format | rendered `chat_template.jinja` | it states it exactly: `<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key><arg_value>{arg-value-1}</arg_value>...` |
| a patched/substituted template | model dir mtimes | `chat_template.jinja` is the checkpoint's own (Sep 27 15:08); only `tokenizer_config.json` was touched (Sep 27 15:59, empty inline template -> external file) |

**What it is.** A checkpoint-level deficiency in emitting this tool format. The
model has the format in its prompt, at temperature 0, and still substitutes
Python, C, shell, markdown and invented XML dialects for it a majority of the
time, with coherent reasoning around it. Nothing in the serving stack selects
those tokens.

**Consequences and what to do.**
  * Keep the fail-open patch (section 25): the attempt is delivered as content,
    and a client (or a human) can act on it.
  * The 12 k-token loops are the same failure: the model writes the whole file
    into an `<arg_value>` it never closes, and with `max_tokens=32000` that is
    ~20 minutes of GPU. Bound it client-side (opencode `limit.output` /
    `maxTokens` ~8192 is ample for normal edits) -- it turns a 20-minute black
    hole into a 5-minute one.
  * The obvious cross-check -- the same model at a different quantization -- is
    not available on this rig (the fp8/bf16 checkpoints do not fit, section 3),
    so "AWQ W4A16 damage" versus "this checkpoint is weak at tool calls" cannot
    be separated here. Worth testing on any box that can hold another build.
  * MTP-off cannot be used as a control: it crashes the worker with an illegal
    memory access. Note also that MTP-off changes `scheduler_block_size`
    (2304 -> 2176), so `VSH_GLM53_APC_RETENTION` must be re-derived or the
    coordinator refuses to start (`prefix_cache_retention_interval ... must be
    a multiple of scheduler_block_size`).

Tooling added: `scripts/loop_probe.py` (per-run capture plus a repetition
score), `scripts/tool_count.py`, `scripts/effort_test.py`, `scripts/replay.py`
(replay a captured request verbatim), `scripts/render_prompt.py` (render the
tool prompt through the checkpoint's template).

## 27. Long-context sparse attention was broken past ~4K tokens (four stacked bugs) -- fixed; Issue B was this, not the checkpoint

**Correction to section 26.** The degenerate tool-call mode was not a
checkpoint deficiency. Above `index_topk` (2048 tokens) GLM-5.3's DSA
attention only sees the 512 kpool pools the indexer selects, and on this rig
that selection -- and the index cache it reads -- were broken in four
independent ways. The model could not look back at the tool schema at the top
of a 7K-token opencode prompt, so it improvised formats while the local
reasoning (recent tokens, always attended) stayed coherent.

### Receipts

| test | before | after |
|---|---|---|
| 3 watchwords at 25/50/75 % of a 13K prompt (`scripts/midneedle.py`, 3 seeds) | **0/3** each, 3000-token budget exhausted, not even found in the reasoning | **3/3** each, exact, 124-158 tokens |
| same at 6.8K (single prefill chunk) | right needle, wrong digits | 3/3 exact (3 seeds) |
| same at 32K | -- | 3/3 exact |
| 1.7K control (no selection) | 3/3 | 3/3 |
| opencode captured request, `scripts/replay_n.py 6` | ~2/6 usable tool calls | **6/6**, ~1000 tokens each, short reasoning |

### The four bugs, in the order the data exposed them

1. **fp8 overflow to NaN** (`vsh-idx-fp8max-fnuz.py`). gfx1151's
   `fp8_dtype()` is e4m3fn (max 448); the kpool writer quantized with
   `FP8_MAX = 448`, and our `pa_mqa_logits.py` overlay converted the cache to
   e4m3fnuz (max 240) -- a non-saturating cast, so every value above 240 became
   NaN (CPU-verified: 256..448 -> nan). Fix: quantize to 224 (vLLM's own fnuz
   max) on ROCm+e4m3fn. Made moot for decode by fix 2, kept for safety.

2. **Wrong decode reader** (`vsh-kpool-paged-logits.py` +
   `vsh_kpool_paged_logits.py`). gfx1151 fell through to aiter's
   `deepgemm_fp8_paged_mqa_logits_stage1`, a block_size==1 reader: it indexed
   the page table per *pool position* and read the first 128 bytes of a page as
   a pool row, while the GLM cache is paged (64 pools/page, 16x16 SHUFFLE values
   + trailing fp32 scales). GPU unit test with the production writer
   (`scripts/test_kpool_logits.py`): top-512 overlap with ground truth **0.32-0.35
   (= random for 1500 pools)**; with the bounded page table it even
   memory-faulted. New Triton reader decodes e4m3fn bits in-kernel, reads the
   real layout: overlap 1.000, max rel err 9e-8; live, logits match a torch
   reference on the real cache to ~1e-5. Also removes the 12.6 ms/step
   whole-pool `float8_copy`. Gate `VSH_KPOOL_PAGED_LOGITS=0`.
   (The prefill path was verified correct by `scripts/test_prefill_logits.py`.)

3. **Page aliasing in the index cache** (`59412-pooled-indexer-kernel-blocks.py`
   = port of upstream PR #59412, fixes #58858). On ROCm the indexer backend
   advertised `[1, MultipleOf(16)]`, so the kernel block stayed at the hybrid
   manager block while writer and readers address 64-pool pages. Live, 13K
   prompt: **17 valid table columns for 51 needed pages**; the slots written for
   pools 3261..3264 were 61, 62, 63, 0 -- every pool past ~1088 (token ~4.3K)
   was written to and read from page 0, overwriting each other. This is what
   the "every pool scored exactly 0" / pool-7,295 dead zone was. With the PR the
   block table is page-granular; `vsh-idx-bt-gather-v4-ops.py`'s gather-site
   expansion becomes inert (drop it when rebasing onto #59412).

4. **Force-tail evicted selected pools at random**
   (`vsh-idx-force-tail-boost.py`). `_force_recent_tail_pools` wrote the 128
   most recent pools over the last 128 top-k columns, assuming they were the
   lowest-ranked; the top-k output is not sorted, so ~1/4 of the selected pools
   were discarded. Live: all needle pools selected, only 6/10 needle tokens
   attended -> digits copied wrong. Fix: boost the recent pools' logits before
   top-k (same guarantee, nothing evicted). Live after: 10/10, 9/9, 9/9 needle
   tokens attended, zero duplicate pools.

### Consequences for earlier sections

* Section 26 is superseded: keep the fail-open patch (section 25), but the
  "AWQ damage vs checkpoint weakness" question no longer needs a second
  quantization -- the model calls tools fine once it can see the prompt.
* Section 15.8's long-context tool failures, the >2K greedy nondeterminism and
  the needle results that only passed for *tail* needles (the force-tail made
  the tail visible while selection was random) all belong to this family.
  Re-measure them.
* Section 26's MTP-off "illegal memory access": PR #59412 notes the same
  aliasing writes past the row buffer at other block geometries, and MTP-off
  changes the scheduler block (2304 -> 2176). Likely the same bug; untested.
* Performance: decode no longer runs the whole-pool fnuz conversion
  (~12.6 ms/step). Not yet re-benchmarked.

### Leftovers worth cleaning (not changed here)

* The prefill path carries **ungated** debug logging from the 2026-09/10 hunt
  (`/tmp/glm_scores.log`, `glm_topk.log`, `glm_gather.log`, `glm_kpool.log`,
  `glm_map.log`, `glm_idxplan.log`), each with host syncs; they grow without
  bound (MBs per session).
* Debug-only, inert unless a flag file exists in the container:
  `container/patches/debug/vsh-kpl-diag.py` (`touch /tmp/vsh_kpl_diag` -> live
  selection check) and `debug/vsh-kpl-hook.py` (execs `/tmp/vsh_kpl_hook.py`
  with the decode locals; `scripts/vsh_kpl_hook*.py` are the probes used above).

Tooling added: `scripts/midneedle.py` (mid-context retrieval, the test that
discriminates this bug -- tail needles cannot), `scripts/needlepos.py`,
`scripts/test_kpool_logits.py`, `scripts/test_prefill_logits.py`.

### 27.1 Verification of section 27, and the decode numbers it was missing

Section 26 concluded that the tool-call degeneration was a checkpoint-level
deficiency. **That conclusion was wrong**, and this is the measurement that
settles it: the same harness, the same captured requests, run against the
containers carrying section 27's four fixes.

| harness | before the four fixes | after |
|---|---|---|
| opencode's exact request (7 055-token prompt, 10 tools), 8 replays | ~30-40 % usable (4/6, 5/12, 2/6 across three samples) | **8/8 usable** |
| the same request with a trivial user prompt, 6 replays | 0/4 usable | **6/6 usable** |

Why the sparse-attention bugs produced exactly the symptom in section 26: in
opencode's prompt the *tool-format instruction* -- the line that tells the
model `<tool_call>{name}<arg_key>{k}</arg_key><arg_value>{v}</arg_value>` --
sits at character 21 113 of a 30 931-character prompt, i.e. about 5 k tokens in,
right inside the "dead zone past ~4 K tokens". The model was being asked to
emit a format whose specification it could no longer read, so it guessed:
Python calls, shell one-liners, markdown fences, `<parameter=name>` XML, mixed
scripts. That also explains why every synthetic probe in section 26 (600-1 100
token prompts, format spec well inside the healthy region) produced clean tool
calls on the first try, and why disabling thinking, cutting the tool list or
forcing k=1 changed nothing -- none of them touch the broken retrieval.

**Decode, re-measured** (the section-27 open item; medians of 5, 200 tokens,
same harness as the pre-fix baseline):

| workload | before | after | delta |
|---|---|---|---|
| JSON output | 2.37 tok/step, 209 ms, 11.62 tok/s | 2.67 tok/step, 217 ms, **11.87 tok/s** | tok/step +12.7 %, rate +2.2 % |
| tool calls | 2.53 tok/step, 226 ms, 11.34 tok/s | 2.95 tok/step, 226 ms, **13.07 tok/s** | tok/step +16.6 %, rate **+15.3 %** |
| prose | 2.01 tok/step, 220 ms, 9.14 tok/s | 1.95 tok/step, 223 ms, 8.74 tok/s | within run-to-run spread |

Prefill: a 4 683-token prompt now measures **279 tok/s** against 268 tok/s at the
same size before (+4 %). The 16 028-token point (182 tok/s, 88 s TTFT) is not
comparable to the earlier 13 879-token baseline (261 tok/s) because it is a
different prompt crossing a chunk boundary -- re-measure at matched sizes before
reading anything into it.

The wins land where the fix applies: acceptance per step rises everywhere (the
model is no longer writing into a corrupted context), and the tool-call cell
gains most. `vsh-toolcall-failopen` is kept as defence in depth: it is inert
unless a turn ends with an unterminated tool call, and it now has nothing to do
in normal operation.

## 28. Decode speed, round 1 (2026-10-03): power budget, fp8 BMM, router, split attention

Starting point after §27: ~224 ms/step at short context, ~229 ms at ~3K
(`scripts/stepbench.py`, thinking on, back-to-back requests). The plan and the
corrected profile analysis are in [FRESH-EYES-20tps.md](FRESH-EYES-20tps.md).

| change | short ctx step | ~3K ctx step | tok/s (3K) |
|---|---:|---:|---:|
| baseline (§27 build) | 224 ms | 229 ms | 12.0 |
| + iGPU SCLK cap 2100 MHz (28.1) | ~202 ms | -- | ~13.9 |
| + `VLLM_ROCM_USE_AITER_FP8BMM=0` + packed-key router (28.2, 28.3) | **183 ms** | **192 ms** | 14.5 |
| + split-KV decode attention (28.4) | 185 ms | 206 ms* | 16.7* |

\* different generated text (3.45 vs 2.78 tok/step), so the 3K step is not
directly comparable; the profile (below) is the receipt for 28.4.
Prefill: 264-270 -> **283 tok/s** at 9.6K (28.1). Mid-context retrieval
(§27 needle test) re-checked after each change: 3/3 at 6.8K, 13K and 32K.

### 28.1 The CPU was clamped to 2.0 GHz during decode (`glm53_gpu_sclk_max`)

The iGPU and CPU share one package power budget. Under decode the GPU runs at
~2.4-2.5 GHz and the firmware drops **every CPU core to its 2.0 GHz floor**
within ~5 s (temperatures 53-58 C: not thermal; governor/EPP/tuned are already
`performance`). Steady-state decode is partly CPU-bound (Python launch path),
so the first ~25 steps of a request after idle run at 188 ms and everything
after at ~226 ms -- which is also why single runs disagreed (186 vs 224 ms) and
why "thinking vs non-thinking" looked like a factor (it was run order).

Capping the iGPU SCLK frees budget for the rest of the chip. Sweep (both boxes,
same cap, 3K context, steady state):

| cap MHz | 1500 | 1800 | 2000 | **2100** | 2200 | 2300 | 2400 | 2600 | 2900 (stock) |
|---|---|---|---|---|---|---|---|---|---|
| step ms | 244 | 219 | 206 | **202** | 205-212 | 226 | 227 | 226 | 229-234 |

Prefill also improves (264-270 -> 283 tok/s at 9.6K): at 2.9 GHz the GPU is
already power-throttled. New launcher knob `glm53_gpu_sclk_max: 2100`
(`host/vsh-gpu-sclk.sh`, applied to both boxes in `vsh-cluster-restart.sh`;
non-persistent sysfs `pp_od_clk_voltage`; `auto` restores stock). Re-sweep if
the step becomes fully GPU-bound (graphs) -- the optimum will move up.

### 28.2 `VLLM_ROCM_USE_AITER_FP8BMM=0`

MLA's W_UK/W_UV absorption ran aiter's Triton fp8 BMM, emulated on RDNA3.5:
~0.38 ms x 28 calls = 10.7 ms/step. Off -> plain bf16 `torch.bmm` (0.9 ms/step
in the profile), more precise, and boot skips the 2x1024-shape precompile.
Set in `vsh-cluster-env.odl.sh` (`VSH_GLM53_FP8BMM` overrides).

### 28.3 Packed-key router top-k (`vsh-router-packed-topk.py`)

The deterministic router's stable sort compiled, *in-model*, into one persistent
Triton kernel at 2 workgroups x 32 threads: 242 us x 45 calls = ~10.9 ms/step.
Replaced by `topk` over unique int64 keys (monotonic fp32 bits << 16 |
65535 - index): bit-identical selection (600 random/tie/-inf cases; full
`grouped_topk` ids and weights identical at M = 1/4/8/740), ~1 ms/step.

### 28.4 Split-KV sparse decode attention (`vsh-sparse-attn-split.py`)

Decode ran the sparse MLA attention through the prefill ragged kernel (one
program per query x 16-head block: 8 workgroups at M=4). New flash-decoding
split + combine for <=64 (query x head-block) programs:
unit test max abs diff ~1e-3 (bf16 output rounding), 2.3x at M=4 / 2048 keys,
3.9x at M=1; in-model at 3K: ~7.7 -> 4.0 ms/step. Gate `VSH_SPARSE_ATTN_SPLIT=0`.

### 28.5 HIP int4 MoE decode kernel (`vsh-moe-int4-hip.py`) -- the big one

`container/patches/vsh_moe_int4.hip` (+ `vsh_moe_int4.py` ctypes wrapper,
built with `hipcc --offload-arch=gfx1151 -O3 -shared -fPIC`): a wvSplitK-style
GEMV for symmetric int4 / group-128 MoE weights in the layout the stock kernel
uses (`B[E][N][K/2]`, `S[E][N][K/128]`, moe_align blocks). One workgroup =
(expert block, 32 output rows), 8 wave32s x 4 rows; each lane streams 16
contiguous packed bytes per step, the block's tokens are staged once in LDS,
fp32 accumulate, shuffle reduce. Routed from `invoke_fused_moe_wna16_triton_kernel`
for <= 64 (token, expert) pairs; anything else stays on Triton.
Gate `VSH_MOE_INT4_HIP=0`.

Unit test (`container/patches/debug/test_moe_gemv.py`, real shapes, 48 experts
to fit memory): rel err 2.4e-3 vs stock (= the stock kernel's bf16 rounding of
the dequantised weights; this kernel accumulates in fp32), and

| case | stock Triton | HIP | |
|---|---:|---:|---:|
| M=1 w13 (N 2048, K 4096) | 0.493 ms, 70 GB/s | 0.166 ms, **208 GB/s** | 2.97x |
| M=4 w13 (MTP verify) | 1.517 ms, 71 GB/s | 0.666 ms, 162 GB/s | 2.28x |
| M=4 w2 (N 4096, K 1024) | 0.705 ms | 0.283 ms, 191 GB/s | 2.49x |
| M=8 w13 | 2.069 ms | 0.978 ms | 2.12x |

In model (profile, ~3K context): routed MoE **89.8 -> 37.3 ms/step**.
End to end (`stepbench.py`): **185 -> 125 ms/step at short context, 192-206 ->
130 ms at ~3K: 22.0-22.3 tok/s** (was 12.0 at the start of section 28).
Correctness: §27 needles 3/3 (6.8K, 13K), opencode replay 4/4 valid tool calls.
SCLK cap re-swept after this change: 1800-2100 MHz all 127-133 ms, 2400 154 ms,
2900 183 ms -- the cap matters *more* now (less GPU work, larger CPU share);
2100 kept (best for prefill).

Remaining (profile): BF16 GEMV ~49 ms/step (40 % of GPU time) -> 8-bit weights
+ a W8A16 kernel of the same shape as this one; MTP layer-45 BF16 experts
8.4 ms (requant to int4 and they ride this kernel); odl2 waits 6.7 ms; mHC
4.4 ms; fp32 router gate 4.3 ms; sparse attention 4.0 ms. CPU is now close
behind the GPU (launch->start median 5-24 ms), so graphs come back into play.

### 28.6 Load-time int8 for the BF16 linears (`vsh-w8a16-linear.py`)

The ds4 Q8_0 idea, done online instead of as a checkpoint rewrite. After loading,
`UnquantizedLinearMethod.process_weights_after_loading` quantizes eligible BF16
linears to int8 with one fp32 scale per 128 inputs and frees the BF16 copy
(`vllm/model_executor/layers/vsh_w8a16.py`). Eligible: 2-D bf16, K in
{1024, 1536, 2048, 3072, 4096, 6144, 8192}, N >= 32, the layer's `forward` is a
stock Column/Row/Replicated one (the MoE router gate overrides forward and reads
`.weight` directly -- first boot failed on it), and not `kv_b_proj` (absorbed by
MLA), `wk_weights_proj` (sliced by the indexer), embeddings, `lm_head`, vision.
Decode (<= 8 rows) runs `vsh_w8a16.hip` (one wave per output row, 64
contiguous int8 per lane per step -- best of an 8-way tile sweep on DRAM-bound
shapes); larger batches dequantize to a transient BF16 weight + stock GEMM.
Gate `VSH_W8A16` (set to 1 in `vsh-cluster-env.odl.sh`, `VSH_GLM53_W8A16`
overrides).

| check | BF16 | int8 |
|---|---|---|
| mean NLL, PATCHES.md excerpt (2100 tok) | 2.2141 | 2.1938 |
| mean NLL, AGENTS.md excerpt (2140 tok) | 1.8093 | 1.8068 |
| mean NLL, vsh_ab.py source (1474 tok) | 0.9596 | 0.9529 |
| §27 needles 6.8K / 13K / 32K | 3/3 | 3/3 |
| opencode tool-call replay | 4/4 | 3/3 |
| prefill, 9.5K prompt | 283 tok/s | 285 tok/s |
| MemAvailable at boot | ~8.0 GB | **10.4 GB** |
| decode step short / 3K | 125 / 130 ms | **118 / 126 ms (23.8 / 27.9 tok/s)** |

Kernel microbench (unit test `scripts/test_w8a16.py`): the DRAM-bound KDA
in_proj (12576 x 4096) runs 171-177 GB/s at M=1/4 (1.5-1.6x the BF16 path on
half the bytes); kernel error vs dequant reference 2.3e-3, quantization error
vs BF16 6.7e-3 per layer. In model: BF16 GEMV ~40 -> int8 GEMV 25.6 ms/step.

**The step is balanced again.** GPU busy ~118 ms/step; after the queue drains,
kernels start ~0.5 ms after launch (CPU floor). Next levers: CUDA graphs (CPU
side), `lm_head` x4 reads (8.4 ms, still BF16), MTP layer-45 BF16 experts
(8.0 ms; int4 would put them on the 28.5 kernel), fp32 router gate (6.7 ms),
odl2 waits (7.6 ms).

### 28.7 DFlash2 vs MTP, re-run on the fixed and faster stack: DFlash2 k=3 wins

Section 25 kept MTP k=3 (DFlash2 k=3 ~parity, k=7 13-23 % costlier per step).
Both inputs to that verdict changed: the sparse-attention bugs (§27) corrupted
the context past ~4K tokens for any drafter, and §28.5-28.6 changed how step cost
scales with verify rows and removed most of the BF16 cost. Same harness
(`scripts/vsh_ab.py`, 5 reps, 220 tokens, medians), same build, one boot per arm:

| arm | step ms | JSON tok/s | tool calls tok/s | prose tok/s |
|---|---:|---:|---:|---:|
| MTP k=3 | 126-128 | 18.88 | 20.13 | 15.96 |
| **DFlash2 k=3** | **114-119** | **19.92** | **21.46** | **16.12** |
| DFlash2 k=5 | 136-142 | 19.10 | 21.05 | 13.72 |
| DFlash2 k=7 | 160-165 | 16.09 | 19.49 | 11.96 |

DFlash2 k=3 correctness: §27 needles 3/3 (6.8K, 13K), opencode replay 3/3
(k=7: 3/3 and 3/3 too). Its step is cheaper than MTP's because drafting is
one forward of a small dense model instead of three MTP-layer forwards with
BF16 experts and three lm_head reads; the extra verify rows of k=5/7 cost more
than their extra accepted tokens. Memory: without the MTP layer, MemAvailable at
boot is **16.6 GB** (MTP: 10.4 GB). `stepbench.py`: 110.5 ms/step short context
(23.8 tok/s), 117.5 ms at 3K.

Default changed: `glm53_spec_method: dflash`, `glm53_mtp_tokens: 3`
(`glm5_next_mtp` reverts). The section 24/25 "corruption" was the section 27 family.

### Where the step is now (profile, ~3K context, 28.1-28.4, before 28.5)

196.6 ms/step profiled, **GPU 90 % busy** (was 72 %): int4 routed MoE 89.8 ms
(51 %), BF16 GEMV ~49 ms, MTP BF16 experts 8.4 ms, odl2 all-reduce waits 7.5 ms,
mHC 4.3 ms, fp32 router gate 4.1 ms, sparse attention 4.0 ms, indexer logits
0.45 ms (§27's reader; was ~16.7 ms with the fp8 copy + stage1 + head-sum).

### Tried and parked: a Triton int4 MoE GEMV

`container/patches/debug/vsh_moe_int4_gemv.py` (not installed): bit-correct
(rel err 2e-3 = the stock kernel's bf16 rounding), but 0.6-0.9x the stock
kernel in two formulations (broadcast-FMA spills registers on RDNA; a
byte-once WMMA variant ~0.87x). Note the stock MoE kernel slowed 75.6 -> 85.3
ms/step when SCLK dropped 2900 -> 2100: it is latency/ALU-bound, not
bandwidth-bound, so the ~2x headroom is real -- but it needs a hand-written
HIP kernel (wvSplitK-style), not more Triton.

### Next

1. CUDA graphs (README known issues #2): the step is now ~90 % GPU-busy, so
   graphs mainly remove the remaining CPU-side gaps and protect against the
   power-clamp; required before the CPU frequency stops mattering.
2. HIP int4 MoE decode kernel (~90 ms -> ~45 ms target).
3. 8-bit weights for the BF16 classes (~49 ms of GEMV -> ~25 ms; needs a W8A16
   skinny kernel), lm_head x4 reads (draft vocab or Q8), MTP experts int4.

## 29. CUDA graphs, 2026-10-03 attempt: first graphed decode step hangs in an RCCL kernel (resolved by section 33)

Launcher knobs added (inert while `glm53_enforce_eager: 1`): capture sizes
`VSH_GLM53_CG_SIZES` (default `1 2 4 8`) and `VSH_GLM53_CG_MODE` (default
`FULL_DECODE_ONLY`, so prefill-bearing steps -- the KDA chunk-index host sync of
§11/§22 -- stay eager). Capture itself is fine: target FULL graphs 2/2 sizes and
the DFlash2 drafter graphs capture in ~2 s, 0.4 GiB/rank (FULL_AND_PIECEWISE:
1.7 GiB).

**Every configuration tried hangs on the first graphed decode step** (warmup
request): {DFlash2 k=3, spec off} x {FULL_DECODE_ONLY, FULL_AND_PIECEWISE} x
{odl_ar2 on, off} x {the §27/§28 kernels on, all four off}. Evidence:

* torch PG watchdog on both ranks: stuck at SeqNum 32 `_ALLGATHER_BASE`
  (309 760 elements = 4 rows x 77 440 vocab: the logits all-gather right after
  the target forward), last completed 31 -- symmetric, no PG divergence.
* rocgdb on both workers (`info dispatches`): the GPU (100 % busy) is spinning in
  `ncclDevKernel_Generic_4` on queue QID 3 with ~3 667 packets still queued
  behind it on *both* ranks (read 4423/write 8090 vs 4415/8082) -- the same point
  of identical graph replays, i.e. the ranks did not diverge; a collective inside
  the replayed graph never completes.
* Not the transport: standalone 2-rank RCCL over the same OdinLink net plugin
  and env (`scripts/nccl_graph.py`, `nccl_graph2.py`) -- all-reduce captured in a
  graph and replayed, and 30 x (graph of 50 all-reduces on comm A + eager
  all-gather on comm B) -- all complete and correct.
* Not the new kernels (hangs with VSH_MOE_INT4_HIP / W8A16 / SPARSE_ATTN_SPLIT /
  KPOOL_PAGED_LOGITS all 0). The 2026-10-02 MTP-off graph success predates §27
  (PR #59412 kernel-block change, force-tail boost, fp8max) and §28.2-28.3 --
  those are the remaining bisection candidates, plus vLLM's pynccl path under
  graph capture.

Next steps, cheapest first: (1) bisect the remaining 10-02 -> now deltas with
graphs on (59412 port first: it changes the MLA/indexer group's kernel block
size); (2) `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=COLL` at capture to list each
rank's captured collectives (op, count, comm); (3) the ds4-vllm approach -- an
eager break around every TP all-reduce (`communication_op.py` wrapper), which
keeps collectives out of graphs entirely and lets odl_ar2 serve them. Expected
value once working: the step is CPU/GPU balanced at ~110-118 ms, so graphs alone
are worth ~5-10 %, and they make later GPU savings land 1:1.

Production stays eager (DFlash2 k=3, odl_ar2): 109.9 ms/step short context
(24.3 tok/s), 116.9 ms at 3K (23.3 tok/s).

## 30. GPU-side quick wins on the DFlash2 stack (2026-10-03) -- and the CPU is now the floor

Profile of production (DFlash2 k=3, all of §27-§28) before this section: GPU busy
107.6 ms/step. Three items:

1. **DFlash2 drafter attention -> `TRITON_ATTN`** (`VSH_GLM53_DRAFT_ATTN`, default
   `TRITON_ATTN` in `vsh-manual-serve.sh`; empty = vLLM default). The drafter's
   non-causal 4-token block was running a prefill-style `_fwd_kernel` with 16
   workgroups: 5 x 1.49 ms = 7.4 ms/step. Acceptance unchanged (tokens/step
   2.33/2.63/1.82 vs 2.27/2.55/1.89, JSON/tools/prose), steps 3-4 ms cheaper:
   JSON 19.9 -> 21.0, tools 21.5 -> 22.2 tok/s, prose within noise; needles 3/3.
2. **int8 `lm_head`** (`vsh-gate-lmhead.py`, `VSH_W8A16_LMHEAD`): an int8 group-128
   copy next to the BF16 weight (kept because the DFlash2 drafter shares the
   module); target (4 rows) and drafter (3 rows) logits run the W8A16 GEMV.
3. **fp32 router-gate GEMV** (`vsh_w32_gemv`, `VSH_ROUTER_GEMV`): written and unit
   tested (rel err ~2e-6, 1.4-2.3x standalone) but it **does not engage** -- on
   this pin the GLM gate weight is BF16 with an fp32 *output* (GateLinear tier 4,
   hipBLASLt, 42 x 150 us = 6.4 ms/step); a BF16-weight/fp32-out variant is the
   follow-up.

Quality after 1+2: NLL 2.2012 / 1.8057 / 0.9594 (BF16 2.2141 / 1.8093 / 0.9596;
int8-linears 2.1938 / 1.8068 / 0.9529) -- inside the band; needles 3/3 at 6.8K
and 13K; opencode replay 3/3. A/B: JSON 21.2, tools 22.5, prose 17.2 tok/s.

**GPU busy dropped 107.6 -> 98.2 ms/step, but the step did not move** (111-116
ms): kernels now start ~0.3 ms after launch (empty queue) -- the CPU is the
floor. Profiled CPU per step ~123 ms: MoE custom op 40 ms (963 us/layer: 387 us
of Python between ops, 202 us in the torch.compile'd `grouped_topk` + 52 us of
Dynamo guard lookups, 57 us router mm, 38 us moe_align), model glue 29 ms, raw
kernel launches 14 ms, mHC 10 ms. Further GPU work will not show until this
shrinks. Next CPU levers: a streamlined decode MoE path (one fused
sigmoid+bias+top-8+renorm routing kernel, HIP GEMVs called directly, no
moe_align/Dynamo for M <= 8), then the mHC wrapper; or graphs (§29).

## 31. Decode MoE host path: fused router + direct HIP MoE (2026-10-03)

Follow-up to section 30's "the CPU is the floor" (MoE custom op ~963 us of host
time per layer). Two pieces, both decode-only (<= 64 token-expert pairs) and
both falling back to the stock path otherwise:

1. **Fused routing kernel** (`vsh-fused-router.py`, `vsh_sigmoid_bias_topk` in
   `libvsh_w8a16.so`, `VSH_FUSED_ROUTER`): `GroupedTopKRouter._compute_routing`
   with sigmoid scoring, a correction bias and n_group == 1 (GLM-5.3) runs one
   256-thread kernel -- sigmoid, +bias, top-8 by biased score, weights from the
   unbiased scores, renormalise, x routed_scaling -- instead of the
   torch.compile'd `grouped_topk` and its Dynamo guards. Unit test
   (`scripts/test_fused_router.py`): ids identical 400/400, weights within 9e-8,
   67 -> 10 us wall per call. Server: needles 3/3, replay 3/3, NLL in band;
   steps 111-116 -> 108-113 ms.
2. **Direct int4 MoE** (`vsh-moe-direct.py`, `vsh_moe_int4.direct_moe`,
   `VSH_MOE_DIRECT`): `CompressedTensorsWNA16MoEMethod.apply` skips the modular
   kernel (prepare/finalize, workspace sizing, config lookup, moe_align_block_size)
   and issues three HIP launches: w13 with one workgroup column per (token, k)
   slot, where the first slot holding an expert gathers every pair routed to it
   and streams that expert's weights once (no alignment pass); w2 with
   `silu(min(g, L)) * clamp(u, -L, L)` fused into its activation staging
   (rounded to bf16 like `silu_and_mul_with_clamp`) and the router weight
   applied; an fp32 top-k sum. Engages only for the plain config (TritonWNA16Experts,
   symmetric int4 group 128, no EP/expert_map, no LoRA, sync prepare/finalize,
   SILU with a clamp limit); the check is cached per layer. Unit test
   (`scripts/test_moe_direct.py`) against the stock pipeline: within 1 bf16 ulp
   (worst rel 0.0074, the same as Triton-vs-HIP), kernel time at parity.

Result (DFlash2 k=3, production config):

| | JSON | tools | prose |
|---|---|---|---|
| step, section 30 | 111 ms | 116 ms | 113 ms |
| step, + fused router | 108 ms | 113 ms | 111 ms |
| step, + direct MoE | **102 ms** | **107 ms** | **105 ms** |
| tok/s, section 30 | 21.2 | 22.5 | 17.2 |
| tok/s, now | 21.5 | **25.9** | 17.9 |

(tok/s also carries acceptance noise: tokens/step 2.21 / 2.79 / 1.88.) Quality:
NLL 2.2016 / 1.8058 / 0.9594 (section 30: 2.2012 / 1.8057 / 0.9594); needles 3/3
at 6.8K and 13K; opencode replay 3/3. The step is now within ~4-9 ms of the
section 30 GPU-busy figure (98.2 ms), so GPU-side work counts again. Remaining
host levers: the mHC wrapper (~10 ms/step), model glue (~29 ms), graphs (section 29).

## 32. Router gate: bf16-weight / fp32-out GEMV (2026-10-03)

The follow-up from section 30 item 3. GLM-5.3's MoE gate on this pin is a BF16
weight [288, 4096] with an fp32 output, so `GateLinear` takes tier 4
(`torch.mm(out_dtype=fp32)` -> hipBLASLt `Cijk_..._BSS_BH`), ~150 us per layer in
the server profile. `vsh-gate-bf16.py` inserts a ROCm branch in front of tier 4:
for <= 8 bf16 rows, `vsh_w8a16.w16_gemv` -> `vsh_w16_gemv` in `libvsh_w8a16.so`
(4 waves per output row, 16 contiguous bf16 weights per lane per step, fp32
accumulate, LDS reduction of the 4 partials). `VSH_ROUTER_GEMV=0` disables it.

Unit test (`scripts/test_gate_bf16.py`, M = 1..8, three shapes): max error vs
fp64 ~4e-7, against ~3e-5 for the hipBLASLt epilogue, so it is *more* exact than tier 4;
288 x 4096, M=4: GPU 51.3 -> 6.6 us, wall 42 -> 9 us per call. A forward-level
check confirmed the branch engages with the gate's tier-4 flags.

Server (DFlash2 k=3): NLL 2.2015 / 1.8001 / 0.9627 (band); needles 3/3 at 6.8K
and 13K; replay 3/3. Steps 102 / 107 / 105 -> **101 / 106 / 104 ms**; tok/s
22.8 / 24.8 / 18.5 (JSON / tools / prose; tokens/step 2.32 / 2.63 / 1.93).
Only ~1 ms of the ~6 ms of GPU time shows up in the step: the rig is still
partly host-bound (mHC wrapper, model glue, launches), so the remaining levers
are host-side or graphs (section 29).

## 33. CUDA graphs work: PIECEWISE breakable capture, TP collectives eager (2026-10-03)

Resolves section 29. In FULL modes (`FULL_DECODE_ONLY`, `FULL_AND_PIECEWISE`)
the breakable wrapper deliberately disables every eager break
(`eager_break_during_capture` returns straight into the op when the runtime
mode is FULL), so the whole forward -- including the TP all-reduces over RCCL --
is one graph, and its first replay hangs inside `ncclDevKernel` (section 29). The root
cause of that RCCL-in-graph hang was not isolated (standalone RCCL graphs over
the same plugin work); this sidesteps it instead of fixing RCCL:

1. **`vsh-cg-eager-collectives.py`** (`distributed/parallel_state.py`):
   `GroupCoordinator.all_reduce / all_gather / reduce_scatter`, when called inside
   an active *breakable* capture (not FULL), end the current graph segment, run
   the collective eagerly -- odl_ar2 for decode sizes, exactly as in eager mode --
   and record a replay step that re-runs it and copies into the capture-time
   result tensor (a fixed address the next segment was captured against).
   Outside a breakable capture nothing changes. `VSH_CG_EAGER_COLLECTIVES=0`
   disables.
2. **`glm53_cg_mode: PIECEWISE`** (new passthrough `VSH_GLM53_CG_MODE` in
   `vsh-cluster-restart.sh`; the default in it and in `vsh-manual-serve.sh` is now
   PIECEWISE) with `glm53_enforce_eager: 0`. Capture sizes stay `1 2 4 8`;
   capture takes ~2 s and 1.56 GiB per rank (target + DFlash2 drafter).
3. **Fused-router NaN bug found by the graph warmup**: the dummy batches of
   capture warmup feed uninitialised hidden states, and with NaN logits the
   section 31 `sigmoid_bias_topk` kernel never selected anything -- it wrote
   expert id 0x7fffffff (and out of LDS), and the direct MoE kernel then faulted
   (`HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION`, rank 1, first boot). NaN now ranks
   highest (as in `torch.topk`), so ids are always valid and distinct
   (`scripts/test_fused_router_nan.py`: all-NaN / partial-NaN / inf / garbage;
   finite inputs unchanged). Real decode inputs never hit this, so eager
   production was not affected.

Result (DFlash2 k=3):

| | JSON | tools | prose |
|---|---|---|---|
| step, eager (section 32) | 101 ms | 106 ms | 104 ms |
| step, PIECEWISE graphs | **96 ms** | **100 ms** | **98 ms** |
| tok/s, graphs | 26.5 | 28.5 | 19.9 |

(tokens/step 2.57 / 2.86 / 1.96 -- acceptance noise inflates the tok/s gain; the
step is the clean number, -5 to -6 ms.) Quality: NLL 2.1948 / 1.8068 / 0.9520;
needles 3/3 at 6.8K and 13K; opencode replay 3/3; image input fine (9.1 s for the
448x448 probe vs 12 s eager). The first request after a boot can stall ~2 min
while Triton JIT-compiles graph-mode kernel variants (rank 1 logged
`FusedQKVRMSNormKernel`, `rotary_kernel`); later requests and later boots hit the
cache. Revert: `glm53_enforce_eager: 1`.

## 34. Cleanup: debug log writers removed; spec-off boots and runs (2026-10-03)

1. **Debug leftovers removed.** The long-context hunt (sections 22-27) left nine ungated
   `try: open("/tmp/glm_*.log", "a") ... except: pass` blocks on the prefill
   path -- every one with host syncs (`.item()`, `int(tensor)`, `.tolist()`,
   `kthvalue`): `models/glm5next/amd/sparse_indexer.py` (kpool, map, scores x2,
   topk), `v1/attention/backends/mla/indexer.py` (idxplan, btfix) and
   `v1/attention/ops/rocm_aiter_mla_sparse.py` (gather, btfix); 122 lines. They
   were still writing (~15 MB of logs). Removed from both containers and
   `container/pinned-vllm/` (the patch-history reference copies and
   `container/patches/debug/` are left as history); stale `/tmp/glm_*.log` deleted.
2. **Spec-off boot fix** (`host/vsh-manual-serve.sh`). With no drafter the
   attention / scheduler block is 2176 tokens (no drafter KV group to pad for),
   and the hard-coded `--prefix-cache-retention-interval 2304` failed engine
   init (`must be ... a multiple of scheduler_block_size (2176)`). When
   speculative decoding is off and the interval is the 2304 default, the serve
   script now passes `VSH_GLM53_APC_RETENTION_NOSPEC` (default 2176).
3. **MTP-off validated -- the section 26 rank-1 crash is gone** (the page aliasing
   fixed in section 27 was the likely cause). Spec decoding off
   (`glm53_spec_method: glm5_next_mtp`, `glm53_mtp_tokens: 0`):

| | CUDA graphs (PIECEWISE) | eager |
|---|---|---|
| boot + first generation | OK | OK |
| NLL (patches / agents / pyfile) | 2.1979 / 1.8059 / 0.9644 | 2.1984 / 1.8052 / 0.9640 |
| needles 6.8K / 13K | 3/3, 3/3 | 3/3, 3/3 |
| opencode replay | 3/3 | 3/3 |
| decode, JSON / prose | 17.0 / 16.8 tok/s | 16.5 / 16.3 tok/s |

   No new errors in either run. Spec-off graphs were 8.8 tok/s on 2026-10-02;
   production stays DFlash2 k=3 (20-28 tok/s).
4. **Production re-check after the cleanup** (DFlash2 k=3, graphs): JSON 96 ms/step,
   23.4 tok/s (tokens/step 2.27); prefill 285-296 tok/s cold at 4.8K-16.7K
   (283 before) -- the removed writers cost nothing measurable at these sizes,
   they were host syncs per layer per chunk. The first large prefill after a
   boot stalled on Triton JIT (11.5K tokens: 183 s, then 40 s on a repeat).


## 35. DS4 native Phase 0 (2026-10-05): template bug found, profile solved, two boot blockers fixed

> **Corrected by §36:** the "chat-template bug" is a decode-path bug (prompts are rendered
> correctly; every prefill is right, decode steps degrade); the compressor-fusion error was W8A16's
> freed weight; the int8 lm_head is numerically fine; "MoE exonerated" held only with MTP off.

Boot: `DS4OV_MTP=0 ~/vsh-ds4-reserve.sh` (no spec). Probes: `scripts/ds4probe.py`,
`scripts/ds4-gen.py`; trace aggregation: `scripts/ds4-trace-agg.py`.

1. **Chat-template bug (the "degenerate output").** The serve default
   `--default-chat-template-kwargs '{"thinking":true,"reasoning_effort":"high"}'`
   renders a corrupt prompt on this build: deterministic prompt-blind garbage
   (' \n * * **'), all needles FAIL. With `thinking:false` the model is
   correct ('Paris', 1.3 s TTFT). Every kernel A/B only "mattered" through
   this bug. Fix for now: thinking:false for DS4 chat requests.
2. **Speed is a separate, profiled problem.** No-spec decode ~580 ms/step:
   `_w8a8_triton_block_scaled_mm` = 10 320 invocations x 2.36 ms = ~507 ms/step
   (~87 %) — the dense FP8 linears on the stock Triton block kernel at decode
   sizes; `_matmul_NNT_bf16xbf16xmxfp4` (MXFP4 MoE) ~99 ms/step; odl wait ~20 ms.
   Phase 1-2 (June's `ds4_fp8_gemv_hip.cpp` / `ds4_moe_mxfp4.cpp` ports) target
   exactly these two.
3. **MoE backend exonerated** for correctness: `moe_backend:"emulation"` via
   the new `VSH_DS4_MOE_KC` kernel-config hook produces the same (correct,
   with thinking:false) output as TRITON_UNFUSED.
4. **Boot blockers:** (a) upstream `prepare_compressor_gemm_fusion` raises on
   the Vision-Exp compressor weights — now fail-soft
   (`container/patches/cgfusion-soft.py`, marker `vsh-cgfusion-soft`; the
   unfused path is pre-fusion behavior); (b) **per-box env divergence**:
   `vsh-cluster-env.odl.sh` is per-box; box2 kept `VSH_W8A16=1` while box1
   was 0 → rank 1 int8-converted its linears (bf16 freed to shape (0,)) and
   rank 0 did not → rank-divergent weights. Keep `VSH_W8A16=0` and
   `VSH_W8A16_LMHEAD=0` (the lm_head int8 sub-hook defaults ON and also
   corrupts DS4 logits) for DS4 boots; SYNC THE ENV FILES ACROSS BOXES.
5. Serve-script hooks added (env-gated, inert by default): `VSH_DS4_MOE_KC`
   (kernel-config JSON fragment) and `VSH_DS4_PROFILER_DIR` (torch profiler),
   both read AFTER the cluster-env source.


## 36. Phase 0 review fixes, the real DS4 failure, and the FP8 GEMV (2026-10-05)

1. **Per-box env divergence, live GLM regression.** After §35, box2's
   `vsh-cluster-env.odl.sh` still defaulted `VSH_W8A16` to 0 (box1: 1), and GLM was
   restarted that way: rank 0 int8 linears, rank 1 bf16 — 106 / 108 ms/step vs 96 / 98.
   Fixed: env files synced; the restart scripts now pass each model's values to *both*
   boxes through `ENVPASS` (GLM `VSH_GLM53_W8A16=1 VSH_W8A16_LMHEAD=1`; DS4 `0 0`, plus
   `VSH_FP8_GEMV` / `VSH_SPARSE_ATTN_SPLIT` knobs), and refuse to start when the two
   boxes' copies of the env file differ (md5). GLM back to 96 / 101 / 98 ms/step.
2. **DS4 debug leftovers / rank-divergent code.** box1's — and `pinned-vllm`'s —
   `models/deepseek_v4/amd/{model.py,rocm.py}` still carried 09-30 diagnostics (a
   `/tmp/pfbeat` writer with a host sync on prefill; per-layer `DS4_LAYER_DUMP` hooks,
   armed by the DS4 `ENVPASS`) that box2 did not. Removed everywhere (both boxes and
   `pinned-vllm` now identical: rocm.py `1b2d260a`, model.py `80910c3e`); the
   `vsh-cgdiag` instrumentation dropped, `vsh-cgfusion-soft` kept. The DS4 serve /
   restart / reserve scripts are now tracked in `host/`.
3. **Corrections to §35.** The compressor-fusion error was W8A16's freed `(0,)` weight
   on rank 1 (§35's own diagnostic checks `vsh_w8_q`), not an upstream bug. The int8
   lm_head is numerically fine on DS4's real head (rel 6.7e-3, top-1 agreement 100 %);
   it stays off for DS4 only as a precaution. "MoE exonerated" held only with MTP off —
   upstream's `TRITON_UNFUSED` gate is MTP-specific and still untested here.
4. **The real DS4 failure: decode, not the template.** The 0.31 renderer output is
   well-formed (thinking: effort preamble + `<｜User｜>…<｜Assistant｜><think>`, 95
   tokens; non-thinking: 16 tokens ending `</think>`; `scripts/ds4_render_prompt.py`).
   Battery `scripts/ds4len.py`, spec off, thinking off:

   | probe | result |
   |---|---|
   | chat, prompt 16 / 94 / 238 / 450 / 1102 tokens | first token right every time ('Paris') |
   | raw completion, prompt 22 … 548 tokens | first token ' Paris' every time |
   | raw continuation after it | coherent at 22-81 tokens; degrades ~120 ('Paris. (3) The The the'); debris from token 2 past ~150 |
   | chat "count 1..100" | '1, 2, 3, 4, 5, 5, 5, 6, 6, 6' |
   | 8K needle | stops after 'AM' |
   | thinking on, short prompt (95 tokens) | degenerate ('We need just answer1 answer…') |

   **Prefill is correct; every decode step degrades, faster the longer the context.**
   The thinking template only "mattered" by adding 79 tokens. The bug is pre-existing:
   identical failures with split-KV attention off and with split-KV + FP8 GEMV both off
   (the stock path). Prime suspect: the BF16-KV decode path — June served DS4 with FP8 KV;
   native runs BF16 because FP8 KV asserts in aiter on this build — i.e. the decode-time
   compressor / SWA cache updates and the decode attention over them.
5. **FP8 GEMV** (`vsh-fp8-gemv.py`, `vsh_fp8_gemv` in `libvsh_w8a16.so`, `VSH_FP8_GEMV`):
   on gfx1151 with M <= 8, `TritonFp8BlockScaledMMKernel.apply_weights` runs a HIP GEMV
   on the bf16 activation (no fp8 activation round trip), and `apply_block_scaled_mm`
   with pre-quantized fp8 activations (DS4's fused q-norm → `wq_b`) dequantizes the
   1x128 groups first. One wave per output row, fp8 decoded in registers (e4m3fn and
   e4m3fnuz, exact subnormals), 128x128 block scales, padded rows. Unit tests
   (`scripts/test_fp8_gemv.py`, `test_fp8_preq.py`): max rel err 1.8e-3 vs exact (stock
   W8A8 2.6e-2); 16-91x faster per call (16384x1024, M=1: 54 us vs 1.51 ms). DS4, spec
   off: decode ~150 ms/token vs ~580 (5.6-6.4 vs 1.68 tok/s incl. prefill), same outputs
   as stock on the battery (the decode bug is unchanged). GLM has no FP8 linears.
   Next: fix the decode bug; then the MXFP4 MoE port (~99 ms/step) and DSpark.
