# Fresh-eyes review: GLM-5.3-Flash decode on 2x Strix Halo — is ~20 t/s reachable?
Date: 2026-10-02 · Read-only review of the live rig (`vllm-glm`, pin `73859fec`, MTP k=3, eager).
Nothing on the boxes was modified; three short probe requests were sent to `:1234`.
Scripts used are in `fresh-eyes/` (run against the traces in `box1:~/glmprof/`).

---

## 0. TL;DR

* **Yes, ~20 t/s is physically reachable, but not from any single change.** Your ds4
  result (~19 t/s on this exact pair of boxes, Q4 experts + Q8 KDA/head) is the
  existence proof: if that is plain decode, it is ~53 ms/token. vLLM today runs a
  **186 ms step at short context / 217 ms at 3K** (measured today, unprofiled), at
  2.0–2.8 accepted tokens per step.
* **Two measurement errors in the existing analysis hid the opportunity:**
  1. **Every per-step GPU number in REPORT-20tps.md and PATCHES §12/§21/§22/§23 is
     halved.** vLLM annotates each step on *two* GPU streams; the analysis divided
     by the annotation count (96, 196) instead of the step count (48, 98). The MoE is
     **76–85 ms/step at ~90 GB/s (~42 % of achievable)**, not "42.7 ms at 87 %".
  2. **"The rig is GPU-bound, not CPU-bound" (§22) is wrong for steady-state decode.**
     CPU and GPU cost per step are *both* ~190–220 ms. Except for the first two steps
     after a prefill, kernels start **0.3–0.5 ms after launch** (empty queue). This is
     why every GPU-only change since September (tuned MoE config, ptr-cache, k-sweep,
     DFlash, shm spin) measured as noise. Each one sat under a CPU floor.
* **~50 ms/step of GPU time is plain waste** that small patches can remove (§3). The
  biggest item is a 12.6 ms/step fp8 conversion of the *entire* indexer K-cache
  pool, done on every MLA layer of every step.
* **The BF16 requant is worth ~2x what the agents estimated.** BF16 traffic is
  **~10.9 GB/rank/step**, not 6.47 GB. The lm_head alone is read **4x per step**
  (target + 3 MTP drafts).
* **CUDA graphs: solvable. They are the enabler, not the win** (§5). On their own
  they are worth ~0–10 %. Their value is that afterwards every GPU fix shows up 1:1.

---

## 1. Corrections to the current picture (with evidence)

### 1.1 Per-step GPU times were divided by 2x too many steps

| trace | CPU `execute_context..generation_1(4)` | GPU annotations | used as "steps" |
|---|---:|---:|---:|
| `...1790930314...` (REPORT-20tps, §12/§21/§23) | **48** | 96 (streams 4 + 9) | 96 |
| `...1790978940...` (duty.sh, 16:09) | **98** | 196 (streams 4 + 9) | 196 (`profiler_out_0.txt`) |

`fused_moe_kernel_gptq_awq` in the first trace totals 4.09 s → **85 ms/step**, not 42.7.
In the second, 76.3 ms/step over 84 launches (42 layers × w13/w2). At ~26–30 unique experts
per layer for a 4-row verify (6.5 MB each per rank), that is **~90–95 GB/s**. It matches the
isolated sweep in §6 (M=4 at 2.09 ms/layer ≈ 93 GB/s). **The MoE kernel is at ~42 % of
the ~215 GB/s this GPU sustains on BF16 GEMVs. There is ~35 ms/step of headroom, not 5.**
(Worth confirming by counting unique experts per layer; random routing gives 30.)

### 1.2 The BF16 byte inventory: 10.9 GB/rank/step, not 6.47

Reconstructed from the per-kernel sequence of one step (`onestep.py`):

| BF16 tensor class | GB/rank/step | ms/step | GB/s | in §23? |
|---|---:|---:|---:|---|
| KDA `in_proj_qkvbfg_a` + `o_proj` + f/g_b (34 layers) | 4.72 | 21.6 | 219 | yes |
| MLA `fused_qkv_a`, `q_b`, indexer `wq_b`/`wk`, `o_proj` (11 layers) | 1.43 | 7.2 | 199 | 0.55 |
| shared experts (42 layers) | 1.06 | 6.4 | 165 | **missing** |
| dense MLP layers 0–2 | 0.45 | 2.0 | 227 | **missing** |
| `lm_head` (vocab-parallel: **0.63 GB/rank**, not 1.27) **× 4 reads/step** | 2.53 | 10.9 | 238 | 1 read |
| MTP `eh_proj`/MLA/shared + fp32 router gates | ~0.7 | ~5.5 | — | no |
| **total** | **~10.9** | **~53** | | 6.47 |

The kernels are at roofline (that part of §23 holds), so the prize is purely bytes.
**8-bit weights save ~25 ms/step at MTP k=3 (~12 % of GPU time, +14 % t/s once
GPU-bound)**, and ~18 ms of a ~55–75 ms step on the non-spec route (§6).

### 1.3 Steady-state decode is CPU-bound (or balanced), not GPU-bound

`qdepth.py`: kernel *launch→start* latency, per step:

| step | period | launch→start median | regime |
|---|---:|---:|---|
| 0–1 after each prefill | 190 ms | **67–142 ms** | deep queue, GPU-bound, 6 ms idle |
| every later step | 260–275 ms | **0.3–0.5 ms** | empty queue, GPU waits for CPU, ~75 ms idle |

The profiler inflates the CPU side (~15 k aten ops/step). Unprofiled (`itl.py`,
today): **186 ms/step at 24-token context** (≈ GPU time, 185 ms) and **217 ms at 3.1 K**
(GPU ≈ 195–205 ms with the indexer active). §17 shows the CPU floor directly: k = 1/2/3
all ran at 225–227 ms at 2.4 K, even though k=1 removes ~40–50 ms of GPU work (smaller
MoE verify, two fewer draft forwards). CPU ≈ GPU, so a step only gets faster when *both*
get faster.

Where the CPU goes (profiled, `cpuside.py`; unprofiled is lower but has the same shape):
`moe_forward_shared` **2.16 ms per layer × 45 = 97 ms** (46 ms of it pure Python
self-time), model glue 58 ms, mHC 22 ms, MLA 21 ms, GEMM wrappers 17 ms, all-reduce
12 ms, Dynamo guard lookups for the compiled router ~6 ms. There are **no blocking
host syncs** in decode: the 228–240 `item()` calls/step are on CPU tensors (~2 µs each).

---

## 2. GPU budget of one decode step (MTP k=3, ~2 K context, indexer active)

| component | ms/step | calls | note |
|---|---:|---:|---|
| routed MoE int4 `fused_moe_kernel_gptq_awq` | 76.3 | 84 | ~90 GB/s, BM=32 padding, 2-wave WGs |
| BF16 linears (`wvSplitK` + hipBLASLt), incl. lm_head ×4 | ~53 | ~350 | at roofline; bytes are the issue |
| **`float8_copy`: whole indexer K-cache e4m3fn→fnuz** | **12.6** | 12 | see §3.2 |
| **aiter fp8 BMM (MLA W_UK/W_UV), emulated on RDNA3.5** | **10.7** | 28 | ~0.38 ms each; see §3.1 |
| **router: inductor persistent sort kernel** | **10.6** | 45 | grid [2]×32 threads, 242 µs each |
| MTP layer-45 experts, BF16 `fused_moe_kernel` | 8.2 | 6 | 3.9 + 1.5 ms at M=4 |
| **sparse MLA attention via the *prefill* ragged kernel** | **7.3** | 14 | grid [4,2] = **8 workgroups** |
| mHC pre/post/fused | 4.3 | ~180 | |
| odl_ar2 wait + doorbell + D2H stage | 3.6 | 100 | |
| fp32 router gate GEMM (hipBLASLt MT16x32x64) | 2.8 | 45 | ~52 GB/s on a 4.7 MB matrix |
| indexer head-sum over a `max_model_len`-sized buffer | 2.6 | 12 | 33.5 MB read for ~1 K valid pools |
| MQA logits + top-k | 1.7 | 24 | |
| KDA recurrent + conv | 1.4 | 68 | |
| logits all-gather | 1.3 | 4 | |
| ~1,400 other small kernels | ~6 | | |
| **total** | **~200** | ~2,340 | |

---

## 3. Cheap GPU fixes (small patches, ~50 ms/step combined)

**Measure each one with a profile (GPU busy ms/step), not tok/s.** Under today's CPU
floor most of them will read as noise end to end. That is the trap the earlier rounds fell into.

| # | fix | saves (k=3) | effort / risk |
|---|---|---:|---|
| 3.1 | `VLLM_ROCM_USE_AITER_FP8BMM=0` (must reach the Ray worker env). MLA then uses `torch.bmm` on bf16 W_UK/W_UV (`mla_attention.py:1412`). RDNA3.5 has no fp8 dot, so the aiter triton a8w8 BMM runs at ~10 GB/s. It also skips the 2×1024-shape fp8-BMM precompile at boot (~50 s). | ~9 ms | env only; more precise |
| 3.2 | `aiter/ops/triton/attention/pa_mqa_logits.py` (the vsh overlay) runs `kv_cache_fp8.view(e4m3fn).to(e4m3fnuz).contiguous()` on the **whole pool** (~43 M elements/layer at 16 GiB KV) on every call. e4m3fnuz(bits) = e4m3fn(bits)/2 for every finite value, so reinterpret with zero copies, fold ×2 into `weights`/scale, and sanitize 0x80 (fn −0 = fnuz NaN) in-kernel or at cache write. The cost also scales with `glm53_kv_bytes`. | 12.6 ms | small; validate with the needle suite |
| 3.3 | Router (`vsh-moe-router-deterministic-topk`): the stable `sort` over 288 compiles to a 2-workgroup persistent kernel (242 µs at M=4). Same total order, fast: pack `orderable(fp32) << 32 \| (0xFFFFFFFF − idx)` into int64 and `topk(sorted=True)`. Keys are unique, so it's deterministic. Skip the group stage when `n_group == 1`. | ~10 ms GPU + some CPU | small; bit-identical selection |
| 3.4 | Sparse MLA decode on `_sparse_attn_prefill_ragged_kernel` launches tokens×2 workgroups (8 at M=4, 2 at M=1). Split the 2,048 selected keys flash-decoding style. | ~5.5 ms | moderate (Triton) |
| 3.5 | Indexer logits buffer pitched by `max_model_len` (262 144 → 65 536 pools): the head-sum `aten::sum` reads 33.5 MB/layer. Size it to the batch's real max context, or use the fused (non-stage1) logits kernel. | ~2.3 ms | small–moderate |
| 3.6 | fp32 router gate on hipBLASLt at ~52 GB/s: use the skinny path / bf16 copy for M ≤ 8. | ~2 ms | small |
| 3.7 | lm_head is read 4×/step (2.7 ms each). Give the MTP head a truncated draft vocab (EAGLE-3 style, ~32 K), or Q8 the head. | 5–8 ms | moderate |
| 3.8 | MTP layer-45 experts BF16 → int4 (the §12 shard-patch). This also frees ~7 GB/rank. | ~5 ms | moderate; drafter-only risk |

---

## 4. The two big GPU levers (real kernel work)

1. **Int4 MoE decode kernel: 76 → ~40 ms/step at k=3, 27 → ~12 ms at M=1.** The Triton
   kernel pads every expert to BLOCK_M=32 and runs 2-wave workgroups that stream
   1 KB of weights per K-step. A per-expert skinny GEMV (wvSplitK-style, split-K,
   1–4 rows, no padding) should reach ~180 GB/s. ds4's gfx1151 Q4_K MoE decode
   kernels are a working reference on this exact GPU.
2. **8-bit weights for the BF16 classes in §1.2: −20 to −25 ms/step.** This is *not*
   data-only on this stack. The only dense weight-only kernels for gfx11 in this pin are
   4-bit (`RDNAHybridW4A16`, `TritonW4A16`), so int8 needs a W8A16 skinny kernel
   (wvSplitK adapted to int8 + per-channel/32-block scales). The requant itself
   follows the existing `requant-fp8attn.py` shard-patch. Cheaper partial step that
   works today: W4A16 for shared experts + dense layers 0–2 (MLP weights, same
   precision class as the routed experts). That is ~1.5 GB → 0.4 GB, about −5 ms.

---

## 5. CUDA graphs: can the blockers be solved? Yes

| blocker | what is actually happening | fix |
|---|---|---|
| **odl_ar2 is graph-unsafe by design** | `odl2_enqueue` does `host_seq++` on the CPU and bakes `seq` and the slot address into captured kernel args/memcpy. `OdlAllReduce2.eligible()` returns False while capturing, so **every graphed all-reduce silently falls back to RCCL**, while eager segments keep using odl_ar2: two transports in one run. | Device-side round counter: a stage-copy kernel computes the slot from `atomicAdd(dev_seq)` and writes the doorbell, and the wait kernel reads the same seq. The CPU progress thread is unchanged. ~50 lines in `odl_ar2.hip`. |
| KDA `prepare_chunk_indices` `.tolist()` sync at replay | Only the chunked (prefill-bearing) path calls it (§22). | `cudagraph_mode=FULL_DECODE_ONLY` so mixed steps run eager, plus pass host-side lengths the way `gdn_attn.py:201` already does (the #51540 fix). Also check that graphed pure-decode steps don't get classified as prefill by the KDA metadata builder. |
| drafter "rebuilding attention metadata between draft steps" | Not graphable on these backends. | Leave the drafter eager. Its ~25 ms of CPU queues behind the ~160 ms target graph, so it is hidden. |
| graph memory 11–13 GiB/rank | Capture list 1..32 + speculator. | Single-user needs sizes 1, 2, 4 (k=3 verify). §11 measured ~2.7–3.3 GiB for 7 sizes. |

**MTP-off graphs already work** (§11 retry: 114 ms/token, GPU-bound). That makes them
the ideal testbed: every GPU fix shows there directly.

---

## 6. Projection (sums of kernel-level estimates; expect 60–80 % to land)

**Route A: MTP off + graphs** (works today, low risk)

| step | ms/token | t/s |
|---|---:|---:|
| today (§11 retry) | 114 | 8.8 |
| + §3.1/3.2/3.4/3.5/3.6 (M=1 sizes) | ~88 | ~11.4 |
| + MoE decode kernel | ~73 | ~13.7 |
| + 8-bit BF16 classes | ~55 | ~18 |
| + graph-safe odl_ar2 (vs RCCL in graph, unmeasured) | ~50 | ~20 |

**Route B: MTP k=3 + graphs** (needs §5 landed)

| step | ms/step | prose (2.0 tok) | no-think (2.75) |
|---|---:|---:|---:|
| GPU time today | ~200 | (CPU floor ~217) | |
| + all of §3 | ~148 | 13.5 | 18.6 |
| + MoE decode kernel | ~113 | 17.7 | 24 |
| + 8-bit BF16 classes | ~93–100 | 20–21.5 | 27–29 |

Notes: the M=4 verify reads ~3.3× the expert bytes of M=1. For prose, MTP's margin
over Route A is therefore small once both are lean. For structured/no-think output,
Route B is clearly ahead. Route A is the safer first target and the better measurement bench.

---

## 7. Suggested order

1. **Measure with GPU busy ms/step** (profile + `steps.py`) next to tok/s, always. Under
   the CPU floor tok/s is the wrong instrument.
2. Boot the **Route A testbed** (`glm53_enforce_eager: 0`, `glm53_mtp_tokens: 0`,
   capture sizes `1 2 4`) and re-baseline.
3. Land §3.1 → §3.2 → §3.3 one per boot on the testbed. Each should be visible there.
4. In parallel: graph-safe odl_ar2; `FULL_DECODE_ONLY` + MTP + eager drafter.
5. MoE decode kernel.
6. W8A16 kernel + 8-bit shard-patch (W4A16 shared/dense first if you want an early win).

## Appendix: reproduction

```
python fresh-eyes/steps.py    <trace.json.gz>   # per-step period / GPU busy / kernel table
python fresh-eyes/steplist.py <trace.json.gz>   # every step: busy, idle, indexer on/off
python fresh-eyes/qdepth.py   <trace.json.gz>   # launch->start latency = CPU- vs GPU-bound
python fresh-eyes/onestep.py  <trace.json.gz> 30  # kernel sequence of one step, with gaps
python fresh-eyes/stacks.py   <trace.json.gz> float8_copy _batched_gemm_a8w8 sigmoid_sort_sum
python fresh-eyes/cpuside.py  <trace.json.gz>   # CPU ms/step by op (inclusive / self)
python fresh-eyes/syncs.py    <trace.json.gz>   # blocking host syncs per step
python3 fresh-eyes/itl.py "<prompt>" 200 300    # unprofiled per-step latency via streaming
```
