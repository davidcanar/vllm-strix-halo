# vllm-strix-halo — GLM-5.3-Flash (and DeepSeek-V4-Flash) on 2× AMD Strix Halo, TP=2 over Thunderbolt

> **STATUS 2026-10-03 — working end to end.** GLM-5.3-Flash serves correctly:
> long-context retrieval passes at every tested length (mid-context and tail,
> 6.8K–255K, including adversarial), tool calling works (8/8 on the captured
> opencode request), image input works, prefix caching works, and decode is
> **20–28 tok/s** (DFlash2 k=3, CUDA graphs) after the §28–§33 decode rounds.
> DS4 (DeepSeek-V4-Flash) runs on the June 2026 stack (`ds4_engine: delegate`,
> 19–24 tok/s); the native 0.31 port is still open (§ Known issues).

| metric (GLM-5.3-Flash AWQ W4A16, TP=2, DFlash2 k=3, CUDA graphs) | value |
|---|---|
| decode | **26.5 / 28.5 / 19.9 tok/s** JSON / tools / prose (`vsh_ab.py`), 96–100 ms/step — was 12.0 tok/s before §28 |
| prefill | 285–296 tok/s cold at 4.8K–16.7K (the first large prefill after a boot can stall on Triton JIT) |
| TTFT, 13.9K context | ~50 s cold → **0.95 s cached** (§20) |
| long-context retrieval | ✅ 3/3 mid-context at 6.8K/13K/32K; tail needles PASS to 255K; adversarial PASS |
| tool calling | ✅ 8/8 on the captured opencode request; opencode replay 3/3 |
| image input | ✅ (vision encoder in BF16; 448×448 probe answered correctly, 9.1 s) |

The detailed history — every bug, fix, measurement and dead end — lives in
[PATCHES.md](PATCHES.md) (§1–§34). This README is the current state only.

---

## What this is

vLLM serving across **two AMD Strix Halo (gfx1151) boxes** — one 8060S iGPU
per box, tensor-parallel (TP=2), with the inter-GPU fabric carried over a
**Thunderbolt-4/USB4** cable via the **OdinLink driver** (`odl_tb5`): RCCL
rides the OdinLink net plugin for big prefill collectives, the native
`odl_ar2` carries the small decode all-reduces, and `odl_mq` (fail-open)
moves the EngineCore↔worker control plane off per-step TCP. `transport: tcp`
(plain sockets) remains the fallback.

GLM-5.3-Flash support is upstream vLLM since 2026-09-03
([#53906](https://github.com/vllm-project/vllm/pull/53906)); DeepSeek-V4-Flash
model code is upstream since the 2026-09-27 wave. This repo pins vLLM at
`73859fec` on a kyuz0 ROCm-10.0/torch-2.11 base and adds the OdinLink
plumbing plus the gfx1151 fixes below.

## Quick start

```bash
~/vllm-strix-halo.sh glm53 start     # GLM-5.3-Flash on :1234
~/vllm-strix-halo.sh ds4 start       # DS4 (engine per ds4_engine, see below)
~/vllm-strix-halo.sh glm53 stop      # ... etc
```

Both models share the `vllm-glm` container and API port 1234 — either/or at
runtime. Site config: `~/vsh-config.yaml` (flat keys → `VSH_*` env vars;
template in `host/vsh-config.yaml`). The template ships `ds4_engine: native`
(the 0.31 port, not production-ready); set `ds4_engine: delegate` for the
June stack.

## Patch set (current)

**Long-context sparse attention (§27, 2026-10-02/03)** — four stacked bugs;
without these, decode indexer scoring was random past ~2K tokens and every
pool past ~4K aliased onto one page:

| patch | what it fixes |
|---|---|
| `59412-pooled-indexer-kernel-blocks.py` | Port of upstream #59412: page-aligned kernel blocks for the kpool indexer |
| `vsh-kpool-paged-logits.py` | Correct decode indexer reader for the paged SHUFFLE cache on gfx1151 (aiter stage1 scored at random); also drops a 12.6 ms/step fp8 copy. Env `VSH_KPOOL_PAGED_LOGITS` |
| `vsh-idx-force-tail-boost.py` | Force-tail boosts recent pools *before* top-k instead of overwriting (unsorted) columns |
| `vsh-idx-fp8max-fnuz.py` | Indexer K quantized to 224 so the e4m3fn→fnuz cast cannot overflow to NaN. Env `VSH_IDX_FP8MAX_FNUZ` |

**Decode + cache (2026-10-02/03):**

| patch | what it does |
|---|---|
| `vsh-glm53-apc-align.py` | EAGLE last-block drop resolved to pure-drafter groups (upstream flags every group) — prefix hits stop losing a 2304-token page per lookup. Env `VSH_GLM53_APC_ALIGN` (on). §18 |
| `vsh-apc-retention.py` | `--prefix-cache-retention-interval 2304` (stock flag; 2176 when speculative decoding is off — the scheduler block shrinks): keeps KDA/Mamba state checkpoints so the *first* repeat hits. Env `VSH_GLM53_APC_RETENTION` (on). §20, §34 |
| `vsh-adaptive-k.py` | EMA policy for the verified draft length, wired for async scheduling + live override. Installed, instrumented, **off** — step time is draft-length independent here. §17 |
| `vsh-triton-ptr-cache.py` | Memoises Triton's per-pointer `hipPointerGetAttribute`. Validated bit-exact, **off** — worth ~0.3 % of a step. §21 |
| `host/moe-configs/E=288,N=1024,...json` | Tuned MoE tiles under the filename the runtime actually asks for |

**Decode speed (§28–§33, 2026-10-03):**

| patch | what it does |
|---|---|
| `host/vsh-gpu-sclk.sh` (`glm53_gpu_sclk_max: 2100`) | iGPU clock cap so the shared power budget stops clamping the CPU. §28 |
| `vsh-router-packed-topk.py`, bf16 MLA BMM (`VLLM_ROCM_USE_AITER_FP8BMM=0`) | Cheaper deterministic router top-k; no fp8 BMM round trip. §28 |
| `vsh-sparse-attn-split.py` | Split-KV sparse decode attention. §28 |
| `vsh-moe-int4-hip.py` + `vsh_moe_int4.hip` | HIP int4 MoE decode GEMV (routed MoE 90 → 37 ms/step). Env `VSH_MOE_INT4_HIP`. §28.5 |
| `vsh-w8a16-linear.py` + `vsh_w8a16.hip` | Load-time int8 (group 128) for the checkpoint's BF16 linears + W8A16 decode GEMV. Env `VSH_W8A16`. §28 |
| DFlash2 k=3 (`glm53_spec_method: dflash`) | Default drafter; its attention on `TRITON_ATTN` (`VSH_GLM53_DRAFT_ATTN`). §28.7, §30 |
| `vsh-gate-lmhead.py` | int8 `lm_head` copy for target + drafter logits. Env `VSH_W8A16_LMHEAD`. §30 |
| `vsh-fused-router.py` | One HIP kernel for sigmoid + bias + top-8 + renorm routing. Env `VSH_FUSED_ROUTER`. §31, §33 |
| `vsh-moe-direct.py` | Decode MoE straight to HIP (no modular kernel / `moe_align`). Env `VSH_MOE_DIRECT`. §31 |
| `vsh-gate-bf16.py` | bf16-weight / fp32-out router-gate GEMV (replaces hipBLASLt tier 4). Env `VSH_ROUTER_GEMV`. §32 |
| `vsh-cg-eager-collectives.py` (`glm53_cg_mode: PIECEWISE`) | CUDA graphs: breakable piecewise capture with every TP collective eager. Env `VSH_CG_EAGER_COLLECTIVES`. §33 |

**Still carried from earlier rounds:** #55222 workspace-units ports (indexer +
GLM attention), #57979 KDA stride fix, upstream ragged-index rewrite, the
gfx1151 platform gates (AITER unlock, fp8 fnuz, pre-shuffle skips, mHC,
deterministic top-k, WNA16 OOM), and the OdinLink hooks (`odl_ar2`, `odl_mq`).
The older gather-site block-table expansion (`vsh-idx-bt-gather-v4-ops.py`)
and column-overwrite force-tail are superseded by §27 and are inert.

**DS4 native port** (`container/patches/ds4-native-fixed-refs/`): boots and
serves; decode quality/speed unresolved (Known issues 5).

## Architecture

```
┌──────────────────────────────┐       ┌──────────────────────────────┐
│ box1 (192.168.0.102)         │       │ box2 (10.0.2.2)              │
│  ┌────────────────────────┐  │  TB4  │  ┌────────────────────────┐  │
│  │ vllm-glm container     │  │◄─────►│  │ vllm-glm container     │  │
│  │  vllm 0.31.0-dev       │  │odl_tb5│  │  (ray worker)          │  │
│  │  GLM-5.3-Flash AWQ     │  │       │  │  GPU 1/2               │  │
│  │  ray head :6379        │  │       │  │                        │  │
│  │  API :1234             │  │       │  │                        │  │
│  └────────────────────────┘  │       │  └────────────────────────┘  │
└──────────────────────────────┘       └──────────────────────────────┘
```

- **Image**: `vllm-strix-halo:local` from `container/Dockerfile` (base
  `kyuz0/vllm-therock-gfx1151:rocm10.0.0-torch2.11.0-vllm0.30.0`; vLLM pinned
  at `73859fec`; `container/pinned-vllm/` + `container/pinned-triton/` carry
  the byte-exact patched files for reproducible rebuilds). Snapshot of the
  running stack on both boxes: `vllm-strix-halo:glm-perf34-20261003`
  (`:local` points at it)
- **KV cache**: 16 GiB pinned, 256K max context
- **Speculative decoding**: DFlash2 drafter, k=3 (`glm53_spec_method: dflash`,
  `glm53_mtp_tokens: 3`); MTP (`glm5_next_mtp`) is the fallback and spec-off
  works too (§34)
- **Execution**: CUDA graphs, PIECEWISE breakable capture, sizes 1/2/4/8;
  `glm53_enforce_eager: 1` reverts to eager (§33)
- **Quantization**: AWQ W4A16 for MoE experts (checkpoint). The checkpoint's
  BF16 linears (KDA/MLA projections, shared experts, dense MLPs) are
  re-quantized to int8 group-128 at load (§28); `lm_head` gets an int8 copy
  (§30); the router gate, `kv_b_proj`, `wk_weights_proj`, embeddings and the
  vision encoder stay BF16

## Measured performance (2026-10-03)

See the table at the top. Receipts and methods:
`scripts/vsh_ab.py` (matched A/B with engine-side step accounting + tool-call
gate), `scripts/vsh_bench.py` (per-request decode/TTFT with spec and
prefix-cache deltas), `scripts/stepbench.py` (unprofiled decode step time),
`scripts/trace_*.py` (kernel share, GPU duty cycle, copies, blocking ops),
`scripts/bf16_bw.py` (linear bandwidth at real weight shapes). Numbers are
medians of ≥2 boots or ≥5 repetitions.

## Repository layout

```
vllm-strix-halo/
├── vllm-strix-halo.sh       # supervisor launcher (glm53 | ds4)
├── host/                    # host-side scripts (restart, serve, env, heal, sclk, moe-configs)
│   └── vsh-config.yaml      # site config template
├── container/
│   ├── Dockerfile           # image build (base + patches)
│   ├── patches/             # patch scripts, HIP kernels + reference copies (+ debug/, ds4 refs)
│   ├── pinned-vllm/         # byte-exact patched vllm files (authoritative rebuild input)
│   └── pinned-triton/       # pinned Triton AMD driver
├── odinlink/                # OdinLink (odl_tb5) driver patch, odl_ar2, build/install scripts
├── scripts/                 # harnesses + unit tests (vsh_ab, stepbench, nll, test_*, trace_*)
├── PATCHES.md               # detailed patch history and lessons (§1–§34)
├── FRESH-EYES-20tps.md      # decode-speed analysis behind §28–§33
└── README.md                # this file
```

## Known issues and next steps

1. **Decode speed** — 96–100 ms/step, 26.5 / 28.5 / 19.9 tok/s
   (JSON / tools / prose, DFlash2 k=3, CUDA graphs) after §28–§33; 12 tok/s
   at the start. Done: GPU clock cap, HIP int4 MoE, int8 BF16 linears
   (W8A16), int8 `lm_head`, split-KV sparse attention, fused router, direct
   decode MoE, bf16 router-gate GEMV, piecewise CUDA graphs. Remaining
   levers: the mHC wrapper and model glue on the host, fewer eager breaks
   per layer. See [FRESH-EYES-20tps.md](FRESH-EYES-20tps.md).
2. **CUDA graphs** — on by default since §33: PIECEWISE breakable capture,
   every TP collective an eager break (odl_ar2 serves them as in eager
   mode). FULL modes still hang on the first replay (RCCL inside the graph,
   §29); root cause not isolated. The first request after a boot can stall
   ~2 min on Triton JIT. Revert with `glm53_enforce_eager: 1`.
3. **Prefill MoE is untuned** — 41.5 % of prefill GPU time runs at 12–18 % of
   achievable FLOPS (tuned tiles cover M≤512; prefill chunks run M~65k).
4. **Upstream** — drop `59412-*` (and the inert gather-site expansion) at the
   next rebase; worth reporting: gfx1151's `rocm_fp8_paged_mqa_logits`
   stage1 fallback scores paged SHUFFLE caches at random (our
   `vsh-kpool-paged-logits` replaces it).
7. **DS4 native decode** — repetition loops, 2–6 t/s (last boot 09-30). The
   ratio-1/2-indexer lead is wrong for this model: DS4-Flash only has ratio-4
   indexer caches, already routed to a correct reader. Phase-0 diagnosis
   (2026-10-05) found the boot itself now fails on rank 1: the C4 compressor
   `fused_wkv_wgate` weight loads as an empty 1-D tensor (rank 0 is fine and
   fuses 21 layers; the compressor GEMM fusion is now fail-soft, marker
   `vsh-cgfusion-soft`). Prime suspect: a shared loader change in the §28–§34
   rounds — DS4's own files are byte-identical to 09-30; the delta is the
   full-file `linear.py` overlay (05410c9) and `fused_moe.py` (b521be0).
   The W8A16 hazard is real but separate: `VSH_W8A16=1` int8-converts DS4's
   compressor/indexer/vision-aligner linears — keep it 0 for DS4 boots
   (env default flipped for the window). Next: bisect `linear.py`'s
   weight_loader for the rank-1 shard miss, then the original §34 plan.
