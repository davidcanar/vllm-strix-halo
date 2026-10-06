# vllm-strix-halo — GLM-5.3-Flash (and DeepSeek-V4-Flash) on 2× AMD Strix Halo, TP=2 over Thunderbolt

> **STATUS 2026-10-06 — working end to end.** GLM-5.3-Flash serves correctly:
> long-context retrieval passes at every tested length (mid-context and tail,
> 6.8K–255K, including adversarial), tool calling works (8/8 on the captured
> opencode request), image input works, prefix caching works, and decode is
> **20–28 tok/s** (DFlash2 k=3, CUDA graphs) after the §28–§33 decode rounds.
> DS4 (DeepSeek-V4-Flash-Vision-Exp) runs on the native 0.31 port: correct
> since §37, **25–27 tok/s** with DSpark (§41; the June stack via
> `ds4_engine: delegate` measures 17–24), image input works, 512K context fits
> (needles pass at the full 512K window), and prefill runs ~330–460 tok/s up to
> 128K (§42). Table below.

| metric (GLM-5.3-Flash AWQ W4A16, TP=2, DFlash2 k=3, CUDA graphs) | value |
|---|---|
| decode | **26.5 / 28.5 / 19.9 tok/s** JSON / tools / prose (`vsh_ab.py`), 96–100 ms/step — was 12.0 tok/s before §28 |
| prefill | 285–296 tok/s cold at 4.8K–16.7K (the first large prefill after a boot can stall on Triton JIT) |
| TTFT, 13.9K context | ~50 s cold → **0.95 s cached** (§20) |
| long-context retrieval | ✅ 3/3 mid-context at 6.8K/13K/32K; tail needles PASS to 255K; adversarial PASS |
| tool calling | ✅ 8/8 on the captured opencode request; opencode replay 3/3 |
| image input | ✅ (vision encoder in BF16; 448×448 probe answered correctly, 9.1 s) |

| metric (DeepSeek-V4-Flash-Vision-Exp, TP=2, DSpark k=5, eager) | value |
|---|---|
| decode | **25.4–27.3 / 23.1–25.7 tok/s** prose / JSON, counting 48–52 tok/s, acceptance 2.6–2.8 (110 ms step) — was 2.2 tok/s at the start of §35 |
| decode, spec off | 15.4 tok/s eager, 16.7 with PIECEWISE graphs |
| CUDA graphs | work, but with DSpark they are slower (20.8 / 21.6 tok/s), so DS4 runs eager (§39, §41) |
| prefill | ~460 tok/s at 8K, 408 at 32K, 378 at 64K, 331 at 128K (§42: indexer kernel, mHC tiles, 2048-token chunks) |
| time to first token | 0.4 s for a short prompt, 15 s at 8K, 80 s at 32K, 2.9 min at 64K, 6.6 min at 128K (§42); predicted ~17 min at 256K and ~46 min at 512K (§41 measured 32 / 97 min before §42) |
| context | 512K max (`ds4_max_ctx`); KV pool 901,584 tokens with 2048-token chunks (fp8_ds_mla in the 6 GiB pin, 1.72 × 512K); needles 3/3 at 32K–512K (§41, §42) |
| image input | ✅ OCR, colours/shapes, a table cell, a 1920×1080 image; 1.5–3.4 s per image request; counting slips (7 discs → 6) |
| thinking / tools | thinking mode ✅ (`chat_template_kwargs.thinking`); `deepseek_v4` tool + reasoning parsers |

The detailed history — every bug, fix, measurement and dead end — lives in
[PATCHES.md](PATCHES.md) (§1–§42). This README is the current state only.

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
(the 0.31 port: correct since §37, faster than the June stack since §41); set
`ds4_engine: delegate` for the June stack.

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

**DS4 native port** (`container/patches/ds4-native-fixed-refs/` plus the `vsh-ds4-*` patches):
serves correctly since §37. On top of the 09-30 gfx1151 port: `vsh-ds4-decode-inv-rope.py`
(the decode kernel's missing inverse RoPE, §37), `vsh-fp8-gemv.py` (HIP FP8 GEMV for the
block-FP8 linears at decode sizes, §36/§38), `vsh-fp8-prefill.py` (the same linears at
prefill sizes: exact bf16 weight + hipBLASLt, §41), `vsh-mxfp4-direct.py` + `vsh_moe_int4.hip`
(direct MXFP4 MoE, v5 kernel tuned on real routing, §38–§41), `vsh-ds4-mmf32.py` (compressor
scores on the bf16 GEMV, §40), `vsh-ds4-idx-logits.py` (prefill indexer on fp16 WMMA, §42),
`vsh-ds4-mhc-cfg.py` (gfx1151 mHC tiles, §42), `vsh-ds4-decode-rows.py` (bounded
decode-logits workspace for 2048-token chunks, §42), `cgfusion-soft.py` (§35); speed and
remaining work in Known issues 5.

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
  stack on both boxes: `vllm-strix-halo:glm-perf42-20261006` (through §42;
  `:local` points at it)
- **KV cache**: 16 GiB pinned, 256K max context (GLM); DS4: 6 GiB pinned = 901,584
  tokens of fp8_ds_mla with 2048-token prefill chunks, 512K max context
- **Speculative decoding**: DFlash2 drafter, k=3 (`glm53_spec_method: dflash`,
  `glm53_mtp_tokens: 3`); MTP (`glm5_next_mtp`) is the fallback and spec-off
  works too (§34)
- **Execution**: CUDA graphs, PIECEWISE breakable capture, sizes 1/2/4/8;
  `glm53_enforce_eager: 1` reverts to eager (§33). DS4 runs eager with DSpark k=5
  (`ds4_enforce_eager: 1`, `ds4_mtp_tokens: 5`; graphs are slower with DSpark, §41)
- **Quantization**: AWQ W4A16 for MoE experts (checkpoint). The checkpoint's
  BF16 linears (KDA/MLA projections, shared experts, dense MLPs) are
  re-quantized to int8 group-128 at load (§28); `lm_head` gets an int8 copy
  (§30); the router gate, `kv_b_proj`, `wk_weights_proj`, embeddings and the
  vision encoder stay BF16. DS4 runs its checkpoint as shipped: MXFP4 routed
  experts and block-FP8 linears (decoded by the HIP kernels of §36–§41), BF16
  router gate, compressor and vision encoder

## Measured performance (GLM 2026-10-03, DS4 2026-10-06)

See the table at the top. Receipts and methods:
`scripts/vsh_ab.py` (matched A/B with engine-side step accounting + tool-call
gate), `scripts/vsh_bench.py` (per-request decode/TTFT with spec and
prefix-cache deltas), `scripts/stepbench.py` (unprofiled decode step time),
`scripts/trace_*.py` (kernel share, GPU duty cycle, copies, blocking ops),
`scripts/bf16_bw.py` (linear bandwidth at real weight shapes). Numbers are
medians of ≥2 boots or ≥5 repetitions. DS4: `scripts/ds4len.py` (battery),
`scripts/ds4speed.py` (decode), `scripts/ds4ctx.py` (long-context needles + TTFT),
`scripts/ds4image.py` (image probes), `scripts/ds4trace.py` / `ds4who.py` / `ds4prefill.py`
(torch-profiler traces: step wall vs GPU busy, top kernels, kernel → launching op),
`scripts/moe_replay.py` (MoE kernels on recorded DS4 routing).

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
├── scripts/                 # harnesses + unit tests (vsh_ab, stepbench, nll, ds4*, moe_*, test_*, trace_*)
├── PATCHES.md               # detailed patch history and lessons (§1–§42)
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
5. **DS4 native** — correct since §37; 25.4–27.3 / 23.1–25.7 tok/s prose / JSON with
   DSpark k=5 (eager, §41), 15.4 tok/s without spec. Needles, counting, thinking mode
   and image input pass; 512K context fits. Prefill ~330–460 tok/s up to 128K (§42);
   it still slows with context because the indexer scores every earlier compressed
   position (now on fp16 WMMA, ~9x aiter's kernel). CUDA graphs work but are slower
   with DSpark (+8 % without spec), so DS4 stays eager. Remaining levers: a fused
   indexer logits + top-k kernel (the [rows x context/4] logits matrix is the rest of
   the quadratic term), the prefill MoE, the all-reduce over Thunderbolt; decode — the
   MoE (38 % of GPU time), the bf16 `wo_a` einsum (10.7 ms/step). DS4 boots need
   `VSH_W8A16=0 VSH_W8A16_LMHEAD=0` on both ranks and `ds4_max_seqs` for the 2048-token
   chunks — the DS4 restart/reserve scripts pass them, and every restart script refuses
   to start when the two boxes' env files differ.
