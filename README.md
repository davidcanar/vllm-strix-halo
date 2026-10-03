# vllm-strix-halo — GLM-5.3-Flash (and DeepSeek-V4-Flash) on 2× AMD Strix Halo, TP=2 over Thunderbolt

> **STATUS 2026-10-03 — working end to end.** GLM-5.3-Flash serves correctly:
> long-context retrieval passes at every tested length (mid-context and tail,
> 6.8K–255K, including adversarial), tool calling works (8/8 on the captured
> opencode request), prefix caching works, and decode is **~22 tok/s** after
> the §28 decode round. DS4 (DeepSeek-V4-Flash) runs on the June 2026 stack
> (`delegate` mode, 19–24 tok/s); the native 0.31 port is still open.

| metric (GLM-5.3-Flash AWQ W4A16, TP=2, MTP k=3) | value |
|---|---|
| decode | **~22 tok/s** (stepbench; engine gauge 22–26) — was 12.0 before §28 |
| prefill | ~283 tok/s at 9.6K; 250–270 tok/s cold at 4K–13.9K |
| TTFT, 13.9K context | 52 s cold → **0.95 s cached** |
| long-context retrieval | ✅ 3/3 mid-context at 6.8K/13K/32K; tail needles PASS to 255K; adversarial PASS |
| tool calling | ✅ 8/8 on the captured opencode request |

The detailed history — every bug, fix, measurement and dead end — lives in
[PATCHES.md](PATCHES.md) (§1–§28). This README is the current state only.

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
~/vllm-strix-halo.sh ds4 start       # DS4 via the June stack (delegate)
~/vllm-strix-halo.sh glm53 stop      # ... etc
```

Both models share the `vllm-glm` container and API port 1234 — either/or at
runtime. Site config: `~/vsh-config.yaml` (flat keys → `VSH_*` env vars).

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
| `vsh-apc-retention.py` | `--prefix-cache-retention-interval 2304` (stock flag): keeps KDA/Mamba state checkpoints so the *first* repeat hits. Env `VSH_GLM53_APC_RETENTION` (on). §20 |
| `vsh-adaptive-k.py` | EMA policy for the verified draft length, wired for async scheduling + live override. Installed, instrumented, **off** — step time is draft-length independent here. §17 |
| `vsh-triton-ptr-cache.py` | Memoises Triton's per-pointer `hipPointerGetAttribute`. Validated bit-exact, **off** — worth ~0.3 % of a step. §21 |
| `host/moe-configs/E=288,N=1024,...json` | Tuned MoE tiles under the filename the runtime actually asks for |

**Perf (§28, 2026-10-03):** GPU clock cap (power budget), bf16 MLA BMM,
packed-key router, split-KV attention, and the **HIP int4 MoE decode GEMV**
(§28.5: routed MoE 90 → 37 ms/step; gate `VSH_MOE_INT4_HIP`).

**Still carried from earlier rounds:** #55222 workspace-units ports (indexer +
GLM attention), #57979 KDA stride fix, upstream ragged-index rewrite, the
gfx1151 platform gates (AITER unlock, fp8 fnuz, pre-shuffle skips, mHC,
deterministic top-k, WNA16 OOM), and the OdinLink hooks (`odl_ar2`, `odl_mq`).
The older gather-site block-table expansion (`vsh-idx-bt-gather-v4-ops.py`)
and column-overwrite force-tail are superseded by §27 and are inert.

**DS4 native port** (`container/patches/ds4-native-fixed-refs/`): boots and
serves; decode quality/speed unresolved (§ Known issues 7).

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
  the byte-exact patched files for reproducible rebuilds)
- **KV cache**: 16 GiB pinned, 256K max context
- **Speculative decoding**: MTP k=3. DFlash2 k=7 is one config line away
  (`glm53_spec_method: dflash`) but is a wash on prose and slower on JSON in
  a matched A/B (§25–§26)
- **Quantization**: AWQ W4A16 for MoE experts; bf16 for KDA projections, MLA,
  and lm_head (checkpoint `ignore` list)

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
├── vsh-config.yaml          # site config template
├── host/                    # host-side scripts (restart, serve, heal, moe-configs)
├── container/
│   ├── Dockerfile           # image build (base + patches)
│   ├── patches/             # reference copies of patched files (+ debug/, ds4 refs)
│   ├── pinned-vllm/         # byte-exact patched vllm files (rebuild input)
│   └── pinned-triton/       # pinned Triton AMD driver
├── scripts/                 # measurement harnesses (vsh_ab, vsh_bench, stepbench, trace_*)
├── PATCHES.md               # detailed patch history and lessons (§1–§28)
└── README.md                # this file
```

## Known issues and next steps

1. **Decode speed** — 102–107 ms/step, 21.5 / 25.9 / 17.9 tok/s
   (JSON / tools / prose, DFlash2 k=3) after §28–§31; 12 tok/s at the start.
   Done: GPU clock cap, HIP int4 MoE, int8 BF16 linears (W8A16), int8
   `lm_head`, split-KV sparse attention, fused router, direct decode MoE.
   The step is ~4–9 ms above GPU busy (~98 ms), so GPU work counts again.
   Remaining: router gate (BF16 weight / fp32 out, ~6 ms), mHC wrapper and
   model glue on the host, CUDA graphs (2). See
   [FRESH-EYES-20tps.md](FRESH-EYES-20tps.md).
2. **CUDA graphs** — blockers understood, not shipped: `odl_ar2` is
   graph-unsafe by design (host-side counter; needs a device-side counter),
   the KDA chunk-index sync is prefill-only (`FULL_DECODE_ONLY` keeps it
   eager), drafter metadata rebuild can stay eager. Graphs alone are worth
   ~0–10 %; their value is making every later GPU fix land 1:1.
3. **MTP-off crashes rank 1** on first generation (§26) — likely the §27
   page-aliasing bug at the changed block geometry; retest now that #59412 is
   ported. Gates the MTP-off graph bench.
4. **Prefill MoE is untuned** — 41.5 % of prefill GPU time runs at 12–18 % of
   achievable FLOPS (tuned tiles cover M≤512; prefill chunks run M~65k).
5. **Debug leftovers** — the long-context hunt leaves six ungated
   `/tmp/glm_*.log` writers (with host syncs) on the prefill path. Remove.
6. **Upstream** — drop `59412-*` (and the inert gather-site expansion) at the
   next rebase; worth reporting: gfx1151's `rocm_fp8_paged_mqa_logits`
   stage1 fallback scores paged SHUFFLE caches at random (our
   `vsh-kpool-paged-logits` replaces it).
7. **DS4 native decode** — repetition loops, 2–6 t/s. New lead: DS4's ratio-1/2
   indexer caches hit the same broken `stage1` fallback §27 fixed; the new
   reader covers that path — retest DS4 native before further bisection.
8. **Image snapshots predate §27/§28** — commit fresh `podman commit`
   snapshots on both boxes; the running containers and `pinned-vllm/` are
   current.
