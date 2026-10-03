# vllm-strix-halo — GLM-5.3-Flash (and DeepSeek-V4-Flash) on 2× AMD Strix Halo, TP=2 over Thunderbolt

> # STATUS 2026-10-03: GLM long-context sparse attention fixed end to end (§27); tool calling works; decode speed is the open item
>
> **The GLM-5.3 long-context retrieval bug is fixed.** Four stacked fixes
> (one upstream PR port + three local) took tail-retrieval from failing at
> ~5K tokens to passing deterministically at every tested length up to 105K,
> including the adversarial case (needle buried in 1000× repeated decoy):
>
> | context | before | after |
> |---|---|---|
> | ≤ 10K | PASS | PASS |
> | 12K | FAIL | PASS |
> | 54K | FAIL | **PASS** (deterministic) |
> | 75K | FAIL | **PASS** |
> | 95–105K | FAIL | **PASS** |
> | 146K | — | **PASS** |
> | 204K | — | **PASS** |
> | 255K (max) | — | **PASS** |
> | adversarial 54K (1000× decoy) | FAIL | **PASS** |
>
> **Prefix caching was losing a scheduler page on every hit** — the EAGLE
> last-block drop was being applied to *all five* KV groups because the GLM
> grouping path returns before the annotator runs. Fixed (§18), plus the
> Mamba/KDA retention interval (§20): an identical 13.9K-token follow-up now
> costs **0.95 s TTFT instead of 52 s** cold, and a prompt that used to get
> *nothing* gets 2304 of its 2362 tokens back.
>
> **Measured speeds (2026-10-02, GLM-5.3-Flash AWQ W4A16, TP=2, MTP k=3):**
> prefill **250–270 tok/s** cold (4000: 268, 13.9K: 261) · decode **9.1 tok/s
> prose / 11.6 tok/s JSON** (2.0–2.6 accepted tokens per ~210–224 ms step) ·
> TTFT on a cached 13.9K context **0.95 s**
>
> **§27 (2026-10-02/03): four stacked sparse-attention bugs fixed.** Past
> ~2K tokens the decode indexer scored at random, the index cache aliased every
> pool past ~4K tokens onto one page, an fp8 cast produced NaN and the
> force-tail patch evicted selected pools. The tail-needle table above passed
> only because force-tail kept the tail visible; **mid-context** retrieval was
> 0/3 at 13K and is now 3/3 at 6.8K/13K/32K, and opencode's real tool-call
> request went from ~30-40 % to **8/8** usable (§27.1).
>
> **Decode speed is not measured to its limits** (the earlier verdict rested on
> halved per-step profile numbers and a GPU-bound misdiagnosis -- see Known
> issues #1). Current: ~217-226 ms/step at 2-4K, JSON 11.9 / tool calls 13.1 /
> prose 8.7 tok/s.
>
> **DS4 (DeepSeek-V4-Flash) native port is incomplete**: prefill works at
> ~213 tok/s but decode runs at 2–6 tok/s with degraded output quality. The
> June 2026 stack (`delegate` mode) remains the working DS4 engine at 19–24 t/s.
> See [PATCHES.md](PATCHES.md) §15.9–15.12 for the full DS4 bug-hunt history.

Serve **GLM-5.3-Flash** (and **DeepSeek-V4-Flash**) with vLLM across **two AMD
Strix Halo (gfx1151) boxes** — one 8060S iGPU per box, tensor-parallel (TP=2),
with the inter-GPU fabric carried over a **Thunderbolt-4/USB4** cable between
them via the **OdinLink driver** (`odl_tb5`): RCCL rides the OdinLink net
plugin for the big prefill-sized collectives, the `odl_ar2` native carries the
small decode all-reduces over odl streams, and `odl_mq` (fail-open) moves the
EngineCore↔worker control plane off the per-step TCP round trips. The default
is `transport: odl`; `transport: tcp` (plain sockets, no kernel modules)
remains the fallback.

GLM-5.3-Flash support is **upstream vLLM** since 2026-09-03
([#53906](https://github.com/vllm-project/vllm/pull/53906)), and the
DeepSeek-V4-Flash model code is upstream since the 2026-09-27 wave. This repo
pins vLLM at `73859fec` (2026-09-27/28 main) on a
[kyuz0 vllm-therock-gfx1151](https://github.com/kyuz0) ROCm-10.0/torch-2.11
base, adds the OdinLink/AITER plumbing and the gfx1151-specific fixes below.

---

## Quick start

```bash
# on box1 (the ray head):
~/vllm-strix-halo.sh glm53 start          # GLM-5.3-Flash on :1234
~/vllm-strix-halo.sh ds4 start            # DS4 via the June stack (delegate)
# stop:
~/vllm-strix-halo.sh glm53 stop
~/vllm-strix-halo.sh ds4 stop
```

Both models use the same `vllm-glm` container/image (`vllm-strix-halo:local`)
and the same API port (1234) — they are either/or at runtime. Site
configuration lives in `~/vsh-config.yaml` (flat keys → `VSH_*` env vars).

## Current patch set

### GLM-5.3 long-context sparse attention fixes (2026-10-02, late) — PATCHES.md §27

Mid-context retrieval (3 needles at 25/50/75 %) went from **0/3 to 3/3** at 13K
and 32K, and the captured opencode tool-call request from ~2/6 to **6/6**.

| patch | file | what it fixes |
|---|---|---|
| `59412-pooled-indexer-kernel-blocks.py` | `v1/attention/backends/mla/indexer.py`, `v1/worker/utils.py` | Port of upstream #59412: page-aligned kernel blocks for the kpool indexer. Without it every pool past ~4K tokens was written to and read from page 0 |
| `vsh-kpool-paged-logits.py` + `vsh_kpool_paged_logits.py` | `v1/attention/ops/rocm_aiter_mla_sparse.py` | Correct decode indexer reader for the paged SHUFFLE cache on gfx1151 (aiter's stage1 fallback scored at random); also drops the 12.6 ms/step whole-pool fp8 conversion. Env `VSH_KPOOL_PAGED_LOGITS` |
| `vsh-idx-force-tail-boost.py` | `models/glm5next/amd/sparse_indexer.py` | Force-tail now boosts recent pools before top-k instead of overwriting (unsorted) top-k columns, which evicted ~1/4 of the selected pools |
| `vsh-idx-fp8max-fnuz.py` | `models/glm5next/amd/ops/kpool_compress.py` | Indexer K quantized to 224 so the e4m3fn→fnuz cast cannot overflow to NaN. Env `VSH_IDX_FP8MAX_FNUZ` |

### GLM-5.3 decode + cache fixes (2026-10-02)

| patch | file | what it does |
|---|---|---|
| `vsh-glm53-apc-align.py` | `v1/core/kv_cache_coordinator.py` | The EAGLE last-block drop is resolved to *pure drafter* groups instead of upstream's "flag every group" fallback. On this model that is the empty set (the MTP layer shares group 0 with the target MLA layers), so MLA/KDA hits stop losing one scheduler page (2304 tokens) per lookup. Env `VSH_GLM53_APC_ALIGN`. §18 |
| `vsh-apc-retention.py` | `host/vsh-manual-serve.sh` | Adds `--prefix-cache-retention-interval` (stock EngineArgs flag, `VSH_GLM53_APC_RETENTION=2304`). The stock default of 0 keeps only the latest replay boundary for the Mamba/KDA groups, which made the *first* identical repeat miss. §20 |
| `vsh-adaptive-k.py` | `v1/core/sched/scheduler.py` | EMA policy for the verified draft-prefix length, wired for **async** scheduling (the v1 port hooked a function this build never calls) plus per-step telemetry and a live JSON override for one-boot k-sweeps. Installed, instrumented, **off** — the step time is draft-length independent and acceptance saturates near 2 tokens/step. §17 |
| `vsh-triton-ptr-cache.py` | `triton/backends/amd/driver.c` | Memoises Triton's per-pointer `hipPointerGetAttribute`. Applied and validated bit-exact, **left off**: the call costs 0.5 µs, ~0.3 % of a step. §21 |
| `vsh-sync-instr.py` | `third_party/.../ops/index.py` | Opt-in timer for the KDA chunk-index host sync. Off; that sync is on the chunked path only and the GPU is 99 % busy while it blocks. §22 |
| `host/moe-configs/E=288,N=1024,device_name=AMD_Radeon_8060S.json` | tuned MoE tiles | The runtime asks for this name (no `dtype=` suffix) since the Sep-27 pin; the file deployed in September carried the suffix, so the tuning had been silently unused. Supplying it changes nothing measurable — the September sweep is stale, not lost. §23 |

### GLM-5.3 long-context fixes (2026-10-01)

| patch | file | what it fixes |
|---|---|---|
| `55222-indexer.py` | `vllm/v1/attention/backends/mla/indexer.py` | Prefill workspace sized in uncompressed tokens instead of compressed rows → overflow on long contexts (upstream #55222, ported by hand; no official image has it yet) |
| `55222-glm5next-attention.py` | `vllm/models/glm5next/common/attention.py` | Same units bug in the GLM kpool indexer path (also #55222) |
| `vsh-idx-bt-gather-v4-ops.py` | `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py` | **Block-table granularity mismatch**: the indexer K-gather received the MAIN block table (114 cols at main-cache granularity) but needed 216 cols at the storage-block granularity (64 pools/block) → a 29K-token dead zone where every pool scored exactly 0. Fix: expand the table at the gather call site (`repeat_interleave(K) * K + j`) |
| `vsh-idx-force-tail-sparse_indexer.py` | `vllm/models/glm5next/amd/sparse_indexer.py` | **Force-tail**: the sparse top-k selects only 512 pools (2048 tokens) per query; the needle's recent pools sometimes lost the ranking competition. Fix: `_force_recent_tail_pools()` overwrites the lowest-ranked 128 columns with the 128 most-recent pools (512 tokens), guaranteeing the context tail is always attendable |

### GLM-5.3 perf patches (2026-10-01)

| patch | file | what it does |
|---|---|---|
| `57979-kda-fused_recurrent.py` + `57979-kda-kernels.py` | `glm5next/amd/ops/third_party/kda/` | Upstream #57979 (ROCm stride-aware decode KDA): removes per-step `.contiguous()` copies of q/k/v/beta in all 34 KDA layers — small/noise-level win on this rig |
| `upstream-window-ragged-compact-ops.py` | `rocm_aiter_mla_sparse.py` | Upstream rewrite of the ragged-index kernels (drops −1 holes instead of prefix-copying them) — correctness hygiene |

### gfx1151 platform fixes (2026-09-27)

| patch | file | what it fixes |
|---|---|---|
| `vsh-aiter-gfx1151-gate.patch` | Unlocks the AITER sparse-indexer/MLA ops on gfx1151 (CDNA-only gate hides them from Strix Halo) |
| `vsh-fp8-fnuz-mqa.patch` | NVIDIA-style fp8 (e4nv) → ROCm-native fp8e4m3fnuz conversion at the MQA-logits seam |
| `vsh-ds4-no-aiter-fp8-gfx1151.patch` | AITER fp8 block-scaled MM is MI300-only; gate it off so the stock Triton path works |
| `vsh-ds4-no-attn-preshuffle-gfx1151.patch` | Skip the attention pre-shuffle on gfx1151 |
| `vsh-ds4-no-gateup-preshuffle-gfx1151.patch` | Skip the gate/up-proj pre-shuffle on gfx1151 |
| `vsh-mhc-no-tilelang-gfx1151.patch` | mHC: use the stock aiter kernel, not the TileLang variant |
| `vsh-moe-router-deterministic-topk.patch` | Deterministic MoE top-k for reproducible greedy |
| `vsh-wna16-layerwise-empty-cache.patch` | WNA16 conversion GTT reservation OOM at init |
| `vsh-odl-ar2-allreduce.patch` | OdinLink decode all-reduce (odl_ar2) |
| `vsh-odl-mq-dataplane.patch` | OdinLink control-plane message queue (odl_mq) |

### DS4 (DeepSeek-V4-Flash) native port patches (incomplete — see status)

The `ds4-native-fixed-refs/` directory holds the curated reference copies
for the DS4 native vLLM 0.31 port. The port boots and serves, but decode
quality/speed are unresolved. See [PATCHES.md](PATCHES.md) for the full
history.

---

## Architecture

```
┌──────────────────────────────┐       ┌──────────────────────────────┐
│ box1 (192.168.0.102)         │       │ box2 (10.0.2.2)              │
│  ┌────────────────────────┐  │  TB4  │  ┌────────────────────────┐  │
│  │ vllm-glm container     │  │◄─────►│  │ vllm-glm container     │  │
│  │  vllm 0.31.0-dev       │  │odl_tb5│  │  (ray worker)           │  │
│  │  GLM-5.3-Flash AWQ     │  │       │  │  GPU 1/2               │  │
│  │  GPU 0/2               │  │       │  │                        │  │
│  │  ray head :6379        │  │       │  │                        │  │
│  │  API :1234             │  │       │  │                        │  │
│  └────────────────────────┘  │       │  └────────────────────────┘  │
└──────────────────────────────┘       └──────────────────────────────┘
```

- **Image**: `vllm-strix-halo:local` (built from `container/Dockerfile`;
  base `kyuz0/vllm-therock-gfx1151:rocm10.0.0-torch2.11.0-vllm0.30.0`;
  vLLM pinned at `73859fec`)
- **Container**: `vllm-glm` (podman, shared by both models)
- **KV cache**: 16 GiB pinned (`glm53_kv_bytes`), 256K max context
- **Speculative decoding**: MTP k=3 (`glm53_mtp_tokens`), validated ~1.9×.
  DFlash2 k=7 is one config line away (`glm53_spec_method: dflash`) but is a
  **wash on prose and slower on JSON** in a matched A/B; it only leads on
  tool-call output (§25)
- **Quantization**: AWQ W4A16 (compressed-tensors) for MoE experts; bf16 for
  KDA projections, MLA, and lm_head (checkpoint `ignore` list)

## Measured performance (2026-10-02)

| metric | GLM-5.3-Flash | DS4 (June stack) | DS4 (native 0.31) |
|---|---|---|---|
| prefill | 250–270 tok/s cold | ~112–136 tok/s | ~213 tok/s |
| decode | 9.1 (prose) / 11.6 (JSON) tok/s | 19–24 tok/s | 2–6 tok/s ⚠️ |
| spec acceptance | 2.0–2.6 tokens/step at k=3 | — | — |
| TTFT, 13.9K context | 52 s cold → **0.95 s cached** | — | — |
| long-context retrieval | ✅ to 255K | ✅ | ⚠️ untested |

Measurement method and receipts: `scripts/vsh_ab.py` (matched A/B with
engine-side step accounting and a tool-call gate), `scripts/vsh_bench.py`
(per-request decode/TTFT with spec-decode and prefix-cache deltas),
`scripts/trace_share.py` / `trace_gap_analyze.py` / `trace_dtoh.py` /
`trace_sync.py` (kernel-class share, GPU duty cycle, copy inventory, blocking
ops) and `scripts/bf16_bw.py` (linear bandwidth at the real weight shapes).
All numbers above are medians of ≥2 boots or ≥5 repetitions.

⚠️ DS4 native decode is the open issue — output degenerates into repetition
loops. All A/B eliminations (drafter patches, spec on/off, C4A gate, ragged
kernels) point to the target model's decode-side forward path, not the
drafter or the spec machinery. The June stack (`ds4_engine: delegate`)
remains the production DS4 engine.

## Repository layout

```
vllm-strix-halo/
├── vllm-strix-halo.sh       # supervisor launcher (glm53 | ds4)
├── vsh-config.yaml           # site config template
├── host/                     # host-side scripts (restart, serve, heal)
│   └── moe-configs/          # tuned MoE tiles (both accepted filenames)
├── container/
│   ├── Dockerfile            # image build (base + patches)
│   ├── patches/              # all reference copies of patched files
│   │   ├── debug/            # instrumentation (inert, for future debugging)
│   │   └── ds4-native-fixed-refs/  # DS4 native port curated refs
│   └── pinned-vllm/          # byte-exact patched vllm files (rebuild input)
├── scripts/                  # measurement harnesses (vsh_ab, vsh_bench, trace_*)
├── PATCHES.md                # detailed patch history and lessons (§1–§25)
└── README.md                 # this file
```

## Known issues and next steps

1. **Decode speed (the main open item).** Measured 2026-10-03 after §27:
   ~186 ms/step at short context, ~217-226 ms/step at 2-4K (JSON 11.9,
   tool calls 13.1, prose 8.7 tok/s). The earlier "~+6 % and no more" verdict
   (§21-§23) rested on two measurement errors, corrected in
   [FRESH-EYES-20tps.md](FRESH-EYES-20tps.md): the profiler annotates each step on two GPU
   streams, so every per-step kernel figure was halved (the int4 MoE is
   ~76 ms/step at ~90 GB/s, not 42.7 ms at 87 %; BF16 traffic is ~10.9 GB/
   rank/step, not 6.47), and steady-state decode is CPU-bound or balanced, not
   GPU-bound (kernels start ~0.3-0.5 ms after launch). CPU ~= GPU ~= 190-220 ms
   per step, which is why single-sided changes keep measuring as noise -- e.g.
   §27 removed 12.6 ms/step of GPU work and the step did not move. The plan:
   CUDA graphs (item 2) to remove the CPU floor, then the GPU waste list
   (fp8 BMM on RDNA3.5, router sort kernel, 8-workgroup decode attention,
   4x lm_head reads, BF16 MTP experts), then an int4 MoE decode kernel and
   8-bit BF16 weights. Projected ~95-110 ms/step if most of it lands.
2. **CUDA graphs** -- not shipped; the blockers are understood. MTP-off graphs
   work (§11 retry, 114 ms/token). With MTP: (a) `odl_ar2` is graph-unsafe by
   design (host-side round counter baked into captured kernel args;
   `eligible()` returns False while capturing, so graphed all-reduces fall back
   to RCCL while eager segments use odl_ar2), fix = device-side counter;
   (b) the KDA chunk-index host sync is on prefill-bearing steps only (§22) --
   `FULL_DECODE_ONLY` keeps them eager; (c) the drafter's between-step
   metadata rebuild can stay eager (its CPU cost hides under the graphed
   target). Graphs alone are worth ~0-10 %; their value is that every GPU fix
   afterwards lands 1:1.
3. **MTP-off crashes rank 1** with an illegal memory access on the first real
   generation (§26). Likely the §27 page-aliasing bug (PR #59412 notes the
   same aliasing writes past the row buffer at other block geometries, and
   MTP-off changes the scheduler block 2304 -> 2176). Retest now that #59412
   is ported; it gates the MTP-off graph bench in item 2.
4. **Prefill is the remaining inefficiency** (moves TTFT, not t/s): 41.5 % of
   prefill GPU time is the int4 MoE at ~12-18 % of achievable FLOPS; the tuned
   tile config covers M<=512 while a prefill chunk runs at M~65k rows. Never
   tuned. Also re-measure prefill at matched sizes after §27 (§27.1).
5. **Debug leftovers on the prefill path**: ungated logging from the long-context
   hunt writes six `/tmp/glm_*.log` files (scores, topk, gather, kpool, map,
   idxplan) with host syncs every chunk; they grow without bound. Remove.
6. **Upstream tracking.** We carry a port of PR #59412 (#58858, page-aligned
   kernel blocks for pooled indexers) -- drop `59412-*` when rebasing onto a pin
   that has it; `vsh-idx-bt-gather-v4-ops.py`'s gather-site expansion is inert
   with #59412 and can go too. Not yet reported upstream: on gfx1151,
   `rocm_fp8_paged_mqa_logits` falls through to aiter's block_size==1
   `stage1` reader for paged SHUFFLE caches (random top-k; our
   `vsh-kpool-paged-logits` replaces it) -- worth an issue/PR. #59741 (tail
   gap) mitigation is now `vsh-idx-force-tail-boost` (logit boost before top-k;
   the old column overwrite evicted selected pools).
7. **DS4 native decode** -- output degenerates into repetition loops, 2-6 t/s.
   New lead: DS4's ratio-1/2 indexer caches are also paged SHUFFLE and go
   through the same `rocm_fp8_paged_mqa_logits` stage1 fallback that §27 found
   scoring at random on gfx1151 -- the new reader covers that path; retest DS4
   native before further bisection. The June delegate engine still works.
8. **Image snapshots predate §27.** `vllm-strix-halo:glm-longctx-fixed-20261002`
   does not carry the four §27 fixes; the running containers and
   `container/pinned-vllm/` (commit 0c5fbfb) do. Commit a new snapshot on both
   boxes. The launcher retention flag and `container/pinned-triton/` notes from
   the previous rebuild still apply.

