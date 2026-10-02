# vllm-strix-halo — GLM-5.3-Flash (and DeepSeek-V4-Flash) on 2× AMD Strix Halo, TP=2 over Thunderbolt

> # STATUS 2026-10-01: GLM long-context FIXED at all tested lengths (up to 105K, including adversarial); DS4 native port incomplete
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
> | adversarial 54K (1000× decoy) | FAIL | **PASS** |
>
> **Measured speeds (2026-10-01, GLM-5.3-Flash AWQ W4A16, TP=2, MTP k=3):**
> prefill 400–2100 tok/s · decode 10.8–15.0 tok/s · spec acceptance length 2.7–3.0
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

| patch | what it fixes |
|---|---|
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
- **Speculative decoding**: MTP k=3 (`glm53_mtp_tokens`), validated ~1.9×
- **Quantization**: AWQ W4A16 (compressed-tensors) for MoE experts; bf16 for
  KDA projections, MLA, and lm_head (checkpoint `ignore` list)

## Measured performance (2026-10-01)

| metric | GLM-5.3-Flash | DS4 (June stack) | DS4 (native 0.31) |
|---|---|---|---|
| prefill | 400–2100 tok/s | ~112–136 tok/s | ~213 tok/s |
| decode | 10.8–15.0 tok/s | 19–24 tok/s | 2–6 tok/s ⚠️ |
| spec acceptance | 2.7–3.0 | — | — |
| long-context retrieval | ✅ to 105K+ | ✅ | ⚠️ untested |

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
├── container/
│   ├── Dockerfile            # image build (base + patches)
│   └── patches/              # all reference copies of patched files
│       ├── debug/            # instrumentation (inert, for future debugging)
│       └── ds4-native-fixed-refs/  # DS4 native port curated refs
├── PATCHES.md                # detailed patch history and lessons
└── README.md                 # this file
```

## Known issues and next steps

1. **DS4 native decode** — the open bug. Output degenerates into repetition
   loops; decode at 2–6 t/s. Narrowed by elimination to the target model's
   decode-side forward path. The June delegate engine works correctly.
2. **GLM CUDA graphs** — parked since 2026-09-28 (first replay wedges the MTP
   drafter on hybrid attention backends). Typically 1.5–2× decode if fixed.
3. **Upstream reporting** — the block-table granularity bug and the
   force-tail approach are worth reporting upstream; they affect every ROCm
   user of the GLM kpool indexer.
4. **Image rebuild** — the container currently carries edits in its overlay;
   a `podman commit` or Dockerfile rebuild would bake them in properly.
