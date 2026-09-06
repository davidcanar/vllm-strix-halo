# RESULT — layer-3 tap: the nondeterminism source is the MoE router's top-k

**Date:** 2026-09-05 · **Task:** HANDOFF-layer3-tap.md · **Status:** source
isolated, root-caused to a named op, control arm run. One secondary finding
(rare deviations that survive the fix) is characterized but **not** fully
root-caused — see §5.

---

## 1. Reproduction of the layer-3 localisation: YES

`python3 /tmp/bisect_run.py` (unchanged, on the previous session's per-layer
taps), 3 identical 740-token prefill-only requests, prefix caching off,
sequential, idle server. Result matches the handoff table exactly, on both
ranks:

| layer | hidden | residual | post | comb |
|---|---|---|---|---|
| 0,1,2 | = | = | = | = |
| **3** | **X** | = | = | = |
| 4+ | X | X | X | X |

Input guard: `embed=9b0d90d30511` in all forwards (same hash as the handoff
records), frontend `return_token_ids` sequences identical 3/3 (`id0=2833`).
`input_ids` is `None` in this code path as documented; the embedding-hash
guard is the input check. Note the tap headers were written with
`input_ids=None` so the analyser's "tap id0 matches: False" line is the
expected `na` case, not an input mismatch.

## 2. Six-point tap inside layer 3

Patch: 6 hash points in `Glm5NextDecoderLayer.forward` (mHC branch), gated on
`self.layer_idx == 3` + prefill (ntok>1) + `VSH_TAP_FILE`. Hash-only taps —
no slicing, env-gated, unset ⇒ stock. Applied to **both** containers.

8 identical prefill-only requests (`temperature=0`, `max_tokens=1`,
740 tokens, prefix caching off, 2 warmups first, taps truncated before the
measured reps). Hashes were **identical on rank0 and rank1 at every point**
(including the diverging values — the ranks are in lockstep; this is not a
rank-desync).

| point | what it hashes | result over 8 forwards |
|---|---|---|
| pt=1 | layer entry (`hidden_states` as received) | **=** (all `8a13c47033ec`) |
| pt=2 | after pre-attention `hc_fused_post_pre` (input to self_attn) | **=** (`b811dd69a0b5`) |
| pt=3 | **after `self.self_attn(...)`** — DSA attention out | **=** (`34a280916a46`) |
| pt=4 | after post-attention `hc_fused_post_pre` (MoE input) | **=** (`a143204b191f`) |
| pt=5 | **after `self.mlp(...)`** — MoE out | **X — 8 forwards, 8 distinct hashes** |
| pt=6 | at return (x + r/p/c) | x mirrors pt=5 (sanity anchor holds); r/p/c all **=** |

**First differing point: pt=5 — the MoE.** The DSA sparse attention is clean
in situ (pt=3 identical every call), both mHC fused post/pre ops are clean
(pt=2, pt=4), the mHC state r/p/c is clean — exactly as the earlier table
implied.

Tap-consistency checks that passed: pt=1 == model-level h after layer 2;
pt=5 == pt=6 x; pt=4 == the MoE-runner input hash (round 2); pt=5 == runner
output (round 3). Cross-rank: 0 mismatches at any point.

## 3. Going deeper: routing, not the expert GEMM

Round 2 — tap `Glm5NextMoE.forward` (layer 3 only): MoE input `x` identical,
**`router_logits` identical** (`bc3d628ff22b`, fp32, 8/8), final MoE output
8/8 distinct ⇒ the split is between the gate and the output.

Round 3 — tap inside `MoERunner` (`fused_moe/runner/moe_runner.py`,
layer 3 only): with `moe2=pre` x and `rl` identical every call,
**`select_experts` returns different `topk_ids`/`topk_weights` every call**
(8/8 distinct, mono=False — the modular path), and `moe2=post` deviates in
lockstep. The routing **selection** is nondeterministic; the expert GEMMs
merely consume differing routing.

Root cause, named: `grouped_topk_router.py::grouped_topk` selects with

```python
use_sorted = envs.VLLM_BATCH_INVARIANT          # default False
group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=use_sorted)[1]
...
topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=use_sorted)[1]
```

i.e. `torch.topk(..., sorted=False)`. (The fused `ops.grouped_topk` branch is
**not** taken on this platform: `RocmPlatform.is_cuda()` is False, and
`VLLM_ROCM_USE_AITER_MOE=0` also disables the aiter grouped-topk.) Note the
comment two lines above the flag: *"use sorted=True to ensure deterministic
expert selection"*.

**Op-level standalone proof** (in the container, M=740 / E=288 / groups=1,1 /
k=8 / sigmoid+bias, 20 identical calls, gfx1151):

| arm | distinct raw outputs /20 | distinct selected SETS /20 |
|---|---|---|
| in-situ `grouped_topk` (torch.compile'd, as deployed) | **20/20** | 20/20 |
| raw `torch.topk(..., sorted=False)` | **20/20** | 20/20 |
| raw `torch.topk(..., sorted=True)` | 1/20 | 1/20 |

`sorted=False` is run-to-run nondeterministic on bit-identical input —
including *which* experts get picked, not just their order. This also
explains the earlier exoneration gap: the standalone fused-MoE test
(`moe_prefill_det.py`) fed **fixed** `topk_ids/weights` into
`fused_experts_op`, so the selection layer was never exercised standalone.

**Control arm (in situ, sorted=True forced via a one-line patch):** layer 3
became bit-deterministic — pt=1..6, `router_logits`, `topk tw/ti` all
identical across 8 forwards. This confirms the mechanism end-to-end: layer 3
diverged **only** because of the unsorted top-k.

## 4. What the fix does not cure (measured, 24 forwards, sorted=True)

With layer 3 fixed, per-forward divergence does not vanish; it now **first
appears at layer 21 (or 34)** — both KDA+MoE layers — with r/p/c still
identical at the origin layer, and the first sampled token still flips
(8/24 'Then' vs 16/24 'Which'). Two residual effects, both measured with
all-layer MoE taps:

1. **Exact-tie top-k flips survive `sorted=True`.** At layer 21 the router
   logits were identical in all 24 forwards, yet `topk tw/ti` took a second
   value in 8/24 forwards (layer 34: own-value flips in ~6/24). Standalone
   demo: with exact fp32 ties at the k-boundary, `torch.topk(sorted=True)` is
   nondeterministic **20/20**; without ties, 0/20. Real router scores
   (sigmoid + e_score_correction_bias, fp32) evidently contain exact ties at
   some layers; tie resolution races run-to-run regardless of the `sorted`
   flag.
2. **Combine-stage deviations (NOT isolated).** In 3/24 forwards at layer 21
   (2/24 at 34), the **final** MoE output (`moe=out`, after
   `self.experts(...)` returns) deviated while the runner-internal
   `moe2=post`, top-k, logits and inputs were all bit-identical — i.e. the
   nondeterminism sits between the runner's returned tensor and
   `Glm5NextMoE.forward`'s return: the finalize/combine path (unfinalized
   MoE output materialisation, shared-expert combine, `routed_scaling`
   2.5x, or the TP combine of expert partials). I did not isolate it further.

Net: `sorted=True` (or `VLLM_BATCH_INVARIANT=1`, which however **cannot boot**
on this stack — see §7) removes the every-call nondeterminism and fixes
layer 3, but greedy output remains nondeterministic via (1) and (2).

## 5. What I could NOT exclude / did not run — stated plainly

* The kernel-level micro-mechanism of `torch.topk` nondeterminism (radix
  select + atomic slot writes in this torch-2.11/HIP build) is inferred from
  behaviour, not verified against kernel source.
* I did not dump real layer-21/34 logits to prove the exact-tie structure
  directly; the tie explanation rests on the standalone tie demo + identical
  logits with deviating top-k outputs in situ.
* Effect (2) in §4 (combine-stage deviations) is localized to a pipeline
  stage but **not** attributed to a single op. `moe_sum`/all-reduce
  exonerations on file used synthetic inputs; a repeat with real-shaped
  partials is the missing control.
* Layer-3-only conclusion is airtight; the "why layers 21/34 and not others"
  is data-dependent (tie structure of their score tensors) — plausible but
  not directly proven.
* DS4's determinism on this rig is likely explained by its different router
  path (note `fused_moe/router/dsv4_topk.py` exists) — not verified.
* Decode-side was out of scope (divergence is prefill-side, §1.7); MTP was
  off throughout, as required.
* I ran no GitHub-bound anything; no git commits; `PATCHES.md`, `README.md`,
  `PENDINGWORK.md`, `vllm-strix-halo.sh` untouched.

## 6. Reproduction details

Everything ran on box1 unless noted; rank-1 counterparts on box2. All patches
were revert-verified (see §7). Raw logs: **`~/tap-layer3/`** on both boxes.

| step | command |
|---|---|
| sanity reproduction | `python3 /tmp/bisect_run.py` |
| six-point + residual measurement driver | `python3 /tmp/l3_run.py <N>` (sends 2 warmups, truncates taps, N measured reps; N=8 rounds 1-4, N=24 rounds 5-6) |
| round-2 analyser | `python3 /tmp/l3_analyze2.py` |
| round-5 analyser | `python3 /tmp/l3a5.py` |
| op-level top-k A/B | `podman exec vllm-glm python3 /tmp/topk_det_test2.py` |
| tie demo | inline python via `podman exec` (see session log) |
| patch apply/revert (both containers) | `podman cp /tmp/<patch>.py vllm-glm:/tmp/ && podman exec vllm-glm python3 /tmp/<patch>.py apply|revert` (box2 via `scp` from box1) |

Patches (all in container `/tmp/`, copies in local workspace
`vsh-layer3/`): `tap_patch.py` (pre-existing), `l3_tap_patch.py` (six-point),
`l3_tap_patch2.py` (router), `l3_tap_patch3.py` (MoERunner),
`l3p4.py` (sorted=True control), `l3p5.py` (all-layer runner taps),
`l3p6.py` (all-layer final-output taps). Backups used `.l3orig*`, leaving the
previous session's `.orig` files untouched.

Raw logs kept:
* `~/tap-layer3/tap-round3-layer3-gate.rank{0,1}` — production code path,
  8 forwards, six-point + router + topk lines (the §2/§3 table data).
* `~/tap-layer3/tap-round6-sortedTrue-24reps.rank{0,1}` — sorted=True
  control, 24 forwards, all-layer lines (the §4 data). (A round-5 8-rep log
  was overwritten by the round-6 run; its numbers are quoted from analysis
  output in §2/§3 and the transcript.)

## 7. Server state when finished

* All six patches reverted in **both** containers; verified zero tap markers
  in `glm5next/nvidia/model.py`, `fused_moe/runner/moe_runner.py`,
  `fused_moe/router/grouped_topk_router.py`; no `.l3orig*`/`.taporig` files
  remain; the previous session's `.orig` files untouched.
* `~/vsh-config.yaml`: `glm53_prefix_cache: 1` restored (it had been flipped
  to 0 for the measurements — **note: the env-var route in the handoff does
  not work**: `vsh-cluster-restart.sh` evals `vsh-config`, which re-exports
  `VSH_GLM53_PREFIX_CACHE=1` over whatever you prefixed the command with;
  the yaml is the effective lever).
* `~/vsh-cluster-env.hybrid.sh` restored (a temporary
  `VLLM_BATCH_INVARIANT=1` export was added for one control attempt and
  removed).
* Final restart: production config, no `VSH_TAP_FILE`, prefix caching ON,
  MRV2, MTP off. Verified: `:1235` answers 200, serve env contains no tap or
  batch-invariant vars, RDMA `tbv_ar2` ready, 1 serve process, a live greedy
  completion returns normally. Tap logs stopped growing at 11:59.

**Ops notes for PENDINGWORK:**
* `VLLM_BATCH_INVARIANT=1` hard-fails GLM bring-up on this build:
  `ROCM_AITER_MLA_SPARSE: [batch invariance not supported]` (ValueError in
  `platforms/rocm.py get_attn_backend_cls`). A whole-mode control arm is
  impossible; targeted patches are required.
* `podman exec` inside `vsh-cluster-restart.sh` does not forward arbitrary
  caller env vars into ray/serve; the cluster-env file is the only reliable
  env path.
* Upstream relevance (human to decide): vllm `grouped_topk`'s
  `sorted=False` default + tie-race nondeterminism on ROCm affects every
  noaux_tc router (DeepSeek/GLM family) on this platform; the code comment
  shows the hazard is known for batch invariance but `sorted=True` is not
  sufficient for run-to-run determinism under exact fp32 ties.
