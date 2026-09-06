# RESULT — fp32 k-boundary ties in the GLM-5.3 MoE router scores (layers 21/34)

**Date:** 2026-09-05 · **Task:** dump the real router score tensors at layers
5 (control), 21, 34 and test the exact-fp32-tie explanation for the
"sorted=True still flips top-k at 21/34" defect (PENDINGWORK.md §10.3 item 1,
HANDOFF-layer3-result.md §4 effect 1). **Status: CONFIRMED, demonstrated on
the real tensors, with the op-level loop closed.**

---

## 1. The answer

**Yes — the real, live selection-score tensors at model layers 21 and 34
contain exact bitwise fp32 ties at the top-8 boundary, and the layer-5
control contains none.** The deployed `grouped_topk` is nondeterministic
precisely on those tie rows and deterministic everywhere else.

Measured on 6 identical prefill-only requests (740 prompt tokens,
`temperature=0`, `seed=1234`, prefix caching OFF, 2 warmups first), with the
router forced to `sorted=True` in situ (the exact arm in which the prior
session observed the 8/24 flip). Router scores dumped from
`Glm5NextMoE.forward` on both TP ranks; the reconstruction below applied to
`sigmoid(router_logits) + e_score_correction_bias`.

| layer | rl hash(es) over 6 fwds | rows w/ exact k-boundary tie | 1-ULP-not-equal | tie multiplicity |
|---|---|---|---|---|
| 3 (anchor) | 1 (`bc3d628ff22b`) | **0** / 740 | 0 (0 within 16 ULP) | — |
| 5 (control) | 1 (`d43ddaca6445`) | **0** / 740 | 0 (1 within 16 ULP) | — |
| **21** | 1 (`9baf850ea3f8`) — identical 6/6 | **2** / 740 (rows **40, 549**) | 0 (1 within 4 ULP, 1 within 16) | {2: 2} |
| **34** | 4 distinct states | **1** / 740 in 3 of the 4 states (rows 568, 489, 651); **0** in the 4th | 1 per tensor | {2: 2}, {2: 1}, {3: 1} |

Layer 21's two tie rows, exactly (biased value = 8th-highest = 9th-highest,
bitwise):

- **row 40:** experts **12** and **202**, biased = 12.275362 (fp32 hex
  `414467e2`). logits −0.489040 / −0.395469, sigmoid 0.380119652 / 0.402401417,
  bias 11.8952427 / 11.872961 — **different logits AND different biases; the
  two sums independently round to the same fp32 value.**
- **row 549:** experts **243** and **250**, biased = 12.1801996 (`4142e219`).

Layer 34 (state `693ffcde86d5`) **row 651** is a **multiplicity-3** tie:
experts 58, 162, 188, biased = 12.6659889; their biases are
12.6659708 / 12.6659708 / 12.6659718 (two bitwise-equal, third 1 ULP away)
with sigmoids ~1.7e-5 — the `e_score_correction_bias` values themselves
cluster tightly, which is what makes these rounding collisions likely.

Layer 34 varies across forwards (4 distinct rl hashes in 6 forwards) because
layer 21's tie flip sits upstream of it — expected, and itself evidence the
21 flip is live: the first sampled token again split ' Which' (5/6) vs
' Then' (1/6), the same bimodality the prior session measured (16/24 vs
8/24).

**Op-level closure on the real tensors** (20 identical calls each, in the
container, gfx1151):

| tensor | ties | raw `torch.topk(sorted=True)` | deployed `grouped_topk(sorted=True)` |
|---|---|---|---|
| L21 real rl+bias (`9baf850ea3f8`) | 2 | **1 row varies, 2/20 distinct** — varying row = **40** | **1 row varies, 2/20 distinct** — varying row = **40** |
| L34 state `693ffcde86d5` | 1 | 1 row varies, 2/20 — row **651** | 1 row varies, 2/20 — row **651** |
| L34 state `15b36f53c904` | 0 | 0 vary, 1/20 | 0 vary, 1/20 |
| L5 real rl+bias | 0 | 0 vary, 1/20 | 0 vary, 1/20 |
| L3 real rl+bias | 0 | 0 vary, 1/20 | 0 vary, 1/20 |

Every tie-bearing tensor is nondeterministic under `sorted=True` and exactly
at a tie row; every tie-free tensor is deterministic. (Raw `sorted=False`
on these same tensors matched the same arms — see caveats, §4.)

## 2. Arithmetic used, and how it mirrors `grouped_topk`

Read from `vllm/model_executor/layers/fused_moe/router/grouped_topk_router.py`
before measuring. For this model
(`n_routed_experts=288, num_experts_per_token=8, n_group=1, topk_group=1,
scoring_func="sigmoid", topk_method="noaux_tc"`, fp32 router dtype —
`config.json` text_config), the live path (`aiter fused MoE off:
VLLM_ROCM_USE_AITER=1 but VLLM_ROCM_USE_AITER_MOE=0`, so
`GroupedTopKRouter._compute_routing` → the Python `grouped_topk`) computes:

```
scores       = gating_output.sigmoid()                     # fp32
biased       = scores + e_score_correction_bias.unsqueeze(0)
group_scores = biased.view(n, 1, 288).topk(2, dim=-1)[0].sum(-1)   # [n,1]
group_idx    = torch.topk(group_scores, k=1)[1]             # always group 0
score_mask   = group_mask...expand...reshape                 # [n,288]
tmp_scores   = biased.masked_fill(~score_mask.bool(), -inf)
topk_ids     = torch.topk(tmp_scores, k=8, dim=-1, sorted=use_sorted)[1]
```

The analysis script re-executes **these exact lines** on the GPU with the
dumped tensors and **asserts** (not assumes): the group mask is all-ones and
`tmp_scores` is bitwise identical to `biased` — so the tensor `topk(k=8)`
sees is exactly `sigmoid(rl) + bias`. This holds for every dump analysed.

One methodological trap handled: `grouped_topk` is `@torch.compile`-ed with
the **inductor** backend on ROCm (`simple_compile_backend` default), and its
fused `sigmoid + add` differs from eager on ~0.15% of elements by exactly
1 ULP (measured per dump: 827–1564 differing elements of 212,960, max 1 ULP).
All tie statistics therefore use a replica of the two source lines under the
identical decorator; the deployed function's own returned weights
(renormalize off) match the eager reconstruction to ≤2 ULP. **The layer-21/34
tie rows classify identically under the eager and compiled variants**, so
the conclusion does not hinge on that choice.

## 3. Verdict

**The tie explanation is CONFIRMED.** The prior inference ("layer 21 returns
a different top-k in 8/24 forwards despite bit-identical router logits
because the biased scores contain exact fp32 ties at the k-boundary") is now
demonstrated directly: the canonical layer-21 score tensor — bit-identical
across all 6 forwards, exactly as in the prior 24-forward observation — has
2 exact bitwise ties at the 8th/9th position, the layer-5 control tensor has
none, and the deployed op is nondeterministic exactly on the tie rows and
nowhere else. "sorted=True is insufficient" stands, and the reason is now
measured, not inferred.

Also demonstrated: the ties are **fp32 rounding collisions** (distinct logits
and distinct biases whose sums round to the same float), not duplicated
parameters; and the tight clustering of `e_score_correction_bias` values
(e.g. three biases within 1 ULP at L34 row 651) is the enabling structure.

## 4. What I could NOT verify / did not run — stated plainly

- **Not every tie row manifests in a given sample.** Row 40 flipped in 20/20
  sampled arms; row 549's tie did NOT flip in those 20 calls. The per-row
  race is probabilistic; I cannot give the per-row flip probability, and the
  8/24 in-situ rate is consistent with, not quantitatively matched by, my
  20-call standalone samples.
- **The prior session's standalone `sorted=False` → 20/20-distinct result
  did not reproduce on real tensors.** On every real tensor here (including
  tie-free layer 3) standalone `sorted=False` matched `sorted=True`'s
  behaviour. Their synthetic-input test presumably had a different value
  structure. Consequence: the *in-situ* every-call `sorted=False`
  nondeterminism at layer 3 (prior §10.1, the layer-3 root cause) cannot be
  explained by exact ties alone — the canonical layer-3 tensor has zero
  boundary rows within even 16 ULP, yet in-situ it flipped every call. That
  mechanism (likely engine-concurrency-dependent) is outside this task and
  remains as unexplained as before; nothing here overturns the layer-3
  root-cause, which rests on in-situ evidence.
- **First measurement round ran the wrong arm (caught and discarded).** My
  env-gated `use_sorted` constant was read at module import inside the Ray
  workers *before* the `update_environment_variables` RPC delivered
  `VSH_TAP_FILE` (model.py is imported later, at model-load, which is why
  the dump tap itself saw the env). That round therefore measured the stock
  `sorted=False` arm — visible because layer-5 rl hashes were 4-distinct-in-4
  (impossible under sorted=True, where layers 0–20 are deterministic). Fix:
  the tap now also sets the router module global at forward time (before the
  first MoE router executes). The reported run verified the arm
  behaviourally: layer 3/5/21 rl hashes identical 6/6, matching the prior
  session's 24-forward result. Under that discarded stock-arm round, for the
  record, 4 different upstream states of layer 21 showed **0** boundary ties
  — the ties live in the canonical deterministic-path tensor, which only the
  sorted=True arm reaches.
- 6 measured forwards, not 24; layer-34 coverage is 4 distinct upstream
  states. The combine-stage defect (prior §4 effect 2) was not investigated.
- Tie classification at the exact ULP margin depends on the deployed fused
  kernel's rounding; mitigated (tie rows identical under eager and compiled
  variants; the deployed op itself flips on exactly those rows), but the
  kernel's internals were not read.
- `renormalize=True`/`routed_scaling_factor=2.5` (live config) were used for
  the op tests' fidelity but only affect weights, not ids.

## 5. Reproduction

Scripts (kept on box1): `~/tie_tap_patch.py` (apply|revert|verify; backs up
to `.tieorig`, never touches `.orig`/`.taporig`/`.l3orig*`),
`~/tie_run.py` (driver), `~/tie_analyse.py`, `~/tie_followup.py`
(details). Raw dumps: `~/tie-dump/` (rank0, 24 files
`L{3,5,21,34}_f{9..14}.pt`) and `~/tie-dump-rank1/` (box2 copies —
bitwise-identical, verified). Restart logs `/tmp/restart{,2,3}.log`.

```bash
# on box1; patch BOTH containers (rank 1 is on box2)
scp tie_tap_patch.py box1:~/ ; ssh box1 'scp ~/tie_tap_patch.py 10.0.2.2:~/'
podman exec -i -w ~ vllm-glm python3 tie_tap_patch.py apply   # each box
sed -i 's/^glm53_prefix_cache: 1/glm53_prefix_cache: 0/' ~/vsh-config.yaml
VSH_TAP_FILE=/home/davidcanar/tie-dump setsid nohup ~/vsh-cluster-restart.sh \
    > /tmp/restart.log 2>&1 &        # wait for :1235 -> 200 (4-6 min)
python3 ~/tie_run.py 6               # warmup x2, truncate, 6 measured reps
ssh 10.0.2.2 ...                      # copy box2 ~/tie-dump to box1 tie-dump-rank1
podman exec -i -w ~ vllm-glm python3 tie_analyse.py \
    /home/davidcanar/tie-dump /home/davidcanar/tie-dump-rank1
podman exec -i -w ~ vllm-glm python3 tie_followup.py
```

Prompt: `build(614)` from `/tmp/det_sweep.py` (740 prompt tokens, id0=2833,
embedding chain identical to all prior sessions — layer-3 rl hash
`bc3d628ff22b` matches HANDOFF-layer3-result.md §2 exactly, which is the
instrumentation cross-check).

The tap in `models/glm5next/nvidia/model.py` fires on
`VSH_TAP_FILE` (unset ⇒ stock, zero effect) at layers 3,5,21,34 during
prefill only, and saves `router_logits` (fp32 [rows,288]) +
`e_score_correction_bias` (fp32 [288]) + `rl_sha12` metadata per forward.
The router patch forces `use_sorted=True` **only while the tap env var is
set** (and the tap re-asserts the flag at forward time — see §4 bullet 3).

## 6. Server state when finished

- Both containers reverted to stock and verified:
  `model.py` md5 `e93b8176ee67ba6df5dc17b9a8d867c9`,
  `grouped_topk_router.py` md5 `da7b32ec80cd79291a2b58e3df8ccbbe`;
  zero `_vsh_tie`/`_VSH_TIE` markers; all `.tieorig` backups removed;
  pre-existing `.orig` untouched.
- `~/vsh-config.yaml` restored to `glm53_prefix_cache: 1`.
- Server restarted via `~/vsh-cluster-restart.sh` with **no** tap env var:
  `:1235` returns 200, RDMA rank0+rank1 ready, and a greedy completion
  returns normally (' Paris. In French, Paris is pronounced…').
- Nothing committed/pushed anywhere; no GitHub access used.
