# RESULT — "combine-stage deviation" at layer 21 is a TAP/ANALYSIS ARTIFACT, not a defect

**Verdict: effect (2) in `HANDOFF-layer3-result.md` §4 is RETRACTED — it is an
instrumentation-coverage artifact, not a real defect.** There is **no**
nondeterministic op between the MoE runner's returned tensor and
`Glm5NextMoE.forward`'s return. The evidence, in one line: over all
**1008 (forward, MoE-layer) blocks** of the retained round-6 log, the final MoE
output is a **perfect function of (MoE input, rank0 routed partial, rank1 routed
partial)** — **0 ambiguous cases**; the prior session's analyser only ever read
**rank0**, whose routed partial is a *TP-local* tensor that is **never** equal
across ranks (0/24 at all 42 layers), so a router tie race firing on **rank1
only** was invisible to it.

Mechanism in one sentence: the router's `torch.topk` tie race resolves
**independently on each TP rank**, so at layer 21 rank0's `topk_ids` flip in
8/24 forwards and rank1's flip in a **different** 6/24 — and the 3 forwards the
prior session flagged (`{43,56,58}`) are **exactly the rank1-only flips**, i.e.
the forwards where the only analysed rank looked perfectly clean.

Nothing was restarted, patched or re-measured to reach this: Step 1
(analysis-only, on the retained logs) settled it, so Step 2 was never entered.

---

## 1. The decisive evidence

Data: `~/tap-layer3/tap-round6-sortedTrue-24reps.rank0` (box1) and
`~/tap-layer3/tap-round6-sortedTrue-24reps.rank1` (box2), 24 identical 740-token
prefill-only forwards (tap fwd numbers 37–60), `sorted=True` arm, both ranks.
Structure verified clean: 42 MoE layers × 5 tap lines + 6 `pt=` lines = 216 L3
lines per forward, 5184 total; `embed=9b0d90d30511` in 24/24 (input guard).

### 1.1 Layer 21 — the two ranks' routers do not agree

| field | rank0 | rank1 |
|---|---|---|
| `moe=in` x | **1 distinct** (24/24) | **1 distinct** (24/24), same value |
| `rl` | **1 distinct** `9baf850ea3f8` | **1 distinct**, **same value** |
| `moe2=topk` ti | **2 distinct** — flips in **8/24**: {40,41,46,47,49,51,55,59} | **2 distinct** — flips in **6/24**: {40,41,43,49,56,58} |
| `moe2=post` x | 2 distinct, tracks rank0's ti 1:1 | 2 distinct, tracks rank1's ti 1:1 |
| `moe=out` x | **4 distinct** | **4 distinct, identical values to rank0** (1008/1008 blocks rank-identical) |

The flip sets are **not** the same set. Intersection {40,41,49}; rank0-only
{46,47,51,55,59}; **rank1-only {43,56,58}**.

Router logits are bit-identical on both ranks and across all 24 forwards
(`9baf850ea3f8` — which is **byte-for-byte the canonical layer-21 score tensor
`HANDOFF-tie-result.md` dumped independently**), yet the two ranks' top-k
disagree — i.e. the tie race is **per-rank**, exactly as the tie analysis
predicts, and *worse* than previously reported.

### 1.2 Layer 21 — `out` is an exact function of the two ranks' router states

| rank0 topk | rank1 topk | layer-21 `moe=out` | forwards | n |
|---|---|---|---|---|
| steady | steady | `2d571580dc87` | 37,38,39,42,44,45,48,50,52,53,54,57,60 | 13 |
| flip | flip | `f7c49bee3edd` | 40,41,49 | 3 |
| flip | **steady** | `c347cae5337f` | 46,47,51,55,59 | 5 |
| **steady** | flip | `0fd4b33c31c3` | 43,56,58 | 3 |

4 keys → 4 distinct outputs, no collision, no ambiguity. Every one of the 24
forwards is accounted for.

### 1.3 The prior session's "3/24" is reproduced *exactly* and identified

Recomputing the prior definition on rank0 ("all of `in`/`rl`/`tw`/`ti`/`post`
at their modal values, `out` deviating") at layer 21 yields **{43, 56, 58},
n=3** — and the rank1-only topk-flip set is **{43, 56, 58}, n=3**.
`identical sets? True`. That is the whole of effect (2) at layer 21.

### 1.4 Global closure — no combine-stage nondeterminism anywhere

Across **all 42 MoE layers × 24 forwards = 1008 blocks**:

| key used to predict `moe=out` | ambiguous keys |
|---|---|
| (rank0 `in`, rank0 `post`) — **what the prior analyser had** | **8** (layer 21 ×2, 32 ×1, 34 ×2, 35 ×1, 37 ×2) |
| (rank0 `in`, rank0 `post`, **rank1 `post`**) | **0** |

The combine stretch is therefore **deterministic on real data at real shapes**
(M=740, hidden 4096, 288 experts, topk 8, TP=2) — the exoneration the previous
session said was missing (§10.6 item 2) is delivered by this analysis, in situ.

### 1.5 Per-rank disagreement is the general signature, not a layer-21 quirk

Ranks disagree on `topk_ids` at **12 of 42** MoE layers: first at **layer 21**
(8/24), then 22 (4), 28 (11), 32 (4), 34 (14), 35 (19), 36 (14), 37 (17),
38 (4), 39 (6), 40 (2), 42 (1). Layers 3–20 are **0/24** on both ranks (the
`sorted=True` fix holds there, on both ranks). Layer 34's 14/24 disagreement is
the amplified downstream consequence of layer 21 plus its own tie rows.

---

## 2. Instrumentation audit (Step 1, as required)

### 2.1 Which exact tensor object does each tap hash?

**`moe2=post`** — `/tmp/l3_tap_patch3.py` line 93
(`_vsh_l3_tap(moe2="post", x=fused_out)`), inserted immediately after the
`forward_modular` call inside `MoERunner._apply_quant_method`. In the **stock**
file that call is `moe_runner.py:630-636`
(`fused_out = self.routed_experts.forward_modular(...)`), and
`_apply_quant_method` returns it at `moe_runner.py:642-646`.

`fused_out` **is** the tensor that flows downstream — it is returned up through
`_forward_impl` → `_maybe_combine` (`moe_runner.py:921`) → `forward`, where it
becomes `fused_output` after `_unpack` (`moe_runner.py:756`). It is **not** a
view, a stale workspace, or an `UnfinalizedMoEOutput` (the logs contain real
12-hex hashes and **zero** `err:` fallbacks — verified, 0 in both rank files).

**But it is a TP-LOCAL PARTIAL.** It is the routed-expert output computed from
this rank's expert shard only, before any all-reduce. Proof from the retained
logs: `moe2=post` is **never** bit-identical across rank0/rank1 — 0/24 at every
one of the 42 layers, in both the round-6 log and the round-3 layer-3 log
(md5 of all `moe2=post` lines differs across ranks, while the md5 of all
`pt=` lines is identical). This is the whole bug in the prior inference.

**`moe2=topk`** — `l3_tap_patch3.py:84`, hashing `topk_weights` and `topk_ids`
from `self.router.select_experts(...)` (`moe_runner.py:625-630` stock). Also
TP-local in the sense that each rank runs `select_experts` redundantly and
independently.

**`moe2=pre`** — `l3_tap_patch3.py:67`, hashing the runner's `hidden_states`
and `router_logits`. Both replicated: bit-identical across ranks (verified).

**`moe=in` / `moe=out`** — `/tmp/l3_tap_patch2.py:54-71`, inside
`Glm5NextMoE.forward`. `moe=out` is inserted between stock `model.py:267`
(close of `self.experts(...)`, opening at `model.py:265`) and `model.py:269`.

### 2.2 Is `moe=out` hashed before or after the reduce / scaling / shared add?

**After all of them.** Everything in the combine stretch happens *inside*
`self.experts(...)` (`MoERunner.forward`, `moe_runner.py:678-788`), i.e. before
the `moe=out` tap fires:

| step | stock line |
|---|---|
| shared-expert output all-reduce (early path) | `moe_runner.py:453` |
| latent routed-output all-reduce before transform | `moe_runner.py:473` |
| **`fused_output *= self.routed_scaling_factor` (2.5, in-place)** | **`moe_runner.py:420`** |
| routed output transform | `moe_runner.py:791` (`apply_routed_output_transform`) |
| shared + routed add | `moe_runner.py:780` |
| **final TP all-reduce** | `moe_runner.py:509` (via `_maybe_reduce_final_output`, called at `moe_runner.py:784`) |
| `moe=out` tap | `model.py`, between 267 and 269 |

The only things after the tap are the sequence-parallel all-gather
(`model.py:270-273`, inactive or hash-neutral here — see §5) and the final
`.view()` (`model.py:275`), both shape-only.

**Consistency anchor confirming this:** `moe=out` equals the model-level
per-layer input hash `h` of the *next* layer in **984/984** non-final blocks.
(The 24 exceptions are all layer 44, where `Glm5NextDecoderLayer.forward` takes
the last-layer branch `model.py:508-510` — `x = self.hc_post(x, residual, post,
comb); x = hc_contract(x, self.n); return x, None, None, None` — so the
model-level `h` there is post-`hc_contract`, not the raw MoE output. Benign
bookkeeping, and it explains the `r=none p=none c=none` on every layer-44 line.)

### 2.3 Does anything mutate a tapped tensor in place after the tap runs?

**Yes — one case, and it is benign.** `moe_runner.py:420`
`fused_output *= self.routed_scaling_factor` is an **in-place** multiply on the
very object hashed at `moe2=post`. This means the `moe2=post` hash describes a
**pre-scale** state that no longer exists at return (the returned tensor is
2.5× it, modulo the shared-expert compensation at `moe_runner.py:422`). It does
**not** create a false deviation: the hash is computed before the mutation, and
the ×2.5 is a compile-time constant applied identically in every forward. It is
recorded here because it is exactly the class of thing this audit was meant to
find, and because it means **`moe2=post` values are not directly comparable to
`moe=out` values** — only within-field, across forwards, as used here.

`moe=out`'s tensor (`final_hidden_states`) is not mutated in place after its
tap; the subsequent `.view()` is non-mutating, and the sequence-parallel branch
rebinds rather than mutating.

### 2.4 Was `topk` hashed with enough coverage to exclude a tie flip?

**Yes.** `/tmp/tap_patch.py:43-52`:

```python
def _vsh_hash(t):
    ...
    return _vsh_hashlib.sha256(
        t.detach().to(_t.float32).cpu().numpy().tobytes()
    ).hexdigest()[:12]
```

- **Full coverage, no truncation, no slicing** — the entire `topk_ids` [740,8]
  and `topk_weights` [740,8] tensors are hashed (bf16→fp32 is lossless).
- `.cpu()` blocks on the current stream, so the value read is the finalized
  one; there is no async/staleness window.
- The `try/except → "err:<TypeName>"` fallback (line 51-52) is a theoretical
  hazard (a persistent failure would make every hash "equal"), but
  **0 `err:` strings occur in either rank file**, so it never fired.
- `moe2=topk` hashes **both** `tw` and `ti`, so neither a weight-only nor an
  id-only difference could hide.

**So the prior topk tap was sound — and that is the point.** The topk tap
correctly reported rank0's topk as identical in forwards 43/56/58. It was
rank**1**'s topk that differed, and rank1's log was never read by the round-6
analyser.

### 2.5 Where exactly the prior analysis went wrong

`/tmp/l3a6.py` — the round-6 analyser that produced the effect-(2) claim —
contains, at **line 84**, only:

```python
analyse("rank0", read_lines(None))
```

The rank1 call that its predecessor `/tmp/l3a5.py` had (lines 101-103:
`moe_seq = {}; analyse("rank1", read_lines(BOX2))`) was **dropped**, and the
now-unused `BOX2` constant remains defined at `l3a6.py:10`. Every number in
`HANDOFF-layer3-result.md` §4 item 2 is therefore a **single-rank observation
compared against a two-rank quantity**: `moe2=post` is rank-local,
`moe=out` is the all-reduce of both ranks.

The *inference* was also over-strong. "Identical inputs, identical routing,
identical runner-internal intermediate but different final output is
impossible" is true **within one rank**. It is **not** true across a TP
all-reduce, where the output is a function of *both* ranks' partials and the
router runs redundantly per rank. `moe2=post` never being rank-identical
(§2.1) was the visible clue that this was a partial, and it was available in
the retained logs the whole time.

---

## 3. What inherits the same flaw, and what does not

**RETRACTED (fully explained as artifact):**
- `HANDOFF-layer3-result.md` §4 item 2 — the combine-stage deviation. There is
  no such defect in this data.
- `PENDINGWORK.md` §10.3 item 2, and §10.6 item 2's framing ("isolate the
  combine-stage defect"). The "repeat the `moe_sum`/TP-combine exonerations with
  real-shaped partials" control is now **delivered in situ** by §1.4 (0/1008
  ambiguous) — the TP combine is exonerated on real data at real shapes.

**NEEDS REWORDING (not wrong, but understated / over-broad):**
- `HANDOFF-layer3-result.md` §2: *"Hashes were identical on rank0 and rank1 at
  every point (including the diverging values — the ranks are in lockstep)"*.
  Verified **true for the layer-level taps only** (md5 of all `pt=` lines is
  identical across ranks on the retained round-3 log). It is **false for the
  runner taps**: md5 of all `moe2=post` lines differs across ranks, and
  `moe2=post`/`moe2=topk` are rank-identical in **0/24** forwards at every
  layer. Anyone extending that sentence to the runner taps repeats this
  session's mistake.
- `HANDOFF-layer3-result.md` §4 item 1 / `PENDINGWORK.md` §10.3 item 1
  ("top-k took a second value in **8/24**"). **Not retracted — strengthened.**
  The correct statement is that the two ranks **disagree** on layer-21 top-k ids
  in 8/24 forwards, and at least one rank is flipped in **11/24**. Because the
  race is per-rank, no per-rank-deterministic fix (including `sorted=True`, and
  including a tie-break inside `torch.topk`) can make TP=2 reproducible unless
  the selection is made bit-identical **across ranks** (e.g. computed once and
  broadcast, or made a deterministic function of the score tensor). This is a
  materially stronger requirement than the current write-up implies.
- `HANDOFF-layer3-result.md` §4's layer-34 companion number "**2/24**" is **not
  reproducible** from the retained round-6 log: under the same modal definition
  I get **7/24** ({38,39,48,52,53,57,60}); under the rank0-only functional test
  there are exactly **2 ambiguous keys**. The modal test is lineage-incoherent
  at layer 34 (the modal `out` belongs to the {46,47,51,55,59} lineage while
  the modal `in`/`rl`/`tw`/`ti`/`post` belong to the 13-forward clean lineage),
  so the count depends on definition. Either way **100% of layer-34 deviations
  are resolved by adding rank1's partial** (§1.4). The 2/24 most plausibly came
  from counting ambiguous keys, or from the round-5 8-rep log that was
  overwritten.

**UNAFFECTED:**
- The layer-3 root cause (`sorted=False` in `grouped_topk`,
  `grouped_topk_router.py`) — untouched, and reconfirmed here: layers 3–20 show
  0/24 rank disagreement and 1 distinct value in every field under `sorted=True`.
- `HANDOFF-tie-result.md` in full — op-level on dumped tensors, no cross-rank
  inference. Its canonical layer-21 score hash `9baf850ea3f8` matches the
  round-6 log **on both ranks, 24/24**, which is a good independent
  cross-session instrumentation check. Its "not every tie row manifests"
  caveat is if anything reinforced: row 40's race fires on both ranks in
  {40,41,49}, on rank0 only in {46,47,51,55,59}, on rank1 only in {43,56,58},
  and on neither in the remaining 13.
- The DSA/sparse-attention and expert-GEMM exonerations — no cross-rank
  quantity involved.

---

## 4. What I could NOT verify, and controls I did not run

Stated plainly, in the spirit of the two prior retractions:

1. **Nothing was re-run live.** The entire verdict rests on the retained
   round-6 logs (both ranks). I cannot exclude that a *fresh* run would show a
   genuine combine defect — only that this dataset, which is the dataset the
   claim was made from, contains zero evidence of one. This is per the Step-1
   instruction ("NO restart, analysis only"); it is still a limitation.
2. **Step 2 was never entered** — no fresh taps on the combine stretch, so no
   step in that stretch is individually named as clean. The stretch is
   exonerated *as a whole* by the functional test (§1.4), not op by op.
3. **I did not determine which all-reduce line actually fires.** Whether the
   ROCm fused kernel reports `output_is_reduced()` (making `moe_runner.py:453`
   the live reduce) or the late path (`moe_runner.py:509`) is the live one was
   not checked. Immaterial to the verdict (either is a deterministic sum of the
   two partials) but it means I cannot name the exact reduce op.
4. **`use_sequence_parallel_moe` was not read from the live config.** Evidence
   it is inactive or hash-neutral here: `moe=out` equals the next layer's input
   hash in 984/984 non-final blocks, and `model.py:269-273` would all-gather
   between them if it were active with `already_sequence_parallel=False`.
5. **The specific differing experts were not identified.** I did not dump
   `topk_ids` at the rank1-only forwards ({43,56,58}) to confirm *which* row
   and *which* expert pair flipped. The mechanism is inferred from hash
   inequality plus the tie rows (40, 549) already established by
   `HANDOFF-tie-result.md`; that loop was not closed here.
6. **No isolated op-level control was run** — in particular the task's
   "real-shaped partials" `moe_sum`/all-reduce replay was not executed. It is
   unnecessary for this verdict (the in-situ data already covers real shapes),
   but it means the TP-combine exoneration is **observational**, not
   reproduced standalone.
7. **Per-rank independence of the tie race was not tested op-level** — it is
   demonstrated behaviourally at layer 21 (8/24 vs 6/24 with disjoint-only
   members) but I did not run two processes against the same tensor to
   reproduce it in isolation.
8. **The stock `moe_runner.py` md5 `466d34283c42daa6773aaaf691cf5` has no
   prior handoff reference**; I verified it is tap-free by grep (`_vsh_l3_tap`
   absent) and byte-identical across both boxes, not against a recorded
   pre-session hash. `model.py` (`e93b8176ee67ba6df5dc17b9a8d867c9`) and
   `grouped_topk_router.py` (`da7b32ec80cd79291a2b58e3df8ccbbe`) **do** match
   the stock md5s recorded in `HANDOFF-tie-result.md` §6.
9. **The prior "2/24 at layer 34" is unexplained** (see §3). I flag it rather
   than silently substitute my number.
10. **24 forwards, one prompt, one layer-21 tensor.** The 8-vs-6 split and the
    4-way output partition are one sample; per-row/per-rank flip probabilities
    are not established (same caveat as `HANDOFF-tie-result.md` §4).

---

## 5. Reproduction

All on box1. Analysis scripts kept in **`~/combine-dump/`** (also at
`/tmp/ca_*.py`); raw logs unchanged at `~/tap-layer3/` (box1) and
`~/tap-layer3/` (box2).

```bash
# 1. bring rank1's log next to rank0's (box1)
ssh davidcanar@10.0.2.2 "cat ~/tap-layer3/tap-round6-sortedTrue-24reps.rank1" \
    > /tmp/rank1-round6.log

# 2. the decisive functional test: is `out` a function of (in, r0post, r1post)?
python3 /tmp/ca_verify.py          # -> "TOTAL ambiguous ... keys across all 42 layers: 0"
                                   #    plus the layer-21 4-way map and the flip-set comparison

# 3. reconcile the prior session's exact "3/24"
python3 /tmp/ca_reconcile.py       # -> layer 21 eff2 == rank1-only == [43, 56, 58], "identical sets? True"

# 4. per-layer cross-rank router disagreement table
python3 /tmp/ca_summary.py         # -> 12/42 layers disagree; first at layer 21

# 5. rank0-only view (reproduces the artifact)
python3 /tmp/ca_analyse2.py        # -> layer 21 "AMBIGUOUS (same upstream, different out)"
python3 /tmp/ca_analyse.py         # -> layer 21 out[11 dev], post[8 dev]  (= the prior 11-8=3)
```

Cross-rank md5 sanity checks used in §2:
```bash
grep "pt="    ~/tap-layer3/tap-round3-layer3-gate.rank0 | md5sum   # == rank1
grep "moe2=post" ~/tap-layer3/tap-round3-layer3-gate.rank0 | md5sum # != rank1
grep -c "err:" ~/tap-layer3/tap-round6-sortedTrue-24reps.rank0      # 0
```

Code read for the audit (stock, in-container paths):
`vllm/models/glm5next/nvidia/model.py` (250-275, 495-511, 656-703),
`vllm/model_executor/layers/fused_moe/runner/moe_runner.py`
(365-511, 583-660, 678-935, esp. 420/453/473/509/630/780/784),
`vllm/model_executor/layers/fused_moe/routed_experts.py` (1224-1259).
Patches audited: `/tmp/tap_patch.py`, `/tmp/l3_tap_patch.py`,
`/tmp/l3_tap_patch2.py`, `/tmp/l3_tap_patch3.py`, `/tmp/l3p4.py`,
`/tmp/l3p5.py`, `/tmp/l3p6.py`; analysers `/tmp/l3a5.py`, `/tmp/l3a6.py`.

---

## 6. Server state when finished

**Untouched throughout this task.** No patch was applied to either container,
no restart was performed, no yaml key was changed.

- `~/vsh-config.yaml`: `glm53_prefix_cache: 1` (line 20, production) — never
  modified this session.
- No `VSH_TAP_FILE` (or any tap var) in the environment of any vLLM process
  (`/proc/*/environ` scan: 0 hits).
- Both containers verified stock and **byte-identical to each other**:
  `model.py` `e93b8176ee67ba6df5dc17b9a8d867c9`,
  `moe_runner.py` `466d34283c42daa6773aaaf691cf5`,
  `grouped_topk_router.py` `da7b32ec80cd79291a2b58e3df8ccbbe`;
  zero `_vsh_*`/`vsh round-*` markers in `model.py`, `moe_runner.py`,
  `grouped_topk_router.py`.
- No backup files of my own were created, so none to remove. Pre-existing
  `.orig` backups (`attention.py.orig`, `kda.py.orig`, `model.py.orig`,
  box1 only, dated Sep 4) were left untouched; no `.taporig`/`.l3orig*`/
  `.tieorig` files exist on either box.
- `:1235` returns **200** on `/v1/models`; a live greedy completion
  (`temperature=0`, model id `glm-5.3-flash`) returns normally.
- Nothing committed, pushed, or posted anywhere. `PATCHES.md`, `README.md`,
  `PENDINGWORK.md`, `vllm-strix-halo.sh`, the other `HANDOFF-*.md` files and
  `.git` were not modified. The only files created are this handoff,
  `~/combine-dump/` (5 analysis scripts + a copy of the rank1 log), and
  scratch copies at `/tmp/ca_*.py`, `/tmp/rank1-round6.log`,
  `/tmp/r1-round3.log` on box1.

---

## 7. Where this leaves the investigation

One open defect remains, and it is now better characterized than before:

**The MoE router's expert selection is computed redundantly and
non-bit-identically on each TP rank.** Router logits are bit-identical across
ranks (verified, layer 21: `9baf850ea3f8` on both, 24/24) — the divergence is
entirely inside the per-rank `torch.topk` tie race on the k-boundary ties that
`HANDOFF-tie-result.md` measured at rows 40 and 549. Consequences:

- `sorted=True` fixes layer 3 and layers 0-20 (0/24 rank disagreement there),
  but cannot fix layers 21/34, because a per-rank tie race is per-rank random.
- Any real fix must make the **selection** identical across ranks — e.g. compute
  `topk_ids`/`topk_weights` once and broadcast, or replace the tie race with a
  deterministic total order (score, then expert index). Note `topk_group=1`
  and `n_group=1` here, so the only nondeterministic op in the path is the
  final `torch.topk(tmp_scores, k=8)`.
- The combine path, the shared-expert add, the 2.5x routed scaling and the TP
  all-reduce are all exonerated on real data at real shapes by this analysis
  (§1.4) and need no further investigation on this evidence.
