<!-- Ready to paste. One placeholder remains:
     Signed-off-by name/email (DCO; add via `git commit -s`).
     End-to-end validation numbers are filled in below (2026-09-05). -->

**Title:**

```
[Bugfix][MoE] Deterministic expert selection in the grouped_topk Python fallback
```

---

## The strongest argument, up front: the fallback does not match the kernel

This PR does not introduce a new ordering. **Value descending, expert index
ascending is already the defined, implemented and tested output contract of
the fused CUDA kernel** — the Python fallback simply does not match it.

The single-group fused kernel packs value and index into one comparison key
(`csrc/libtorch_stable/moe/moeTopKFuncs.cuh`):

```cpp
static __host__ __device__ inline TypeCmp makeCmpVal(T val, int32_t idx = 0) {
    auto valueBits = cub::Traits<T>::TwiddleIn(...);
    TypeCmp compactTmp = valueBits;
    compactTmp = (compactTmp << kMoveBits) | (0xFFFF & (kMaxIdx - idx));
    // Use 65535 minus idx to give higher priority to elements with smaller
    // indices.
    return compactTmp;
}
```

The multi-group path is stable for the same reason
(`csrc/libtorch_stable/moe/grouped_topk_kernels.cu`):

```cpp
template <bool greater, typename T, typename idxT>
__forceinline__ __device__ bool is_better_than(T val, T baseline, idxT index,
                                               idxT baseline_index) {
  bool res = (val > baseline && greater) || (val < baseline && !greater);
  if (val == baseline) {
    res = (index < baseline_index && greater) ||
          (index < baseline_index && !greater);
  }
  return res;
}
```

And the contract is already asserted by an existing test,
`tests/kernels/moe/test_grouped_topk.py::test_grouped_topk_single_group_stable_ties`
(all-zero logits → every score tied → the expected ids are ascending):

```python
def test_grouped_topk_single_group_stable_ties(num_experts: int):
    logits = torch.zeros((1, num_experts), dtype=torch.bfloat16, device="cuda")
    bias = torch.zeros((num_experts,), dtype=torch.float32, device="cuda")
    ...
    expected_ids = torch.arange(16, dtype=torch.int32, device="cuda")[None]
```

Most tellingly, the **reference implementation in that same test file already
uses precisely the ordering this PR installs** — a stable descending sort:

```python
def _single_group_reference(...):
    ...
    indices = torch.argsort(
        scores + bias.float(), dim=-1, descending=True, stable=True
    )[:, :topk]
```

So the fused kernel, the test suite's reference, and #55122's convention for
the sparse indexer all agree on value-desc/index-asc. The Python fallback is
the only piece that does not, and it is the piece that actually runs on ROCm.
This PR makes it agree.

This PR makes the Python fallback select under the same total order
(value descending, index ascending on bitwise-equal scores), so the two
implementations of `grouped_topk` finally agree on ties. It also matches the
convention #55122 chose for the sparse indexer's `persistent_topk`
("ascending index order ... top-k by value desc, index asc").

## The bug, for someone with no context

The Python `grouped_topk` selects experts with
`torch.topk(..., sorted=envs.VLLM_BATCH_INVARIANT)` — i.e. `sorted=False` by
default — at three sites in
`vllm/model_executor/layers/fused_moe/router/grouped_topk_router.py`
(group selection L135, expert selection L148 and L152-154 as of 8bf39632).

Measured on a real router score tensor from a GLM-5.3 prefill (740x288 fp32,
k=8; gfx1151, ROCm 10.0, torch 2.11), 20 identical calls of the raw ops:

| arm | distinct raw results / 20 | distinct selected sets / 20 |
|---|---:|---:|
| `sort(descending=True, stable=True)[:k]` (this PR) | **1** | **1** |
| `topk(sorted=False)` (today's default) | **20** | 2 |
| `topk(sorted=True)` | 2 | 2 |

Two independent defects:

1. **`sorted=False` permutes the returned order on every row, every call.**
   The selected set is correct, but the order of the returned top-k differs
   on 740/740 rows from call to call (a 20-call follow-up on other layers
   measured ≥715/740 — essentially every row). The gathered routing weights
   are permuted to match, so the downstream k-term summation order changes
   and the MoE output differs run-to-run.
2. **`sorted=True` still flips the selected set.** The real tensors contain
   bitwise-equal fp32 biased scores exactly at the k-boundary (rows 40 and
   549: experts {12, 202} and {243, 250} — different logits *and* different
   biases whose fp32 sums round to the same value), and `torch.topk`
   guarantees no tie-break between equal values. On row 40, 20-call sampling
   shows the set flipping between expert 202 (39/60) and expert 12 (21/60).
3. `torch.use_deterministic_algorithms(True)` does **not** help: neither in
   `warn_only` nor `strict` mode does it raise, warn, or change `topk`'s
   behaviour on these tensors.

### Why this has stayed invisible: the boundary is at 256 experts

`sorted=False` licenses *any* order, and above 256 columns `torch.topk` takes
a multi-pass path whose output order is not merely unsorted but differs
between calls. The boundary is exact. 20 identical
`torch.topk(x, k=8, sorted=False)` calls on a `64 x E` tensor, same stack as
above:

| experts (E) | distinct results / 20 | output descending |
|:--|---:|:--|
| 128, 250, 255, **256** | 1 | yes |
| **257**, 258, 260, 264, 272, 288, 512, 1024, 2048 | **20** | no |

The same split holds for any `k >= 4` (`k <= 2` is order-trivial), and
`sorted=True` at E=288 gives 1/20 - consistent with defect 2 needing a real
tie rather than mere width.

So the blast radius is **models with more than 256 routed experts**.
DeepSeek-V3/R1 has exactly 256 and sits on the safe side of the boundary,
which is presumably why this has gone unnoticed; GLM-5.3's 288 is over it. The
boundary is an implementation detail of one `topk` on one backend, though -
the fallback should not depend on unspecified ordering at any width.

## The change

A module-level helper realising the kernel's total order, used at all three
selection sites:

```python
def _deterministic_topk_indices(x: torch.Tensor, k: int) -> torch.Tensor:
    """Top-k indices under a total order: value descending, index ascending."""
    return x.sort(dim=-1, descending=True, stable=True).indices[..., :k]
```

- L135: `group_idx = _deterministic_topk_indices(group_scores, topk_group)`
- L148: `topk_ids = _deterministic_topk_indices(tmp_scores, topk)`
- L152-154: `topk_ids = _deterministic_topk_indices(tmp_scores, topk)`
  followed by `topk_weights = tmp_scores.gather(1, topk_ids)` (weights keep
  coming from `tmp_scores`; the bias branch still gathers from
  `original_scores`)

The now-unused `use_sorted = envs.VLLM_BATCH_INVARIANT` local and its
comment are removed. The group-score reduction at L126
(`view(...).topk(2)[0].sum(-1)`) is deliberately untouched: it reduces
values only, and a tie at its k-boundary is between equal values, so the
sum is unaffected.

## This is not ROCm-only

The fused `_moe_C` kernel is taken only when *all* of the following hold
(`grouped_topk_router.py` L91-97):

```python
envs.VLLM_USE_FUSED_MOE_GROUPED_TOPK   # default True
and current_platform.is_cuda()         # False on ROCm
and num_expert_group <= 32
and topk <= 32
and e_score_correction_bias is not None
```

So **CUDA also runs the Python path** whenever `e_score_correction_bias is
None` (softmax-grouped models without a correction bias), or
`VLLM_USE_FUSED_MOE_GROUPED_TOPK=0`, or a shape outside the kernel's tier
table (`num_expert_group > 32`, `topk > 32`). The bug is ROCm-*default*, not
ROCm-only — it is latent on CUDA for the no-bias configs.

## Performance

Same real 740x288 fp32 tensor, k=8, CUDA-event timing, N=50 after warmup
(gfx1151; the GPU is shared so absolute values are contention-inflated —
the relative ordering was reproduced three times):

| arm | median | p90 |
|---|---:|---:|
| `topk(sorted=False)` (today) | 97.8 us | 99.5 us |
| `topk(sorted=True)` | 101.7 us | 102.5 us |
| `sort(stable=True)` + slice (this PR) | **71.6 us** | 72.6 us |

Per prefill forward (42 MoE layers): 4.11 ms -> 3.01 ms, i.e. the fix
**saves ~1.1 ms of a ~2.4 s prefill** rather than costing anything. Cost in
memory: ~2.4 MiB transient per call at this shape (832 KiB values + 1.6 MiB
int64 indices, freed immediately; scales with tokens x experts — ~107 MiB at
32k tokens x 288 experts) vs ~69 KiB for `topk`.

Honest caveat: a full sort is O(E log E) vs `topk`'s O(E log k), so at much
larger expert counts the trade could invert. It is also data-dependent: on
`randn` synthetic scores at this shape `topk` costs ~78 us and the two arms
are comparable — the measured win is on real router data, not universal.
An env-gated variant is possible, but `VLLM_BATCH_INVARIANT` is documented
NVIDIA-SM90-only, so ROCm users cannot opt into a gated mitigation in a
supported way; correctness fixes should not be platform-flag-gated.

## Behavioural delta

- Confined to exact bitwise ties: **1 row in 740** on the real tensors.
- The ascending-index winner is always one of the outcomes `topk` already
  produced (across all measured tensors: 0 rows where the ascending-index
  selection was never returned by `topk` in 60 calls) — the fix pins the
  winner, it does not introduce a new selection.
- On tie-free rows the result is bitwise identical to `topk(sorted=True)`
  in both set and order. Relative to today's default (`sorted=False`), the
  returned order becomes fully sorted — that order change is the point of
  the fix.
- Verified picks on the real tensors: row 40 -> expert 12 (not 202),
  row 549 -> 243, layer-34 row 651 -> 58 — the lower index in every case.

## Known gap (pre-existing, disclosed)

AITER's `biased_grouped_topk` (taken on ROCm when `VLLM_ROCM_USE_AITER_MOE=1`)
is deterministic, but its tie-break is opaque and demonstrably **not**
ascending-index: on the real tie rows it picks 202/250/188 where the
ascending-index rule picks 12/243/58, and a 9-configuration synthetic sweep
found no index rule at all (`{5,200}->200` but `{0,279}->0`). After this PR
the Python fallback and the CUDA `_moe_C` kernel agree on ties; the aiter
kernel keeps its own (deterministic, kernel-defined) tie-break, so a
cross-path inconsistency remains on ROCm-with-aiter. That divergence is
pre-existing and not created by this patch; fixing it means teaching the
aiter kernel the same rule, which is out of scope here.

`fused_topk_bias_router.py` (L330-333) carries the same
`use_sorted = envs.VLLM_BATCH_INVARIANT` pattern; it is reachable only via
the invalid-grouping fallback path and is left to a follow-up so this PR
stays minimal and reviewable.

## Test plan

- New regression tests in `tests/kernels/moe/test_grouped_topk.py`
  (staged copy: they are **not CUDA-gated**, unlike the existing tests in
  that file, because the Python fallback is the code under test — it must
  also run on ROCm and CPU CI):

  ```
  pytest tests/kernels/moe/test_grouped_topk.py -k "determinism or descending or tie_broken"
  ```

  - `test_grouped_topk_repeat_determinism`: 20 identical calls must return
    bitwise-identical `topk_ids` and `topk_weights`. No tie required - this is
    defect 1, the ordering one. Parametrized over sigmoid/softmax,
    bias/no-bias, and 1-group/8-group.
  - `test_grouped_topk_returns_value_descending_order`: the selected experts
    come back in descending score order, and the selected *set* equals
    `topk(..., sorted=True)`'s. This is a single-call assertion - no
    repetition, so it cannot be flaky - and it is the one that pins the
    fallback to the kernel's existing contract.
  - `test_grouped_topk_tie_broken_by_lower_expert_index`: a deliberate,
    bitwise-exact tie at the k-boundary (two experts with identical logit
    bits, so the tie survives any elementwise rounding, eager or compiled) is
    won by the ascending expert index.
  - `test_grouped_topk_group_tie_broken_by_lower_group_index`: a
    bitwise-exact tie at the *group* boundary (`topk_group > 1`) is broken by
    ascending group index, covering the group-selection site at L135.

  **These fail on `main` and pass with this PR**: 16 of 18 cases fail before
  the change, all 18 pass after, same GPU and same command. The 2 that
  already passed are the group-tie pair, whose top-k is over
  `num_expert_group` = 4 values and therefore never crosses the 256-column
  threshold; they are kept as contract tests for the L135 site.

  The tests use **288 experts on purpose**. Per the table above, `topk`'s
  ordering only degrades past 256 columns, so the same tests written at
  DeepSeek-V3's 256 - or at the 32 experts the existing
  `test_grouped_topk_single_group_stable_ties` uses - pass against the
  unfixed code and demonstrate nothing. This was checked the wrong way round
  first: an earlier draft at 32 experts passed on stock and had to be
  rewritten.
- End-to-end on the affected deployment (GLM-5.3, 2x TP, gfx1151, greedy,
  740-token prefill, repeated N times, comparing first-token and full
  completion bit-identity before/after):

  Validated on the affected deployment (GLM-5.3-Flash AWQ-W4A16, TP=2 over
  2x gfx1151, ROCm 10.0, torch 2.11, `--enforce-eager`, speculative decoding
  off, prefix caching off, engine warmed). 5 byte-identical greedy requests
  per prompt length, comparing full completions:

  | prompt tokens | before | after |
  |---:|---:|---:|
  | 244 | 5 distinct / 5 | **1 / 5** |
  | 614 | 5 distinct / 5 | **1 / 5** |
  | 1196 | 3 distinct / 5 | **1 / 5** |
  | 1421 | 5 distinct / 5 | **1 / 5** |
  | 1797 | 5 distinct / 5 | **1 / 5** |

  Per-layer verification: with a hash tap after every one of the 45 decoder
  layers, all layers are bit-identical across repeated forwards **on both TP
  ranks** at 740 prompt tokens (they were not before). Op-level, the patched
  `grouped_topk` gives 1 distinct result in 30 identical calls where a stock
  control in the same process gives 30 — so the test is sensitive, not merely
  passing.

  No regressions: 4 chat prompts return correct, coherent answers; tool
  calling still yields `finish_reason: tool_calls` with valid-JSON arguments,
  streaming and non-streaming; prefill throughput 340 / 328 / 317 tok/s at
  2K / 8K / 32K prompts, inside the 284-334 tok/s pre-patch baseline.

  **Scope of the claim — this fixes the router, not the platform.** Above
  `index_topk` (2048 for this checkpoint) a *separate* defect remains, and this
  PR does not address it: the DSA sparse-attention indexer's own top-k. A
  six-point tap inside the first sparse-attention layer, both ranks, one boot,
  isolates it cleanly — at 719 prompt tokens all six points are deterministic;
  at 2497 the divergence enters at the attention output
  (`self.self_attn(...)`, 6 of 6 distinct) with its input bit-identical, and
  everything downstream merely cascades. That is #54521's original diagnosis
  and is being fixed in #55122. So: greedy decoding becomes reproducible below
  `index_topk`, and the remaining nondeterminism above it is tracked elsewhere.
  Other nondeterministic ops are tracked in #42259.

## Cross-references

- #54521 — the greedy nondeterminism thread this was diagnosed in (the
  gfx1151 GLM-5.3 arm); the router tie is the missing explanation for the
  layer-21/34 divergence there.
- #42259 — determinism/logits-semantics tracker; a row for router expert
  selection would belong in its table (it currently tracks MoE *combine*
  #45683 and sampler top-k ties #50979, but not router selection).
- #55122 — the same value-desc/index-asc convention being fixed for the
  sparse indexer's `persistent_topk`; same bug class, different op.

---

**DCO / attribution**

```
Signed-off-by: David Canar <davidcanar@gmail.com>
```

This PR was prepared with AI assistance (analysis of the nondeterminism,
the fix, the tests and this description); every line has been reviewed and
validated by the human submitter, per the AI Assisted Contributions policy.

```
Co-authored-by: Claude
```
