# FIX SPEC — deterministic expert selection in the MoE router

This is the agreed change. Both the implementation task and the PR-packaging
task work from this spec, so they must match exactly.

## The bug

`vllm/model_executor/layers/fused_moe/router/grouped_topk_router.py`, function
`grouped_topk`, selects experts with `torch.topk(..., sorted=use_sorted)` where
`use_sorted = envs.VLLM_BATCH_INVARIANT` (line 134), i.e. `False` by default.
Two measured consequences on the same real 740x288 fp32 score tensor
(gfx1151, ROCm 10.0, torch 2.11):

| arm | distinct raw results / 20 | distinct selected sets / 20 |
|---|---:|---:|
| `sort(descending=True, stable=True)[:k]` | **1** | **1** |
| `topk(sorted=False)` (today's default) | **20** | 2 |
| `topk(sorted=True)` | 2 | 2 |

1. **`sorted=False` permutes the returned order on 740/740 rows every call**
   (the set is correct; the order is not). Downstream this changes the
   k-summation order, so the MoE output differs every call.
2. **`sorted=True` still flips the selected *set* on exact fp32 ties** — the
   real tensor has k-boundary ties at rows 40 and 549 (experts 12/202 and
   243/250, bitwise-equal biased scores), and the varying row is exactly row
   40. `torch.use_deterministic_algorithms(True)` does **not** help (no raise,
   no warning, still flips).

## The fix

Select with a **total order: value descending, then expert index ascending.**
This is not a new convention — it is what the fused CUDA kernel already does
(`moeTopKFuncs.cuh` packs `65535 - idx` into the comparison key, commented
"Use 65535 minus idx to give higher priority to elements with smaller
indices"; the multi-group path uses `WarpSelect<..., is_stable=true>` whose
`is_better_than` breaks `val == baseline` by `index < baseline_index`; and
`test_grouped_topk_single_group_stable_ties` already asserts it). The Python
fallback simply does not match it. PR #55122 fixes the sparse indexer's top-k
to the same rule.

Add a module-level helper:

```python
def _deterministic_topk_indices(x: torch.Tensor, k: int) -> torch.Tensor:
    """Top-k indices under a total order: value descending, index ascending.

    `torch.topk` guarantees no tie-break, and on some backends is
    nondeterministic on exact ties; with `sorted=False` it additionally
    permutes the returned order run-to-run. Both make expert selection
    irreproducible. A stable descending sort gives the same ordering the fused
    kernel uses (value desc, then lower index first).
    """
    return x.sort(dim=-1, descending=True, stable=True).indices[..., :k]
```

Replace the three selection sites (line numbers as of `8bf39632`):

- **L135** `group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=use_sorted)[1]`
  → `group_idx = _deterministic_topk_indices(group_scores, topk_group)`
- **L148** `topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=use_sorted)[1]`
  → `topk_ids = _deterministic_topk_indices(tmp_scores, topk)`
- **L152-154** `topk_weights, topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=use_sorted)`
  → `topk_ids = _deterministic_topk_indices(tmp_scores, topk)` followed by
  `topk_weights = tmp_scores.gather(1, topk_ids)`
  (weights must come from `tmp_scores`, matching today's behaviour — note the
  bias branch at L150 gathers from `original_scores` instead; keep that.)

Then remove the now-unused `use_sorted` local and the L133 comment. **Check
whether `envs` is still referenced elsewhere in the file before touching the
import.**

`fused_topk_bias_router.py` has the same `envs.VLLM_BATCH_INVARIANT` gating
around line 330 — apply the same treatment there, or state explicitly why not.

**Do not touch line 126** (`scores.view(...).topk(2, dim=-1)[0].sum(dim=-1)`).
It takes values only, and any tie at its k-boundary is between equal values, so
the sum is unaffected. Leaving it alone keeps the diff minimal.

## Why unconditional rather than env-gated

It is a correctness fix, it matches the fused kernel, and on the real tensors
it is **faster** (71.6 us vs 97.8 us; ~1.1 ms saved per 42-layer prefill), with
~2.4 MiB transient. A reviewer may still prefer a flag; the PR should note the
alternative and that a full sort is O(E log E) vs O(E log k), so on much larger
expert counts the trade could invert.

## Behavioural delta

Confined to exact ties: 1 row in 740 on the real tensors. Ascending-index is
always one of the outcomes `torch.topk` already produced, so no new selection
is introduced — it just pins which one. Verified: row 40 → expert 12 (not 202),
row 549 → 243, layer-34 row 651 → 58.

## Known gap to disclose, not hide

AITER's `biased_grouped_topk` (used when `VLLM_ROCM_USE_AITER_MOE=1`) is
deterministic but its tie-break is **opaque and demonstrably not
ascending-index**. So on ROCm-with-aiter the tie-break will differ from this
Python path. That is a pre-existing cross-path inconsistency this patch does
not create, but the PR must say so.
