# RESULT — upstream scan: MoE router selection determinism in vLLM (pre-PR recon)

**Date:** 2026-09-05 · **Task:** scan vLLM upstream for existing/in-flight work on
MoE router selection determinism (`grouped_topk` tie nondeterminism), before
writing our patch. Read-only; nothing posted. **Status: COMPLETE.**

---

## 1. Bottom line (read this first)

1. **Our fix is NOT a duplicate.** No open or closed issue/PR anywhere in
   vllm-project/vllm addresses tie nondeterminism of the MoE *router's* expert
   selection (`torch.topk` in the Python `grouped_topk`). Everything nearby
   is either the sparse-attention *indexer* (`persistent_topk`), the MoE
   *combine/finalize*, the *sampler* top-k, or `moe_align_block_size`.
2. **It stands alone — attach by cross-reference, not by nesting.** Post it as
   its own bugfix PR; link it from/to #54521 (the thread we contribute to —
   our gfx1151 GLM-5.3 data is directly relevant there and has NOT yet been
   posted), and ask for it to be added to the determinism tracker #42259's
   table (which already lists MoE-combine determinism #45683 but nothing on
   router selection). #55122's author (jschmied) is active in #54521 and is
   the natural reviewer ally — same bug class, different op.
3. **There is NO existing deterministic-top-k Python helper to reuse.**
   #55122 ("Make persistent_topk deterministic") is a pure CUDA-kernel PR
   (csrc only, no Python API surface); our patch cannot be "use it here too".
   We must implement the tie-break in the Python path ourselves.
4. **The tie-break to implement is: lower expert index wins** (value desc,
   index asc). This is the defined, tested contract of the NVIDIA fused
   kernel our Python path should match — quoted in §5 — and also the output
   contract #55122 chose for `persistent_topk`. Do not invent an order.
5. **Provenance (§4):** `use_sorted` was introduced by PR #26609 (Meta,
   2025-10-16) which *changed a hardcoded `sorted=False`* to env-gated; no
   maintainer ever stated the gating rationale on the record — the PR simply
   left the default path byte-identical and gated every determinism choice on
   the (NVIDIA-SM90-only) `VLLM_BATCH_INVARIANT` flag. Expect a reviewer to
   ask "why not always on / what's the perf cost" — that question has no
   recorded answer upstream, so we must answer it ourselves (see §4).

---

## 2. Duplicate / near-duplicate table

Searched issues AND PRs, open and closed (full query list in §7). Relevant hits:

| # | Type | Title (short) | State | Covers MoE router? | Overlap with us |
|---|------|---------------|-------|--------------------|-----------------|
| 55122 | PR | [Kernel] Make persistent_topk deterministic | OPEN | **No** — sparse-attention indexer top-k (QSA block selection), `csrc/libtorch_stable/persistent_topk.cuh` only | None code-wise. Same bug class (kernel top-k tie/atomic slot race). **No shared helper introduced.** Its output contract (value desc, index asc) is the convention to match. Fixes #54521 + #51782 |
| 54521 | issue | Qwen3.8-Flash-Next greedy non-deterministic from persistent_topk (sm121/GB10) | OPEN | No — indexer; **but this is the thread we (davidcanar) contribute to**, and its gfx1151 side is our GLM-5.3 prefill divergence | High contextually: our router-tie finding is the missing explanation for the ROCm arm of that thread; not posted there yet (thread's last comments 2026-09-05 still bisecting layer 3 post-attention, o_proj→mHC→MoE) |
| 51782 | issue | persistent_topk silently drops candidates in coarse histogram bins | OPEN | No — indexer | None (fixed by #55122) |
| 54945 | issue | FlashInfer CUTLASS NVFP4 MoE gives different logits (fused finalize) | OPEN | No — MoE **finalize** (weighted combine after experts), not selection | Near-duplicate in "MoE nondeterminism" theme only. Workaround PR #54948 adds env to disable fused finalize |
| 45683 | PR | Deterministic MoE combine (reduce_scatterv) under VLLM_BATCH_INVARIANT | MERGED | No — MoE combine under DP+EP | Theme overlap only; listed in #42259's determinism table |
| 48032 | PR | Make Marlin MoE route alignment deterministic | OPEN | No — `moe_align_block_size` (post-selection token→expert layout) | Theme overlap only |
| 50979 | PR | [Sampler] Keep exactly k tokens in top-k with tied logits | OPEN | No — **sampler** top-k (`topk_topp_sampler`) | Same torch.topk-tie bug class, different path. Its PR writeup (incl. a "why not a duplicate" section) is a good model for ours |
| 53620 | issue | test_spec_decode_logprobs ill-defined at top-k ties | OPEN | No — sampler/tests | Same tie-semantics theme |
| 42259 | issue | [RFC] Logprobs/Logits Semantics and Determinism Across the vLLM Ecosystem | OPEN | Tracker; has a determinism/batch-invariance table | **The place to get our item tracked.** Table currently covers MoE combine (#45683), all-reduce, softmax, conv, spec-decode — nothing on router expert selection |
| 27433 | issue | Batch Invariant Feature and Performance Optimization (tracker) | OPEN | Umbrella for the VLLM_BATCH_INVARIANT effort incl. #26609 | Related (our line's birthplace) but our fix is not batch-invariance-specific; AMD support is only a "nice to have" there (#52231) |
| 52231 | PR | [Feature] Support batch invariance on ROCm | OPEN | No — does NOT touch `grouped_topk_router.py` (collectives, MX linear/MoE GEMMs, tests) | Would be the vehicle if we framed this as "ROCm batch invariance"; not needed for a standalone fix |
| 55131 | issue | Batch-invariant matmul not actually batch-invariant (TF32) | OPEN | Mentions "the MoE router" as a precision-sensitive consumer of the batch-invariant **GEMM** | Different op (the linear feeding router logits), not topk |
| 47069 | issue | Concurrent temp=0 non-determinism, FLASH_ATTN batch-invariant path, Hopper | OPEN | No — attention | Family overlap |
| 53257 | issue | DeepSeek-V4-Flash non-deterministic temp=0, scales with concurrency | OPEN | No — multi-cause (indexer/finalize/collectives) | Family overlap |
| 53436 | issue | Run-to-run perf non-determinism, spec decode, DeepSeek-V4-Flash | OPEN | No — target forward not bit-reproducible; perf symptom | Family overlap |
| 34206 | PR | [Kernel] Optimize grouped topk kernel | MERGED | Yes — but the **fused** kernel | Not a duplicate of the bug: this is where the fused kernel's deterministic tie contract + its test landed (see §5) |
| 28986 | issue | Fused kernel for GPT-OSS router | CLOSED | Router, but perf fusion request | None |
| 26609 | PR | Deepseek-v3 Batch Invariant on 8xH100 | MERGED | Yes — introduced `use_sorted` | **Provenance of our line** (§4), not a competing fix |

Old grouped-topk fixes checked and ruled out as duplicates: #24146/#24145
(2024 correctness fix), #34673 (no-expert-group models), #31781 (bias dtype),
#29575/#30623 (refactors), #49618 (fused-path dispatch perf), #53580 (XPU).

---

## 3. Umbrella analysis — attach or stand alone?

- **#42259 (RFC logprobs/logits determinism tracker).** Scope: MRV2
  logprobs/sampling semantics + a determinism/batch-invariance bug table.
  It already tracks MoE *combine* determinism (#45683) and sampler top-k ties
  (#50979) but has **no row for router expert selection**. Recommendation:
  standalone PR + ask for a row in #42259's table (that issue explicitly says
  "Keep this issue as the working tracker for open items only", so it wants
  exactly this kind of entry).
- **#55122 (persistent_topk deterministic).** Read carefully as instructed.
  Files: only `csrc/libtorch_stable/persistent_topk.cuh` (+290/-243),
  `csrc/libtorch_stable/topk.cu` (launcher), `tests/kernels/test_top_k_per_row.py`.
  It does **not** introduce a shared deterministic-top-k helper callable from
  Python — everything is inside the indexer kernel. Its relevance to us:
  (a) establishes the repo-wide tie convention — "Output contract is now
  **ascending index order**, identical across calls; equal to top-k by
  value desc, index asc"; (b) shows the acceptable shape of a determinism
  PR here (op-level contract + tie-heavy tests + perf table + independent
  validation); (c) its author is already deep in our thread (#54521).
  Our patch is **not** "use this helper" — there is nothing to call.
- **#27433 (batch-invariance umbrella)** + **#52231 (ROCm batch invariance)**:
  our bug exists with the flag OFF; the fix is not batch-invariance-gated.
  Framing it under the batch-invariant effort would be wrong (and
  `VLLM_BATCH_INVARIANT` is documented "Requires NVIDIA GPU with compute
  capability >= 9.0", `vllm/envs.py` L629-631, so ROCm users cannot even opt
  into the current `sorted=True` mitigation in a supported way).
- **Verdict: standalone bugfix PR**, cross-linked to #54521 and #42259,
  matching #55122's tie convention.

---

## 4. Provenance of `use_sorted = envs.VLLM_BATCH_INVARIANT`

Chain (verified from commit history and diffs):

1. **Introduced by PR #26609 — "Deepseek-v3 Batch Invariant on 8xH100"**,
   merged 2025-10-16, author **bwasti (Bram Wasti, Meta)**, co-authored by
   **yewentao256** (vLLM maintainer). In `vllm/model_executor/layers/fused_moe/fused_moe.py`
   it changed, in BOTH `fused_topk_bias` and `grouped_topk`:

   ```diff
   -    topk_indices = torch.topk(scores_for_choice, k=topk, dim=-1, sorted=False)[1]
   +
   +    # For batch invariance, use sorted=True to ensure deterministic expert selection
   +    use_sorted = vllm_kernel_override_batch_invariant()
   +    topk_indices = torch.topk(scores_for_choice, k=topk, dim=-1, sorted=use_sorted)[1]
   ```

   i.e. **the code was hardcoded `sorted=False` before; #26609 made the
   deterministic option opt-in.** (Note the PR's premise is itself imperfect:
   `sorted=True` does NOT resolve ties deterministically — exactly our bug.)
2. Predecessors: #26136 (WIP, closed) and #24583 (PoC, closed); framework PR
   #25603; umbrella #27433 ("Deepseek-v3 ✓ #26609"). Source of the approach:
   thinking-machines-lab batch_invariant_ops.
3. Code moved into `vllm/model_executor/layers/fused_moe/router/grouped_topk_router.py`
   by **#30623** (router OO refactor, merged 2026-01-18) — the comment and
   gating carried over verbatim.
4. Read renamed `vllm_is_batch_invariant()` → `envs.VLLM_BATCH_INVARIANT` by
   **#35007** (merged 2026-03-23; mechanical rename, no semantic change).

**Why gated rather than always-on: no maintainer statement exists.** I read
the PR body, all 11 issue comments and all 13 review comments of #26609 —
none discusses the topk `sorted` choice. The gating rationale is implicit in
the PR's pattern: every determinism-preserving change (kernel config pinning,
`num_kv_splits = 1 if ... else 4` — that exact pattern was reviewer
yewentao256's own suggestion, custom-AR disable, etc.) is gated on the flag so
the default path stays byte-identical, and batch-invariance is treated as an
opt-in mode with a perf cost (their test gated to SM90+). So a reviewer's
likely objections to "always deterministic" are: (a) perf of sorting — weak,
since #55122 measured exact `torch.topk` fallback at only −10–13% prefill
throughput for the far-larger indexer top-k, and our k=8-of-288 sort is
trivial; (b) changing default numerical behavior for existing users — the
real objection; our answer is that only tie *ordering* changes, and only on
bitwise-equal scores where current behavior is a race. Bring numbers.

**Other files with the same pattern (fix them in the same PR or note them):**
`fused_topk_bias_router.py` L313/L330-333 has the identical
`use_sorted = envs.VLLM_BATCH_INVARIANT` + `torch.topk(..., sorted=use_sorted)`
(and L325: a `torch.topk` without `sorted=` for the bias_vl/hash-table path).

---

## 5. The fused kernel's tie-break, and the fused-vs-Python branch conditions

### 5.1 Fused kernel tie-break: YES — defined and deterministic, lower index wins

`ops.grouped_topk` → `torch.ops._moe_C.grouped_topk` (host fn in
`csrc/libtorch_stable/moe/grouped_topk_kernels.cu`, quoted at upstream main;
the wheel ships no csrc so this was read from main, not from 8bf39632):

- **Single-group path (n_group==1 && topk_group==1 — exactly GLM-5.3's
  shape):** `invokeNoAuxTc` tries `single_group_topk::invoke` first; tiers
  include `SigmoidBiasTiers = ... Tier<384, 8> ...` so 288 experts/topk 8
  matches. Both `single_group_topk_block_kernel` and
  `single_group_topk_warp_kernel` select via
  `reduce_topk::HighExpertLaneOwnedTopKRange` from
  `csrc/libtorch_stable/moe/moeTopKFuncs.cuh` (vendored from TRT-LLM /
  FlashInfer `RoutingKernelTopK.cuh`), which packs value and index into one
  comparison key:

  ```cpp
  // moeTopKFuncs.cuh, TopKRedType::makeCmpVal (lines ~50-62)
  static __host__ __device__ inline TypeCmp makeCmpVal(T val, int32_t idx = 0) {
      auto valueBits = cub::Traits<T>::TwiddleIn(...);
      TypeCmp compactTmp = valueBits;
      compactTmp = (compactTmp << kMoveBits) | (0xFFFF & (kMaxIdx - idx));
      // Use 65535 minus idx to give higher priority to elements with smaller
      // indices.
      return compactTmp;
  }
  ```

  On bitwise-equal values the packed key is decided by `(65535 - idx)`, i.e.
  **smaller expert index wins**, deterministically (max-reduce over the packed
  key; no atomics, no arrival order). (`kMaxIdx = 65535` — indices must be
  < 65536; fine for ≤ 1024 experts.)
- **Multi-group path** `grouped_topk_fused_kernel` (and
  `grouped_topk_fused_small_expert_count_kernel`): group and expert selection
  both use `WarpSelect<..., greater=true, ..., is_stable=true>` (lines
  ~547-549 and ~577-579), where stability is implemented by:

  ```cpp
  // grouped_topk_kernels.cu lines ~75-84
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

  → ties broken by **lower index**, in both sort directions.
- **This contract is already tested**: `tests/kernels/moe/test_grouped_topk.py`
  `test_grouped_topk_single_group_stable_ties` (line 297): all-zero logits and
  bias, topk 16 → asserts `expected_ids = torch.arange(16)` (ascending) and
  equal weights. Landed with #34206 (fused-kernel optimization, merged
  2026-02-20).
- **Consequence for patch design:** our Python-path patch must select the
  same set and order — value desc, index asc on exact ties — so that the
  Python (ROCm) path and the fused (CUDA) path agree bitwise, as the main
  test already asserts (`assert_close(..., atol=0)` on ids, with
  `VLLM_BATCH_INVARIANT=True` forced). Also matches #55122's stated contract
  ("ascending index order ... top-k by value desc, index asc").
- **The ROCm AITER path** (`rocm_aiter_ops.biased_grouped_topk`,
  `rocm_aiter_moe.py` L206-224) lives in AMD's aiter repo — I did not read
  that kernel; unknown whether it shares the tie contract. Note this gap in
  the PR if relevant.

### 5.2 Fused vs Python fallback branch conditions (with line numbers)

File `vllm/model_executor/layers/fused_moe/router/grouped_topk_router.py`
— **identical line numbers in upstream main and our installed tree**
(`0.29.0.dev0+git.8bf39632`, verified by diff of the relevant regions):

1. **Router entry** `GroupedTopKRouter._compute_routing` (L286):
   - L296-304 `valid_grouping()`: `num_experts > num_expert_group` and
     divisible; else falls to `fused_topk_bias`/`fused_topk` (L304-324).
     GLM-5.3: 288 experts, n_group=1 → valid → grouped path.
   - **L327** `if rocm_aiter_ops.is_fused_moe_enabled():` → L330
     `rocm_aiter_grouped_topk`; **else L335** the Python `grouped_topk`.
     `is_fused_moe_enabled()` = `VLLM_ROCM_USE_AITER && VLLM_ROCM_USE_AITER_MOE`
     (`_aiter_ops.py` L1903-1904, flags at L1756/L1759; defaults False/True).
     Our deployment: `VLLM_ROCM_USE_AITER=1, VLLM_ROCM_USE_AITER_MOE=0` →
     Python path.
2. **Inside Python `grouped_topk`** (L80, `@torch.compile`-decorated L75-79):
   **L91-97 fused-CUDA gate:**

   ```python
   if (
       envs.VLLM_USE_FUSED_MOE_GROUPED_TOPK      # default True (envs.py L209)
       and current_platform.is_cuda()            # False on ROCm (verified in-container)
       and num_expert_group <= 32
       and topk <= 32
       and e_score_correction_bias is not None
   ):
       return fused_grouped_topk(...)            # L98 → ops.grouped_topk (L43/L57)
   ```

   → on ROCm `is_cuda()` is False so the fused kernel is NEVER taken from
   here (`_custom_ops.py` L2546-2549 also raises NotImplementedError off-CUDA).
   **Confirmed: the Python `torch.topk` path is ROCm-default** (whenever aiter
   MoE is off), which is why NVIDIA users don't hit our bug.
3. **The nondeterministic lines:** L133-134 the comment + `use_sorted =
   envs.VLLM_BATCH_INVARIANT`; L135 `group_idx = torch.topk(group_scores,
   k=topk_group, ..., sorted=use_sorted)`; **L148**
   `topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=use_sorted)[1]`;
   L152-154 the no-bias branch (also `sorted=use_sorted`).
4. Note CUDA users CAN reach the Python path too: `VLLM_USE_FUSED_MOE_GROUPED_TOPK=0`,
   or `e_score_correction_bias is None` (L96 fails → whole gate false; softmax
   grouped models without correction bias take the Python path even on CUDA),
   or shapes outside the tier table (n_group>32, topk>32). So the latent bug
   is not strictly ROCm-only; it is ROCm-*default*.
5. Mirror site: `GroupedTopk` CustomOp `forward_hip` (L219-243) makes the same
   aiter-vs-native choice for the CustomOp-dispatch route.

---

## 6. Contribution mechanics

- **DCO required**: commits must carry `Signed-off-by:` (`git commit -s`);
  docs/contributing/README.md §"DCO and Signed-off-by". No CLA found — DCO
  only.
- **PR template requires** a "Test Plan" and "Test Result" section (the
  checklist enforces it). Bugfix PRs here consistently include a runnable
  pytest command plus before/after evidence.
- **AI-assisted contributions are explicitly allowed** but: human must review
  every line, must not be a "pure agent" PR, and must add
  `Co-authored-by: Claude` style trailers (#55122 does this and passed the
  bot gate).
- **Unit tests for `grouped_topk` already live in**
  `tests/kernels/moe/test_grouped_topk.py` (369 lines): `test_grouped_topk`
  (parametrized shapes incl. (384, 8, 1, 1) and (512, 22, 1, 1); runs the
  Python path with `VLLM_USE_FUSED_MOE_GROUPED_TOPK=0` + patched
  `envs.VLLM_BATCH_INVARIANT=True` and compares ids to the fused kernel at
  atol=0), `test_grouped_topk_single_group_large_batch`, `..._tiers`,
  `..._capacity_tiers`, **`test_grouped_topk_single_group_stable_ties`**,
  `..._nonfinite_scores`. **Caveat: all are `skipif not current_platform.is_cuda()`**
  — a regression test for the ROCm/Python path must not be CUDA-gated (it
  should exercise the Python `grouped_topk` deterministically on any device,
  or run under the AMD CI, which exists — `/amd-ci run`).
- Secondary location for determinism tests: `tests/v1/determinism/`
  (e.g. `test_batch_invariance.py`; #52231 adds
  `test_moe_row_permutation_batch_invariant.py` there). A tight unit test in
  `tests/kernels/moe/test_grouped_topk.py` is the right primary home.
- The one-line test design that would lock our fix: replicate
  `test_grouped_topk_single_group_stable_ties` but through the **Python**
  `grouped_topk` with deliberate bitwise fp32 ties at the k-boundary (e.g.
  duplicate biased scores), asserting ascending-index selection and
  call-to-call bit-identity, on a non-CUDA-gated path.

---

## 7. Queries run, and what I could not determine

### gh queries (all `--repo vllm-project/vllm`; issues unless marked PRs)

Hits summarized above; **0-result queries listed so the negatives are
interpretable**:

Issues: `grouped_topk determinism` (0) · `grouped_topk` (13, none our bug) ·
`topk router deterministic` (0) · `MoE router nondeterminism` (0) ·
`expert selection deterministic` (0) · `torch.topk nondeterministic` (0) ·
`MoE determinism` (0) · `topk tie` (0) · `tie break` / `tie-break` (13;
indexer/sampler/DP-LB) · `topk ties` (0) · `router topk` (0) ·
`deterministic topk` / `deterministic top-k` (→ #54521 only) ·
`VLLM_BATCH_INVARIANT` (20) · `batch invariant` / `batch-invariant` (20) ·
`noaux_tc` (4, unrelated) · `use_sorted` (0) · `topk expert` (2, unrelated) ·
`routed experts deterministic` (0) · `moe_align_block_size determinism` (0) ·
`sorted topk` (0) · `top-k nondeterminism` (0) · `noaux_tc determinism` (0) ·
`greedy nondeterministic ROCm` (0) · `expert selection nondeterministic` (0) ·
`router tie` (0) · `MoE greedy different output` (0) · `tied scores experts`
(0) · `gate topk identical` (0) · `same expert set different order` (0) ·
`ROCm nondeterministic greedy` (0) · `MI300X deterministic` (0).

PRs: `persistent_topk deterministic` (→ #55122) · `grouped_topk` (20) ·
`deterministic topk` (0) · `topk tie` (→ #55314 indexer smem, #46145) ·
`router sorted` (→ #42982, unrelated) · `use_sorted` (0) ·
`VLLM_BATCH_INVARIANT` (25) · `batch invariant` / `batch invariance` (25) ·
`MoE topk sorted` (0) · `deterministic MoE` (→ #45683/#48032/#45677/#41988) ·
`vllm_is_batch_invariant` (→ #28304 etc.) · `MoE topk batch invariant` (0) ·
`topk sorted batch invariant` (0) · `deterministic tie break topk` (0) ·
`expert index tie` (0) · `stable topk router` (0) · `grouped_topk tie` (0) ·
`grouped_topk deterministic` (0) · `grouped_topk sorted` (0) ·
`router determinism` (0) · `deterministic routing` (→ #48032, #34559).

Commit search API: `"sorted=True" topk` (0), `use_sorted` (0). Provenance was
traced instead via `commits?path=` file history for
`router/grouped_topk_router.py`, `fused_moe/fused_moe.py`, `topk.py` (old
path: no history — never existed at that path in this repo),
`tests/kernels/moe/test_grouped_topk.py`, plus targeted diffs of #26609 /
#35007 / #34206 / #30623.

Read in full or in part: #42259, #54521 (body + all 29 comments),
#55122 (body + comments), #26609 (body + 11 comments + 13 review comments +
diff), #26136, #24583, #35007, #30623, #34206, #27433, #52231, #55131, #47069,
#53257, #53436, #54945, PR #54948, #53620, #53142, #45683, #48032, #50979,
#28986 (issue), #50682 (grep for topk/determinism: no hits).

### Could not determine / under-claiming

- **Why `sorted` is env-gated:** no maintainer ever wrote the reason down in
  #26609 or anywhere I could find. §4's rationale is inferred from the PR's
  structure, not quoted. Do not present it as an upstream statement.
- **Kernel sources were read at upstream `main`**, not at our installed
  `8bf39632` (the wheel ships no csrc). The router file was verified
  line-identical between main and installed for all quoted regions; the
  csrc files were not diffed against 8bf39632. The tie-break quotes are from
  main as of 2026-09-05.
- **AITER's `biased_grouped_topk`** (the on-by-default ROCm fused path when
  `VLLM_ROCM_USE_AITER_MOE=1`) is in AMD's aiter repo — not read; unknown
  tie behavior. If our PR claims "Python path now matches the fused kernel",
  scope the claim to the CUDA `_moe_C` kernel.
- **`torch.topk` nondeterminism under `sorted=True` is not acknowledged in
  any vLLM issue/PR** — nothing upstream documents that sorted=True is
  insufficient on ties; our PR will be the first statement of it (worth
  citing PyTorch docs' "order of ties is not guaranteed" language).
- In-flight unmerged work beyond what GitHub shows (private branches, Slack
  `#pr-reviews` discussions) is invisible to this scan.
- `gh search` matches title/body/comment text only; a duplicate framed with
  entirely different vocabulary would not surface. The query list above is
  the boundary of what was checked.

---

## 8. Recommended PR shape (summary of the above)

Standalone `[Bugfix][MoE]` PR: make the Python `grouped_topk` (and
`fused_topk_bias`) selection deterministic by breaking bitwise ties on
expert index (lower wins), matching the fused `_moe_C` kernel's tested
contract (`test_grouped_topk_single_group_stable_ties`) and #55122's
convention; include the non-CUDA-gated tie regression test in
`tests/kernels/moe/test_grouped_topk.py`, a perf number for the tie-break
(e.g. via the existing benchmark harness), cross-link #54521 and #42259,
`git commit -s`, and disclose AI assistance. The one open design question a
reviewer will raise — always-on vs gated — has no recorded upstream answer,
so bring the perf delta and the "only tie order changes" argument.
