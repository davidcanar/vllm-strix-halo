# RESULT — PR package for the deterministic-router fix (upstream test, repo patch, PR body)

**Date:** 2026-09-05 · **Task:** package the deterministic-router fix for
upstream and for this repo, per `FIXSPEC.md`. **Status: COMPLETE.** All three
artifacts written; patch verified against the stock file; Dockerfile change
handed over, not applied; deployment untouched.

## 1. Artifacts produced (the only files I wrote)

| artifact | path | md5 |
|---|---|---|
| upstream unit test | `~/vllm-strix-halo/upstream/test_grouped_topk_determinism.py` | `decc6a1c4d763c5bcc9b73164e6aacac` |
| repo patch | `~/vllm-strix-halo/container/patches/vsh-moe-router-deterministic-topk.patch` | `5648b6b7ee96f43ebe5f52166fd516de` |
| PR body | `~/vllm-strix-halo/upstream/PR-BODY.md` | `eb1037029dae03db5183b2e16ebfcbf0` |

(`~/vllm-strix-halo/upstream/` was created. Scratch work in `/tmp` and
`/tmp/vsh-patchgen/` on box1 only.)

## 2. Patch: content, format, verification

**Content = FIXSPEC verbatim.** Adds the module-level
`_deterministic_topk_indices` helper (docstring byte-identical to FIXSPEC)
and replaces the three selection sites: group selection (L135), expert
selection in the bias branch (L148), and the no-bias branch (L152-154, now
`_deterministic_topk_indices` + `tmp_scores.gather`). The `use_sorted` local
and its L133 comment are removed. The `envs` import is **kept** — it is
still used by the fused-kernel gate at L92 (`VLLM_USE_FUSED_MOE_GROUPED_TOPK`;
verified). L126 (`view(...).topk(2)[0].sum(-1)`) untouched per FIXSPEC.

**Format matches the existing five patches**: `diff --git a/<path> b/<path>`
header, `--- a/` / `+++ b/` labels, plain `@@` hunks, no `index` line (same
as `vsh-aiter-gfx1151-gate.patch`), paths rooted at site-packages so the
Dockerfile's `git apply -p1 --whitespace=nowarn` resolves them.

**Verification** — against a copy of the stock file pulled out of the
container to `/tmp` (nothing patched in place):

- stock md5 `da7b32ec80cd79291a2b58e3df8ccbbe` — identical to the stock md5
  recorded in `HANDOFF-tie-result.md` §6 and `HANDOFF-fixdesign-result.md` §10.
- `patch --dry-run -p1` in a scratch tree:
  `checking file vllm/model_executor/layers/fused_moe/router/grouped_topk_router.py`
  → exit **0**.
- `git apply --check -p1 --whitespace=nowarn` (the Dockerfile's exact
  invocation) → exit **0**.
- a **real** `git apply -p1` into the scratch tree produced a file
  **byte-identical** to the intended patched file
  (both md5 `f2a575d1581024ab44a539538281561a`, 357 lines from 349).

**Cross-check with the implementing agent.** The live in-container file (now
md5 `3f08fb73...`, mid-validation by the other agent — I only read it) is
**functionally identical** to my patched file: same helper text, same three
call sites, same removals. Two cosmetic differences only: (a) they place the
helper after the imports, I place it directly above its only consumer
`grouped_topk`; (b) their `group_idx` line carries a trailing comment making
it 90 chars — over vLLM's 88-char ruff limit — mine keeps the comment on its
own line. Their GPU validation therefore transfers 1:1 to this patch.

## 3. Dockerfile change required — NOT applied (deliberately)

Per instructions the edit is handed over for human review rather than
applied. `container/Dockerfile` lines 159-171; add the new patch to the
COPY, `git apply`, and `rm` lists:

```diff
--- a/container/Dockerfile
+++ b/container/Dockerfile
@@ -159,13 +159,15 @@
 COPY container/patches/vsh-rdma-allreduce.patch \
      container/patches/vsh-mhc-no-tilelang-gfx1151.patch \
      container/patches/vsh-aiter-gfx1151-gate.patch \
      container/patches/vsh-fp8-fnuz-mqa.patch \
-     container/patches/vsh-mtp-ropefree-triton-sparse.patch /tmp/
+     container/patches/vsh-mtp-ropefree-triton-sparse.patch \
+     container/patches/vsh-moe-router-deterministic-topk.patch /tmp/
 RUN cd /opt/venv/lib/python3.12/site-packages && \
     git apply -p1 --whitespace=nowarn \
       /tmp/vsh-rdma-allreduce.patch /tmp/vsh-mhc-no-tilelang-gfx1151.patch \
       /tmp/vsh-aiter-gfx1151-gate.patch /tmp/vsh-fp8-fnuz-mqa.patch \
-      /tmp/vsh-mtp-ropefree-triton-sparse.patch && \
+      /tmp/vsh-mtp-ropefree-triton-sparse.patch \
+      /tmp/vsh-moe-router-deterministic-topk.patch && \
     rm /tmp/vsh-rdma-allreduce.patch /tmp/vsh-mhc-no-tilelang-gfx1151.patch \
       /tmp/vsh-aiter-gfx1151-gate.patch /tmp/vsh-fp8-fnuz-mqa.patch \
-      /tmp/vsh-mtp-ropefree-triton-sparse.patch
+      /tmp/vsh-mtp-ropefree-triton-sparse.patch \
+      /tmp/vsh-moe-router-deterministic-topk.patch
```

Why not applied: the task explicitly hands this over for human review, and
the repo convention is that every patch is "Reviewed in PATCHES.md" — a
PATCHES.md entry should accompany the Dockerfile edit (not written here; I
was scoped to the named files only). The stale comment above the block
(lines 156-157, which mentions only the first two patches) was left alone —
it is already stale for the other three.

## 4. `fused_topk_bias_router.py` — NOT included in the patch

Reasoning (FIXSPEC allows either, provided the choice is stated):

1. **Unreachable for this model.** It is only called when
   `valid_grouping()` is False (stock file L296-324); GLM-5.3 has 288
   experts, `num_expert_group=1`, `288 % 1 == 0` → grouping valid → the
   `grouped_topk` path runs. Including it would change code the deployment
   can never execute.
2. **Validation scope.** The other agent is validating exactly the
   three-site `grouped_topk_router.py` change on the live cluster; shipping
   a second, unvalidated file in the image patch would widen the blast
   radius of a mid-restart-cycle change.
3. **The mirror site is not a pure copy**: it also contains a `torch.topk`
   without `sorted=` in the `bias_vl`/hash-table path (L325) that would need
   its own tie analysis.
4. **Upstream symmetry.** The PR body likewise scopes it out as a follow-up,
   so the repo patch and the upstream PR stay in sync.

## 5. PR body — provenance, gaps, placeholders

Everything in `PR-BODY.md` is sourced from `FIXSPEC.md`,
`HANDOFF-upstream-scan-result.md`, `HANDOFF-fixdesign-result.md`,
`HANDOFF-tie-result.md`, the stock source, upstream
`tests/kernels/moe/test_grouped_topk.py`, and
`docs/contributing/README.md` (fetched read-only via `gh` today for the exact
AI-trailer wording: `Co-authored-by: Claude`, and the DCO
`Signed-off-by:` requirement). Nothing was invented. Caveats baked in:

- The fused-kernel code quotes (`moeTopKFuncs.cuh` `makeCmpVal`,
  `grouped_topk_kernels.cu` `is_better_than`) are reproduced from the scan
  report, which read them at **upstream main**, not at `8bf39632` (the wheel
  ships no csrc) — the scan's own caveat. Line numbers for the router file
  were verified against the installed stock file.
- Source discrepancy disclosed rather than smoothed over: FIXSPEC says
  `sorted=False` permutes order on **740/740** rows; fixdesign §E measured
  **≥715/740** on other tensors. The PR states both ("740/740 ... a 20-call
  follow-up on other layers measured ≥715/740").
- Perf numbers are fixdesign §D verbatim, with its caveats (gfx1151,
  contention-inflated absolutes, data-dependence, O(E log E) vs O(E log k))
  and the ~2.4 MiB transient memory note.
- The AITER tie-break examples are the measured ones from fixdesign §C; the
  PR calls the rule "opaque" (kernel source unread) rather than claiming a
  reverse-index rule the synthetic sweep disproved.

**Two placeholders left, both clearly marked:**

1. `<<END-TO-END RESULT PENDING>>` in the Test plan — end-to-end numbers come
   from the other agent; the body explicitly warns not to claim the PR fixes
   platform-wide nondeterminism until they land.
2. `Signed-off-by: <<REAL NAME>> <<email@example.com>>` — DCO; fill via
   `git commit -s`.

## 6. The upstream test — what it does, what was pre-validated

`upstream/test_grouped_topk_determinism.py` is intended to be appended to
`tests/kernels/moe/test_grouped_topk.py` (imports duplicate that file's; the
docstring says so) and also runs standalone. Per requirements it is **not
CUDA-gated** — the file's other tests all use
`skipif(not current_platform.is_cuda(), ...)` because they call the fused
kernel; these force `VLLM_USE_FUSED_MOE_GROUPED_TOPK=0` and run the Python
`grouped_topk` on CUDA/ROCm/CPU alike (`_DEVICE` picks `cuda` if available,
else `cpu`). The tie is **constructed, not chanced**: two experts get
identical logit bits (elementwise activation ⇒ bitwise-equal biased scores on
any backend, eager or compiled) at positions k-1/k, with the construction
self-validated in-test (exactly k-1 strictly above, exactly 2 on the tie).

- `test_grouped_topk_single_group_deterministic_ties` — (a) 20 calls with
  fresh-cloned inputs give bitwise-identical ids and weights (int32-view
  compare), (b) the boundary tie is won by the **lower** expert index, the
  full selection is `arange(k)`, (c) order is value-descending; parametrized
  sigmoid/softmax × bias/no-bias (covers both changed branches).
- `test_grouped_topk_single_group_tie_free_control` — (d) with the tie
  broken, the result is exactly `topk(..., sorted=True)`'s: no behavioural
  change on tie-free input.
- `test_grouped_topk_multi_group_deterministic_group_tie` — bitwise-equal
  group scores at the `topk_group` boundary; asserts the lower group index
  wins (guards the L135 change, which no other test covers). Losing arm
  verified to yield [0,1,16,17] vs the asserted [0,1,8,9].

**Not run** (per instructions — the other agent validates on GPU). Pre-GPU
checks I did run: `py_compile` clean; and a **CPU-only** construction check
inside the container (pure torch, no vllm import, no GPU) validating every
precondition and expected selection. That check caught and fixed two flaws
in my first draft (group-0's 3rd/4th experts outranked the tie groups in the
multi-group case; a miscalibrated margin guard). Smallest deliberate value
gap in the constructions: 5.8e-4 ≈ 5000 ULP at fp32 ~0.95 — safe against the
measured 1-ULP eager-vs-compiled wobble. Note for whoever appends it
upstream: run `ruff format` first (all lines are <88 chars but a few are
wrapped more conservatively than ruff would).

## 7. Deployment untouched — confirmation

- No writes to anything inside the containers, no server restart/traffic, no
  `~/vsh-config.yaml` access, no GPU work. Container use was read-only
  (`cat`/`md5sum` of source) plus one transient CPU-only python process for
  the construction check.
- The in-container `grouped_topk_router.py` md5 change
  (`da7b32...` → `3f08fb73...`) observed mid-task is the **other agent's**
  in-place implementation, not mine (mine lives only in `/tmp` scratch and
  the repo patch file).
- No `git commit`/`push`/branch; nothing posted to GitHub (`gh` used
  read-only, three GETs). `git status` in `~/vllm-strix-halo` shows my two
  untracked additions (`container/patches/vsh-moe-router-deterministic-topk.patch`,
  `upstream/`) plus this handoff; the pre-existing deliberate modifications
  (`PATCHES.md`, `README.md`, `vllm-strix-halo.sh`) and all `HANDOFF-*` /
  `FIXSPEC.md` / `PENDINGWORK.md` were left byte-for-byte alone.
