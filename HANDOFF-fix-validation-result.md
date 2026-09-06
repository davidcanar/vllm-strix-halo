# RESULT — deterministic-router fix: implemented and validated end to end

**VERDICT: PARTIAL, not full.** The FIXSPEC patch is implemented on both
containers, is deterministic at op level (1/30 vs stock 30/30), and makes
greedy decoding bit-reproducible for prompts up to ~1.4K tokens — but a
**second, pre-existing, length-dependent nondeterminism remains** at >=~2.4K
tokens (first divergence inside layer 3, the first DSA sparse-attention
layer, on both ranks). The router defect the spec targeted is fixed; issue
#54521-style reproducibility at long context is **not** achieved by this
patch alone.

Determinism sweep (`/tmp/det_sweep.py`, temperature=0, seed=1234,
ignore_eos, 64 gen tokens, 5 reps, prefix caching OFF, 2 warmups first;
baseline before the fix was **5/5 distinct at every length**):

| prompt tokens | distinct / 5 (before) | distinct / 5 (after fix) |
|---:|---:|---:|
| 244 | 5 | **1** |
| 614 | 5 | **1** |
| 1196 | 5 | **1** |
| 2373 | 5 | 5 |
| 4718 | 5 | 5 |
| 9705 | 5 | 5 |
| 14782 | 5 | 5 |

---

## 1. What was applied

Both files live at
`/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/fused_moe/router/`
inside the `vllm-glm` container, patched identically on **box1 and box2**
(rank 1). Backups are new `.fixorig` files; the pre-existing `.orig`,
`.taporig`, `.l3orig*`, `.tieorig` backups were not touched. Both files
byte-compile (`python3 -m py_compile`) on both boxes.

| file | md5 patched (box1 == box2) | md5 stock (`.fixorig`) |
|---|---|---|
| `grouped_topk_router.py` | `3f08fb73128e44d8e8a2feda2f17e797` | `da7b32ec80cd79291a2b58e3df8ccbbe` |
| `fused_topk_bias_router.py` | `070952146f981300e5ca5cbc70bb847d` | `4af04bc37e30e3815faede16e92bc9aa` |

Patch applied by `~/fixpatch.py` (exact-match, assert-on-miss, LF-safe,
refuses double-apply). Reproduction scripts left on box1: `~/fixpatch.py`,
`~/opcheck.py`, `/tmp/discrim.py`, `/tmp/quality.py`, `/tmp/bisect2.py`.

### Diff — `grouped_topk_router.py` (exactly the FIXSPEC change)

```diff
--- grouped_topk_router.py.fixorig
+++ grouped_topk_router.py
@@ -25,6 +25,18 @@
 from vllm.platforms import current_platform
 
 
+def _deterministic_topk_indices(x: torch.Tensor, k: int) -> torch.Tensor:
+    """Top-k indices under a total order: value descending, index ascending.
+
+    `torch.topk` guarantees no tie-break, and on some backends is
+    nondeterministic on exact ties; with `sorted=False` it additionally
+    permutes the returned order run-to-run. Both make expert selection
+    irreproducible. A stable descending sort gives the same ordering the fused
+    kernel uses (value desc, then lower index first).
+    """
+    return x.sort(dim=-1, descending=True, stable=True).indices[..., :k]
+
+
 def fused_grouped_topk(
@@ -130,11 +142,7 @@
         )  # [n, n_group]
 
-    # For batch invariance, use sorted=True to ensure deterministic expert selection
-    use_sorted = envs.VLLM_BATCH_INVARIANT
-    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=use_sorted)[
-        1
-    ]  # [n, top_k_group]
+    group_idx = _deterministic_topk_indices(group_scores, topk_group)  # [n, top_k_group]
     group_mask = torch.zeros_like(group_scores)  # [n, n_group]
@@ -145,13 +153,12 @@
     tmp_scores = scores.masked_fill(~score_mask.bool(), float("-inf"))  # [n, e]
 
     if e_score_correction_bias is not None:
-        topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=use_sorted)[1]
+        topk_ids = _deterministic_topk_indices(tmp_scores, topk)
         # Use original unbiased scores for the routing weights
         topk_weights = original_scores.gather(1, topk_ids)
     else:
-        topk_weights, topk_ids = torch.topk(
-            tmp_scores, k=topk, dim=-1, sorted=use_sorted
-        )
+        topk_ids = _deterministic_topk_indices(tmp_scores, topk)
+        topk_weights = tmp_scores.gather(1, topk_ids)
```

Line 126 (`group_scores = ...topk(2,...)[0].sum(-1)`) was NOT touched, per
spec. The `envs` import stays — line 92 still uses
`envs.VLLM_USE_FUSED_MOE_GROUPED_TOPK`.

### Diff — `fused_topk_bias_router.py`

Decision: **the same treatment was applied**, because the reasoning is
identical — the site selects experts via
`torch.topk(..., sorted=envs.VLLM_BATCH_INVARIANT)`, i.e. the same
order-permuting, tie-flipping call. The `vl_indices = torch.topk(...)[1]`
at line ~325 was left alone: it is on the `bias_vl` image-sentinel path
(unreachable for this text-only deployment) and uses the default
`sorted=True`, so it has no order-permutation problem — same
minimal-diff philosophy as leaving line 126 alone. `envs` import stays
(line 24 uses `envs.VLLM_MOE_SKIP_PADDING`).

```diff
--- fused_topk_bias_router.py.fixorig
+++ fused_topk_bias_router.py
@@ -20,6 +20,18 @@
 )
 
 
+def _deterministic_topk_indices(x: torch.Tensor, k: int) -> torch.Tensor:
+    """Top-k indices under a total order: value descending, index ascending.
+
+    `torch.topk` guarantees no tie-break, and on some backends is
+    nondeterministic on exact ties; with `sorted=False` it additionally
+    permutes the returned order run-to-run. Both make expert selection
+    irreproducible. A stable descending sort gives the same ordering the fused
+    kernel uses (value desc, then lower index first).
+    """
+    return x.sort(dim=-1, descending=True, stable=True).indices[..., :k]
+
+
 def _get_padding_mask(num_tokens: int) -> torch.Tensor | None:
@@ -310,7 +322,6 @@
         )
         row_bias = torch.where(image_mask, bias_vl, text_bias)
         scores_for_choice = scores.view(-1, n_routed_experts) + row_bias
-    # For batch invariance, use sorted=True to ensure deterministic expert selection
     if hash_indices_table is not None:
@@ -327,10 +338,7 @@
                 image_mask, vl_indices.to(topk_indices.dtype), topk_indices
             )
     else:
-        use_sorted = envs.VLLM_BATCH_INVARIANT
-        topk_indices = torch.topk(scores_for_choice, k=topk, dim=-1, sorted=use_sorted)[
-            1
-        ]
+        topk_indices = _deterministic_topk_indices(scores_for_choice, topk)
     topk_weights = scores.gather(1, topk_indices)
```

## 2. Step 2 — op-level confirmation (before any restart)

`~/opcheck.py` in box1's container, on the canonical layer-21 dump
(`~/tie-dump/L21_f9.pt`, `rl_sha12 = 9baf850ea3f8`, 740x288 fp32):

- **Tie rows re-derived, not assumed: [40, 549]** — row 40 ties experts
  {12, 202} at fp32 `0x414467e2`; row 549 ties {243, 250} at `0x4142e219`.
  Matches the spec exactly.
- Live config: `sigmoid`, `topk=8`, `num_expert_group=1`, `topk_group=1`,
  bias present, `renormalize=True`, `routed_scaling_factor=2.5`, N=30 calls
  of the **patched** `grouped_topk` (the real decorated function imported
  from the patched module; helper presence asserted, `__file__` asserted):
  **distinct topk_ids 1/30, distinct topk_weights 1/30.**
- **Sensitivity control** (stock `.fixorig` loaded via SourceFileLoader in
  the same process): **30/30 distinct** for both tensors — the test would
  have detected nondeterminism, so 1/30 means something.
- Tie resolution: row 40 8th-slot expert **12** (not 202), row 549 → **243**
  (not 250) — ascending index, as specified. Full row-40 id order:
  `[250, 170, 108, 116, 185, 283, 182, 12]`.

## 3. Step 3 — end-to-end validation

All measured on the live 2-box TP=2 server, prefix caching **off**
(`--no-enable-prefix-caching` verified in the running serve argv, not just
the yaml), 2 warmup requests before every measurement batch.

### (a) Determinism — table at the top of this file

244/614/1196 tokens: **1/5** (were 5/5). 2373/4718/9705/14782: still 5/5.
The short-length rows flipping 5/5 -> 1/5 is itself behavioral proof the
patched code path is what the server is running (guard 5 satisfied twice
over: op-level and this).

**Follow-up localization of the residual** (`/tmp/discrim.py`):
- `max_tokens=1` (prefill + first sampled token only), 5 reps: 1421 tok ->
  **1/5**; 2806 tok -> **2/5** (bimodal); 5571 tok -> **2/5** (bimodal).
  The residual divergence therefore lives in **prefill**, not decode — the
  very first sampled token already differs.
- 2806 tok with 64 gen tokens: 5/5 distinct, common prefix across reps = 1
  character — consistent with the first token being the divergence point.
- Bimodality (exactly 2 outcomes in 5) is the signature of a single
  bit-level race, not chaos.

### (b) Layer-level taps (run; taps reverted afterwards)

`/tmp/tap_patch.py` applied in both containers, one restart with
`VSH_TAP_FILE=/home/davidcanar/tap.log` (verified present in the engine's
`/proc/<pid>/environ`), driver `/tmp/bisect2.py` (all-layer analyser, both
ranks). Input guard passed in every run (identical id hash, identical
frontend id0=2833).

- **740 tokens (614-target), 3 forwards: ALL 45 LAYERS BIT-IDENTICAL across
  forwards on BOTH ranks** (h/r/p/c at layers 0..44). Under stock, layer 3
  diverged 8/8; under `sorted=True`, layers 21/34 flipped. Both are gone.
- 2434 tokens (2048-target), 5 forwards: diverges from **layer 3** onward on
  both ranks (hidden/residual/post/comb; layers 0-2 clean).
- 3323 tokens (2806-target), 5 forwards: same — first divergence **inside
  layer 3** on both ranks.
- Layer 3 is the **first DSA sparse-attention layer** (PENDINGWORK 10.2),
  and the divergence appears only at longer lengths — consistent with
  det_sweep's own "length-dependent sparse-attention path" hypothesis, and
  NOT with the router (which is now provably deterministic at op level).
- Taps reverted on both boxes (`reverted; CLEAN`); `model.py` back to stock
  md5 `e93b8176ee67ba6df5dc17b9a8d867c9` on both; router patch untouched.

### (c) Quality

- 4 real `/v1/chat/completions` prompts with
  `chat_template_kwargs: {"reasoning_effort": "low"}`: all coherent and
  factually correct (train catch-up problem solved correctly → 5pm with
  correct arithmetic and check; Rayleigh scattering named; memoized
  Fibonacci with O(n); TCP vs UDP differences correct). All
  `finish_reason: stop`, no degenerate output.
- Tool calling (`/tmp/stream_tools.py`, 6-tool schema, streaming and
  non-streaming): both returned `finish_reason: tool_calls`, 1 call to
  `zz_unguessable_writer`, arguments parse as **VALID JSON** (3473-char
  body_text). Unchanged from pre-fix behaviour.

### (d) Performance (`/tmp/ctx_sweep.py`, same boot as (a))

| prompt tok | prefill tok/s | baseline (284-334) |
|---:|---:|---|
| 524 | 271 | below range; short-prompt point is TTFT-dominated (1.9 s total) |
| 2086 | 340 | in range |
| 8191 | 328 | in range |
| 32827 | 317 | in range |

No regression; consistent with the spec's prediction that the stable sort
is slightly cheaper than `topk` at this shape (71.6 vs 97.8 us per router
call in the design measurements). **Decode step time was NOT compared to
the 200-231 ms baseline**: that baseline was recorded with MTP speculative
decoding ON, but the cluster deliberately runs `glm53_mtp_tokens: 0`
(PENDINGWORK 1.2 — MTP corrupts structured output on ROCm), so
`ctx_sweep.py`'s step/acceptance columns read 0. What I can report: decode
throughput 7.6-9.4 tok/s across 0.5K-32K contexts (MTP off ⇒ 1 token/step,
so ~106-131 ms/step) — same ballpark as PENDINGWORK's documented MTP-off
decode cost. I did not flip MTP on to chase the comparison (it would
change the determinism conditions and reintroduce a known correctness bug).

## 4. Regressions and skips

- **No regression found**: quality, tool calling, prefill throughput all
  fine; server boots normally; both containers byte-compile.
- Not skipped, but honest: the layer-level test (b) was run at 3 lengths,
  not the exhaustive matrix; REPS were 3-5, not the 24 used historically.
- I did not run the box2 op-level 30-call test (the tie dump used is
  rank0's; cross-rank op determinism was established in the design phase by
  `fdF_rank1.py` and was not re-run).

## 5. What I could NOT verify — under-claims

1. **The central claim is partial.** The fix does NOT deliver end-to-end
   greedy determinism at >=~2.4K-token prompts. A second, independent,
   length-dependent prefill nondeterminism remains, first observed **inside
   layer 3** (first DSA layer) at 2434 and 3323 tokens, on both ranks.
   Between 1421 (deterministic) and 2434 (divergent) the threshold is not
   pinned; I did not bisect the exact token count.
2. **The residual is localized only to "inside layer 3".** Attention vs
   MoE vs norm inside that layer was NOT distinguished (would need the
   six-point l3 tap; not run). Calling it "the sparse-attention path" is
   inference from the layer index and length dependence, not a measurement.
3. **Cross-rank divergence mode unknown**: I compared each rank across its
   own forwards; I did not test whether rank0 and rank1 diverge in the same
   or independent ways.
4. **5-rep determinism cells cannot exclude rare flips.** A race with
   per-call probability under ~10% could pass a 5-rep cell. The 1/5 rows
   are supported by the 30x op-level test and the 3-forward all-layer tap,
   but they are not proof of zero-probability divergence.
5. **No stock-boot A/B for performance.** The "no regression" claim rests
   on the patched boot landing inside the recorded baseline range plus the
   design phase's op-level A/B; I did not reboot with `.fixorig` to
   re-measure prefill on stock.
6. **Decode-step baseline not comparable** (MTP off in production; see 3d).
7. **E2E determinism was measured with prefix caching OFF only**, per
   protocol. The final production config (caching ON) was not
   re-swept — repeated identical prompts would hit the cache and the
   measurement would not mean the same thing.
8. The upstream stock E2E baseline (5/5) is taken from the task brief /
   prior sessions' records; I did not re-measure stock end to end. The
   op-level stock control (30/30) is the fresh sensitivity evidence.
9. `fused_topk_bias_router.py`'s patched path is **not exercised by this
   model** (its `hash_indices_table`/VL branches and the `use_sorted` else
   branch are not the GLM live path); it was patched for spec parity and
   compiles, but no behavioral test covers it here.

## 6. Final server state

- Cluster **up and serving**: `curl http://127.0.0.1:1235/v1/models` -> 200;
  greedy completion verified ("The capital of France is" -> " Paris. In
  French, Paris is pronounced ...").
- `~/vsh-config.yaml`: `glm53_prefix_cache: 1` **restored**
  (`--no-enable-prefix-caching` absent from serve argv), `glm53_mtp_tokens: 0`
  (pre-existing deliberate setting, untouched).
- Final boot has **no** `VSH_TAP_FILE` in the engine environment; taps
  reverted on both boxes; `model.py` stock on both.
- **The patch is a LIVE CONTAINER EDIT and will NOT survive an image
  rebuild.** Both boxes' `vllm-glm` containers carry the edited files with
  `.fixorig` backups beside them
  (`.../router/grouped_topk_router.py.fixorig`,
  `.../router/fused_topk_bias_router.py.fixorig`). The durable version is
  the repo patch file the PR-packaging agent is producing from the same
  FIXSPEC. Nothing was committed, pushed, or posted.
