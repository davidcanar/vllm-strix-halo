# RESULT — deterministic selection primitive for the `grouped_topk` upstream fix

**Date:** 2026-09-05 · **Task:** pick the deterministic selection primitive
(Task brief `pick the deterministic selection primitive for a vLLM upstream
fix`), building on `HANDOFF-tie-result.md` and PENDINGWORK.md §10/§11.
**Status: MEASURED, all five sections closed, one new op-level finding.**

---

## 1. The decision

**YES — `torch.sort(scores, dim=-1, descending=True, stable=True)` is bitwise
deterministic on the real tie-bearing router tensors on this platform
(gfx1151, torch 2.11.0+rocm10.0.0): 1/50 identical results on every
tie-bearing tensor, in eager and under the deployed `@torch.compile`
decorator, in separate processes, on both TP ranks' GPUs. A per-rank fix is
viable and the heavier "compute once and broadcast" patch is NOT needed.**

**The stable sort delivers exactly the required total order — score
descending, then expert index ascending — and on the real tensors it is
*cheaper* than today's `torch.topk` call (~72 us vs ~98 us at 740×288, k=8),
so the fix costs nothing; it saves ~1.1 ms per prefill forward.**

Two secondary findings that change the shape of the upstream PR:
`torch.use_deterministic_algorithms(True)` does **not** make `topk`
deterministic (no raise, no warning, still flips), and
`torch.topk(..., sorted=False)` — **today's default in `grouped_topk`** —
returns a different *order* of the correct top-8 on ~715 of 740 rows on every
call, even on tie-free tensors (§E). The fix must therefore produce a
deterministic **order**, not just a deterministic set, which the stable sort
does and `sorted=True` alone does not.

---

## 2. Section A — is a stable sort deterministic here? (decision point)

All runs inside the container, GPU, on the selection scores `grouped_topk`
actually feeds to the k=8 selection — `sigmoid(router_logits) + bias`,
computed under the **same `@torch.compile(dynamic=True, backend=
simple_compile_backend)` decorator** as the deployed function, because
inductor's fused `sigmoid+add` differs from eager by 1 ULP on ~0.15 % of
elements (established in `HANDOFF-tie-result.md` §2). Ties were re-derived,
not assumed, before every arm (guard 7), and are identical under eager and
compiled arithmetic:

| tensor | rl sha12 | k-boundary tie rows (re-derived) | tied experts | value (fp32 hex) |
|---|---|---|---|---|
| L21_f10 … f14 (all 6) | `9baf850ea3f8` | **40, 549** | {12, 202}, {243, 250} | `414467e2`, `4142e219` |
| L34_f11/f12/f14 | `693ffcde86d5` | **651** (mult. 3) | {58, 162, 188} | `414aa7e4` |
| L5_f10 … f14 | `d43ddaca6445` | **none** | — | — |

**N = 50 identical calls per arm** (task asked for ≥30):

| arm | L21 (2 tie rows) | L34_f11 (3-way tie) | L5 control |
|---|---|---|---|
| `torch.sort(descending=True, stable=True)[:k]` | **1/50**, 0 varying rows | **1/50**, 0 varying | **1/50** |
| `torch.argsort(descending=True, stable=True)[:k]` | **1/50**, 0 varying | **1/50**, 0 varying | **1/50** |
| `torch.sort(descending=True, stable=False)[:k]` (control) | **1/50**, 0 varying | **1/50**, 0 varying | **1/50** |
| `torch.topk(k=8, sorted=True)` (control) | **2/50**, varying row **[40]** | **2/50**, varying row **[651]** | **1/50** |
| same, inside the deployed `@torch.compile` decorator | as above, per arm | as above | as above |

So: the nondeterminism lives in `torch.topk`'s selection, not in the sort
machinery — on these tensors even `stable=False` sorting is deterministic
(though that is not a contract; only `stable=True` guarantees it).

**Tie-break direction — the stable arms do break ties by ascending index:**

| row | tied experts | stable-sort picks | is it the lower index? | where it lands |
|---|---|---|---|---|
| L21 row 40 | 12, 202 | **12** | yes | 8th (last) slot |
| L21 row 549 | 243, 250 | **243** | yes | 8th slot |
| L34 row 651 | 58, 162, 188 | **58** | yes (lowest of three) | 8th slot |

**Controls run to make "deterministic" mean something** (`fdA2_addr.py`):
same input buffer every call → 1/50; **fresh device allocation every call**
(different address each time, as in a real forward) → 1/50; **allocator
churn** (allocating/freeing junk of varying sizes between calls) → 1/50.

**Cross-process and cross-rank (the load-bearing check for TP=2).** The
stable-sort index tensor digest is `a8f7561a1b06d94f` for L21 — identical
across two separate processes on box1 **and on box2's GPU** (the other TP
rank, different machine, `fdF_rank1.py`, N=30). On box2 the same
`topk(sorted=True)` arm flipped 2/30, and one of its two outcomes *is* the
stable-sort digest. Both ranks therefore compute bitwise-identical
`topk_ids` from their bitwise-identical inputs, which is exactly what
PENDINGWORK §11.3's "a fully deterministic per-rank top-k would make them
agree automatically" requires. **Verified empirically, not just argued.**

---

## 3. Section B — `torch.use_deterministic_algorithms(True)`

N = 30 on the L21 tie tensor, warnings captured:

| arm | flag off | `warn_only=True` | `strict=True` |
|---|---|---|---|
| `topk(sorted=True)` | 2/30 distinct | **2/30** | **2/30** |
| `topk(sorted=False)` | 30/30 distinct (raw) | **30/30** | **30/30** |
| `sort(stable=False)` | 1/30 | 1/30 | 1/30 |
| `sort(stable=True)` | 1/30 | 1/30 | 1/30 |

**PyTorch's determinism flag neither raises, nor warns, nor changes `topk`'s
behaviour on ROCm.** No warning of any kind was emitted in any arm. So the
determinism problem cannot be delegated to PyTorch, and a vLLM-side fix is
required. (Note this is PyTorch's flag; vLLM's `VLLM_BATCH_INVARIANT` is a
different mechanism — it only flips `use_sorted=True` inside `grouped_topk`,
and is already known not to boot this model, PENDINGWORK §10.4. Not
re-tested here.)

---

## 4. Section C — the fused path: present, but unreachable here

**`ops.grouped_topk` is present in this build but does not run on gfx1151.**
Its Python wrapper hard-guards the platform:

```python
if not current_platform.is_cuda():
    raise NotImplementedError(
        "The fused grouped_topk kernel is only available on CUDA platforms")
```

Confirmed by direct call on the real tie tensor → `NotImplementedError`.
Furthermore `torch.ops._moe_C` has **no** topk op at all in this ROCm build
(`[x for x in dir(torch.ops._moe_C) if 'topk' in x] == []`), so the CUDA
kernel is not merely gated off, it is not compiled in.

**Exact branch conditions**, from
`vllm/model_executor/layers/fused_moe/router/grouped_topk_router.py`
(md5 `da7b32ec80cd79291a2b58e3df8ccbbe`, stock):

- **Lines 91–97**, inside `grouped_topk()` — the CUDA fused kernel:
  `envs.VLLM_USE_FUSED_MOE_GROUPED_TOPK and current_platform.is_cuda() and
  num_expert_group <= 32 and topk <= 32 and e_score_correction_bias is not
  None`. Live values: `True / **False** / 1≤32 / 8≤32 / True` → **not
  taken**, solely because `RocmPlatform.is_cuda()` is `False`.
- **Lines 327–335**, `GroupedTopKRouter._compute_routing` — the aiter kernel:
  `if rocm_aiter_ops.is_fused_moe_enabled(): grouped_topk_impl =
  partial(rocm_aiter_grouped_topk, …) else: grouped_topk_impl = grouped_topk`.
  Live: `is_fused_moe_enabled()` → **False** (`VLLM_ROCM_USE_AITER_MOE=0`),
  so the **Python fallback is the live path**.
- **Lines 225–243**, `GroupedTopk.forward_hip` — same aiter gate, same result.
- (Lines 296–324, `valid_grouping()`, route to `fused_topk_bias`/`fused_topk`
  only when the grouping is invalid; here 288 % 1 == 0, so not taken.)

**`rocm_aiter_grouped_topk` does run standalone on gfx1151** (called directly
in my own process — nothing on the server was enabled). It is
**deterministic**: 1/50 distinct in both raw order *and* set on L21 and L5.
But its tie-break is an opaque deterministic function of the kernel's
internal reduction, **not** ascending index:

| tensor / row | tied | aiter picks | ascending-index picks |
|---|---|---|---|
| L21 row 40 | 12, 202 | **202** | 12 |
| L21 row 549 | 243, 250 | **250** | 243 |
| L34 row 651 | 58, 162, 188 | **188** | 58 |
| L34 row 568 | 58, 188 | **188** | 58 |
| L34 row 489 | 58, 188 | **188** | 58 |

A controlled synthetic test (9 tie configurations, each asserted to have
exactly 7 experts strictly above the boundary tie) shows there is **no index
rule**: `{5,200}→200`, `{0,279}→0`, `{100,101}→100`, `{140,141}→140`,
`{10,150,200}→200`, `{1,2,3,26}→1`, `{0,1,2}→0`, `{278,279}→278`. The five
real rows all landing on the higher index is coincidence.

**Consequence for the PR:** the bug is **fallback-only on this platform** —
on ROCm with aiter MoE enabled the router is already deterministic, which is
plausibly why NVIDIA users (fused `_moe_C` kernel) and aiter-enabled ROCm
users have not hit it. An upstream fix in the Python `grouped_topk` fixes
this platform, **but** it will make the fallback's tie-break (lowest index)
disagree with aiter's (kernel-defined) on exact ties. Both are deterministic
per path, so TP consistency holds within each path, but the same checkpoint
would route differently on the two paths. That divergence should be stated
in the PR, or the fix extended to the kernels.

---

## 5. Section D — what the fix costs

Real shape **740 × 288 fp32, k = 8**, GPU, round-robin interleaved arms,
CUDA-event per-iteration timing, N = 50 after 20 warmup, three runs
consistent to ±1 us. **Measurement floor control:** a trivial in-place add,
a clone, and a sigmoid of the same tensor all time at **6.4–8.1 us median**
— so the numbers below are genuine kernel work, not launch overhead. The GPU
is shared with the live server, so absolute values are inflated by
contention; the *relative* comparison is what is reliable.

| arm | median us | p90 us |
|---|---|---|
| `torch.topk(sorted=False)` (today's call) | **97.8** | 99.5 |
| `torch.topk(sorted=True)` | **101.7** | 102.5 |
| `torch.sort(stable=True)` + slice | **71.6** | 72.6 |
| `torch.argsort(stable=True)` + slice | **71.6** | 72.4 |
| `torch.sort(stable=False)` + slice | 71.5 | 72.3 |
| same stable sort inside the deployed `@torch.compile` | 84.6 | 87.0 |

**The stable sort is ~26 us cheaper than today's `topk` at this shape.** This
is data-dependent and worth stating precisely (isolated A/B, `fdD3_ab.py`,
two runs): on the **real** score tensors `topk(sorted=False)` costs 97.6–98.0
us, on `randn` data 77.7 us, on a smooth linspace 81.0 us, while the stable
sort is data-independent at 72.0–73.7 us in all three cases. So on real
router data the fix is a net win; on synthetic data the two are comparable
(~74 vs ~74).

**Per prefill forward** (42 MoE layers ⇒ the router runs 42 times, 740
tokens, ~2.4 s prefill at ~310 tok/s):

| | per router call | × 42 per forward | % of 2.4 s |
|---|---|---|---|
| today, `topk(sorted=False)` | 97.8 us | **4.11 ms** | 0.171 % |
| fix, `sort(stable=True)` | 71.6 us | **3.01 ms** | 0.125 % |
| **net change** | −26.2 us | **−1.10 ms** | **−0.046 %** |

The fix therefore **saves** ~1.1 ms per prefill rather than costing anything.

**Memory:** a full 288-wide sort materialises values (740×288 fp32 = 832 KiB)
+ int64 indices (1665 KiB) ≈ **2.44 MiB transient**, vs `topk`'s 69 KiB —
about **2.4 MiB extra per router call**, freed immediately (peak, not
retained; the slice actually consumed downstream is the same 46 KiB in both
cases). Negligible against a live server's footprint, but it scales with
n_tokens × n_experts, so at large batch × many experts it should be
mentioned in the PR (e.g. 32k tokens × 288 experts ⇒ ~107 MiB transient).

---

## 6. Section E — does the tie-break choice change the selected experts?

N = 60 calls of `torch.topk(..., sorted=True)` per tensor, compared with the
ascending-index stable sort:

| tensor | tie rows | rows whose SET varies | `topk` picks at the racing row | ascending-index picks |
|---|---|---|---|---|
| L21 (`9baf850ea3f8`) | 40, 549 | **1 of 740** (row 40) | row 40: **202** in 39/60, **12** in 21/60 | **12** |
| | | | row 549: **243** in 60/60 (no race) | **243** (identical) |
| L34 (`693ffcde86d5`) | 651 | **1 of 740** (row 651) | **58** in 32/60, **162** in 28/60 | **58** |
| L34 (`fa6705893b6b`) | 568 | **1 of 740** (row 568) | **58** in 26/60, **188** in 34/60 | **58** |
| L5 (`d43ddaca6445`) | none | **0 of 740** | — | — |

- **Rows affected at all: 1 of 740** per tensor (the tie row where the race
  actually fires); L21 has 2 tie rows but only row 40 raced in 60 calls.
- **In every case the ascending-index choice is one of `topk`'s existing
  outcomes** — across all tensors there were **0 rows** where the
  ascending-index set was never returned by `topk` in 60 calls. The fix picks
  a winner that `topk` already produces; it does not introduce a selection
  `topk` would never make.
- **Rows where `topk(sorted=True)`'s ORDER differs from the stable sort's
  ORDER: exactly the tie rows** (`[40]` at L21; `[]` at L34 and L5). So on
  tie-free rows the fix is bitwise identical to `topk(sorted=True)` in both
  set and order — the behavioural delta is confined to exact fp32 ties,
  which is what a reviewer needs to see.

**New op-level finding (worth its own line in the PR):**
`torch.topk(..., sorted=False)` — the **default in `grouped_topk` today** —
returns the correct top-8 **set** but in a **different order on ~715 of 740
rows on every call**, even on the tie-free control tensor (60/60 distinct raw
index tensors; 0/60 distinct sets). The gathered routing weights
(`original_scores.gather(1, topk_ids)`, line 150) are permuted to match. The
returned order is never the descending-score order (0 of 740 rows in 20
calls). This is op-level evidence that plausibly explains the one remaining
open question, PENDINGWORK §11.4 — why `sorted=False` diverges on *every*
call in situ at tie-free layer 3 — via a changed k-summation order
downstream. **The op-level fact is measured; the downstream accumulation
link is inference and was not verified end-to-end here** (it would require
patching the deployment, which was out of scope). It also means the earlier
statement that "standalone `sorted=False` matched `sorted=True` on real
tensors" (`HANDOFF-tie-result.md` §4) is true of *sets* only — that
analyser sorted the ids before comparing, discarding the order.

---

## 7. Recommendation — smallest correct upstream change

In `vllm/model_executor/layers/fused_moe/router/grouped_topk_router.py`,
replace the nondeterministic selection in `grouped_topk()` with a stable
descending sort, which realises the total order *score descending, expert
index ascending* in one primitive:

```python
# line 147-150 today
if e_score_correction_bias is not None:
    topk_ids = torch.sort(tmp_scores, dim=-1, descending=True,
                          stable=True)[1][:, :topk]
    topk_weights = original_scores.gather(1, topk_ids)
else:
    topk_weights = torch.topk(tmp_scores, k=topk, dim=-1)[0]  # or gather
    topk_ids    = torch.sort(tmp_scores, dim=-1, descending=True,
                             stable=True)[1][:, :topk]
    topk_weights = tmp_scores.gather(1, topk_ids)
```

and the same substitution at **line 135** for `group_idx`
(`torch.topk(group_scores, k=topk_group, …)`), which has the identical
hazard for multi-group configs (`topk_group > 1`); here `topk_group == 1`
so it is unreachable, but the fix should not leave the same bug one line
above. The `use_sorted = envs.VLLM_BATCH_INVARIANT` gate (line 134) then
becomes unnecessary for selection and can be dropped or repurposed.

**Why this and not the alternatives:**

- *`sorted=True` alone* — necessary, not sufficient; still flips on exact
  ties (2/50 here, and PENDINGWORK §10.3 item 1).
- *`torch.use_deterministic_algorithms(True)`* — does nothing for `topk`
  (§B); not a lever.
- *Broadcast the selection from rank 0* — unnecessary. The ranks' inputs are
  bit-identical and the stable sort is deterministic across processes and
  machines (§A), so both ranks already agree. Broadcasting is a heavier,
  more invasive patch that also changes the TP communication pattern.
- *Detect ties with `topk(k+1)` and re-sort only those rows* — fewer FLOPs
  on paper, but more code, two kernels, and §D shows the full sort is
  already cheaper than a single `topk` on real data at this shape. Not worth
  the complexity.

**Ways this change could alter behaviour beyond exact ties — flag these in
the PR:**

1. **k-order changes on every row relative to today's default.** Under the
   stock `use_sorted=False`, `topk` returns an arbitrary permutation
   (§E); the fix returns fully sorted order. The (id, weight) *pairs* are
   unchanged on tie-free rows, but their sequence changes, and if the
   downstream MoE combine sums over k in sequence order the low-order bits
   of the output change on tie-free rows too — now deterministically.
   Relative to `sorted=True`, the delta is exactly the tie rows (§E).
2. **Cross-path tie-break divergence.** The fallback would pick the lowest
   tied index; the ROCm aiter kernel picks a kernel-defined one (§C), and
   the CUDA `_moe_C` kernel's behaviour is untested. Deterministic per
   path, but not identical across paths.
3. **Perf on other platforms is unmeasured.** On gfx1151 the sort is
   cheaper; on CUDA it may not be (§D measured ROCm only).
4. **Transient memory** grows by ~2.4 MiB per call at this shape, scaling
   with tokens × experts (§D).

---

## 8. What I could NOT verify / controls not run

- **No end-to-end verification.** I did not patch the deployment and did not
  observe greedy decoding becoming reproducible. Everything here is op-level
  on dumped tensors. The claim "a per-rank fix is viable" rests on both
  ranks producing bitwise-identical ids from identical inputs, which I
  verified on both GPUs — not on the live engine.
- **The downstream link for §E's new finding is inference.** That permuted
  k-order changes the MoE combine's accumulation order was not measured; it
  needs the real `fused_experts`/combine, which I did not run (would require
  expert weights; guard 6).
- **No NVIDIA testing.** `_moe_C.grouped_topk`'s tie behaviour is unknown.
- **The aiter tie-break rule is characterised behaviourally** (5 real + 9
  synthetic cases); the kernel source was not read.
- **Timings are contention-inflated and ROCm-only.** The GPU is shared with
  the live server. A trivial-op floor control (6.4–8.1 us) shows the
  measurements are genuine kernel time, but absolute values should not be
  quoted as clean-GPU numbers. The `sort`-vs-`topk` comparison was
  reproduced three times with the same ordering.
- **Per-row flip probabilities are 60-call sample estimates**, not exact
  probabilities. Row 549's tie never raced in 60 calls; it is not immune,
  just unobserved.
- **I did not trace the model's router instantiation** to `GroupedTopKRouter`
  in source; I evaluated the branch conditions at module level (both fused
  gates closed). The live-path identification rests on
  `HANDOFF-tie-result.md`'s in-situ evidence plus those live values.
- **`sort(stable=False)` was also deterministic here** (1/50) — reported as
  an observation, not a recommendation; only `stable=True` is a contract.
- **Not tested at other shapes** (decode batches of 1–8 tokens, very large
  batches). The 740-token prefill shape is the one that was dumped.
- **`VLLM_BATCH_INVARIANT=1` was not exercised** in a standalone process;
  its boot failure is documented elsewhere (PENDINGWORK §10.4) and was not
  re-attempted.

---

## 9. Reproduction

All scripts left in **`/home/davidcanar/fixdesign/`** on box1 (host), run
inside the container via the bind-mounted `/home/davidcanar`. Nothing was
copied into the container's filesystem; every run was
`podman exec -i vllm-glm python3 - < script`.

| script | section | what it does |
|---|---|---|
| `fd0_recon.py` | guard 7 | re-derives tie rows on every dump, eager + compiled |
| `fdA_det.py` | A | 50-call determinism, 6 arms × 4 tensors, tie-break direction, cross-process digests |
| `fdA2_addr.py` | A | fresh-allocation and allocator-churn controls |
| `fdB_detflag.py` | B | `use_deterministic_algorithms` off/warn/strict × 4 arms |
| `fdC_fused.py` | C | live branch conditions, `ops.grouped_topk` probe, aiter probe |
| `fdC2_aiter.py` | C | `_moe_C` presence, aiter determinism vs ascending-index |
| `fdC3_tiebreak.py` | C | aiter tie-break on all 5 real tie rows |
| `fdC4_synth.py` | C | 9 controlled synthetic tie configurations for aiter |
| `fdD_bench.py` | D | timing, 6 arms, + extrapolation + memory (log: `fdD_bench.log`) |
| `fdD2_floor.py` | D | trivial-op floor control |
| `fdD3_ab.py` | D | real vs randn vs linspace A/B for topk vs sort |
| `fdE_delta.py` | E | set-vs-order decomposition, per-tie-row pick frequencies |
| `fdE2_perm.py` | E | `sorted=False` permutation characterisation |
| `fdF_rank1.py` | A | cross-rank check, run on box2's container (deleted from box2 after) |

```bash
ssh davidcanar@192.168.0.102
scp <script> box1:/home/davidcanar/fixdesign/
podman exec -i vllm-glm python3 - < /home/davidcanar/fixdesign/fd0_recon.py
podman exec -i vllm-glm python3 - < /home/davidcanar/fixdesign/fdA_det.py
# ... likewise fdA2_addr, fdB_detflag, fdC_fused, fdC2_aiter, fdC3_tiebreak,
#     fdC4_synth, fdD_bench, fdD2_floor, fdD3_ab, fdE_delta, fdE2_perm
# cross-rank (box2):
scp ~/fixdesign/fdF_rank1.py 10.0.2.2:/home/davidcanar/
ssh 10.0.2.2 'podman exec -i vllm-glm python3 - < /home/davidcanar/fdF_rank1.py'
```

Input data: the existing `~/tie-dump/` (rank0) and, for the box2 run, box2's
own `~/tie-dump/` (rank1) — both contain the canonical layer-21 tensor
`rl_sha12 = 9baf850ea3f8` with tie rows 40 and 549. Note the first synthetic
aiter test I wrote was wrong (logits in [1,20] saturate `sigmoid` to exactly
1.0 in fp32, creating a mass tie); `fdC4_synth.py` is the corrected version
with per-row assertions that the constructed tie is real.

---

## 10. Server and deployment state at finish

- `curl http://127.0.0.1:1235/v1/models` → **200** (checked after every
  measurement batch). The server was never restarted, patched, or sent any
  inference traffic.
- `grouped_topk_router.py` md5 `da7b32ec80cd79291a2b58e3df8ccbbe` and
  `vllm/models/glm5next/nvidia/model.py` md5 `e93b8176ee67ba6df5dc17b9a8d867c9`
  — both **stock**, matching `HANDOFF-tie-result.md` §6; zero
  `_vsh_tie`/`_VSH_TIE` markers; pre-existing `.orig` backups untouched.
- `~/vsh-config.yaml` still `glm53_prefix_cache: 1`. Nothing committed,
  pushed, or posted anywhere.
- box2: `vllm-glm` container still `Up 45 hours` (un restarted), 18.5 GiB GPU
  memory free; the temp script I copied there was deleted. box2 has no
  listener on :1235 (the API server lives on box1) — that is its normal
  state, not something I broke.
- One side effect to declare: `rocm_aiter_grouped_topk` triggered a one-off
  aiter JIT build (`module_moe_asm`, ~16 s) cached under
  `/opt/venv/.../aiter/jit/build/` inside the container. It is a build cache
  write from my short-lived process; the running server was not touched.
