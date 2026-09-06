# PENDINGWORK.md — state of play, last reviewed 2026-09-05

Working notes for picking this up later, most likely once the upstream issues
below are fixed. Two threads of work happened here: a **performance** push that
succeeded, and a **correctness** investigation that found three defects — one
of which invalidates part of the performance story.

Read the [Correctness](#1-correctness-the-blocking-problem) section first. It is
the reason this rig is not currently trustworthy for agent workloads.

**Reading order and current state.** Sections 1-6 are the main body. Sections 7
and 8 are evening addenda from a second investigator; section 8 was reviewed on
2026-09-05 and its two headline claims were revised — **8.6 is retracted** (a
tap off-by-one, not input corruption) and **8.7's mHC attribution is withdrawn**
while its layer-level bisection stands.

Net, as of 2026-09-05: **two independent top-k bugs** (sections 10-12). The
MoE router one is **fixed and shipped** (§12); the DSA indexer one is
upstream's, engages only above `index_topk`, and is being fixed there. The MoE router's `torch.topk(tmp_scores, k=8)` resolves exact fp32
k-boundary ties nondeterministically, independently on each TP rank. That
accounts for layer 3 and for layers 21/34. The "second defect" at the combine
stage was retracted as a rank0-only analyser artifact (11.2). A fix needs a
deterministic tie-break (score, then expert index) or a broadcast selection;
`sorted=True` alone fixes only layers 0-20 (11.3). One question is still open:
why `sorted=False` diverges every call in situ at a tie-free layer (11.4). Four hypotheses of our own were
excluded by measurement along the way (the MoE `SPLIT_K` tuning, the
`NCCL_PROTO` unpinning, the request input path, the model runner), and one of
our exonerations turned out to be too narrow rather than wrong — see the lesson
box in 10.2.

---

## 1. Correctness: the blocking problem

### 1.1 Greedy decoding is not reproducible (upstream, open)

Five byte-identical `temperature=0` requests return **five different
completions**, at every prompt length tested. This is a correctness bug by
definition — greedy decoding is a deterministic function of its input.

`/v1/completions`, no chat template, no tools, no parsers, `seed=1234`,
`ignore_eos`, `max_tokens=64`, strictly sequential on an idle server, MTP
**off**, tbv_ar2 **off**, `--enforce-eager`, **prefix caching disabled**:

| prompt_tokens | distinct / 5, prefix cache **off** | (first run, cache on) |
|---:|---:|---:|
| 244 | **5** | 5 |
| 614 | **5** | 5 |
| 1196 | 3 | 4 |
| 2373 | **5** | 5 |
| 4718 | **5** | 5 |
| 9705 | **5** | 5 |
| 14782 | **5** | 4 |

Not length-gated: it diverges at 244 tokens, about 1/8 of `index_topk`.

> **Correction, and a lesson.** The first version of this measurement — and the
> first upstream comment — was taken with prefix caching at the V1 default while
> every repetition sent a **byte-identical prompt**, so reps 2-5 were partly
> served from cached KV. The upstream comment claimed prefix caching had been
> ruled out; it had not. Nonce-prefixed prompts belonged to a *different* test
> family (prefill throughput, the tool-calling ladder), and
> `--no-enable-prefix-caching` had never been run. Re-run properly, the finding
> holds — but the claim was wrong when it was made and was corrected publicly.
> **Any determinism measurement on this stack needs
> `--no-enable-prefix-caching`**, because repeated identical prompts otherwise
> take a different path.

**The outcome set looks discrete.** Specific hashes recur across independent
runs — `a56aaffa` in the 614-token row both with and without prefix caching,
`ff15d4ee` likewise at 1196. A handful of arithmetic paths rather than unbounded
variety, which is what a nondeterministic reduction *order* produces and which
argues against an out-of-bounds / uninitialised-data explanation.

**Consequences beyond irreproducibility.** Some runs degenerate. One
summarisation run produced `module m00001 depends on m07919, m00002 on m07919,
m00002 on m07919, ...` while another run of the identical request was coherent.
A needle test (unique passphrase placed early, asked for verbatim) returned
**`velvet-harbor-41955` instead of `velvet-harpoon-41955`** at 12,140 prompt
tokens — the right region attended to, a near-tie token chosen wrongly. Recall
was correct at 241, 3,984 and 24,490 tokens, and at 49,408 the model refused.

**Upstream:** [vllm#54521](https://github.com/vllm-project/vllm/issues/54521)
— *"Qwen3.8-Flash-Next: greedy decoding is non-deterministic from
persistent_topk in prefill when prompt length nears indexer_budget
(sm121/GB10)"*. Open. Our gfx1151 + GLM-5.3 data is posted as
[issuecomment-5545047644](https://github.com/vllm-project/vllm/issues/54521#issuecomment-5545047644).

**Our comments on that thread**, in order:
[1) first data drop](https://github.com/vllm-project/vllm/issues/54521#issuecomment-5545047644) ·
[2) follow-up](https://github.com/vllm-project/vllm/issues/54521#issuecomment-5546571352)
(prefix-cache correction, narrowed candidates, M-invariance table) ·
[3) retraction](https://github.com/vllm-project/vllm/issues/54521#issuecomment-5546779876)
(the `max_num_seqs=1` fault is not reproducible) ·
[4) prefill localisation](https://github.com/vllm-project/vllm/issues/54521#issuecomment-5546997128)
(collective excluded, divergence is in prefill).

We also opened
[jschmied/qwen38-flash-next-gb10#1](https://github.com/jschmied/qwen38-flash-next-gb10/pull/1)
contributing `tools/gemm_m_invariance_rocm.py`, the ROCm counterpart to their
M-invariance harness, at their request. The reporter replied with the
sequential-vs-concurrent discriminator used in 1.1 and pointed at four local
patches they carry — [#54945](https://github.com/vllm-project/vllm/issues/54945)
/ [#54948](https://github.com/vllm-project/vllm/pull/54948) (nondeterministic
MoE finalize, FlashInfer/NVFP4 — NVIDIA-only, not our path),
[#54076](https://github.com/vllm-project/vllm/pull/54076) /
[#53798](https://github.com/vllm-project/vllm/pull/53798) (align-mode Mamba
block-size seeding — **all open PRs, not merged**; both state equal-geometry
behaviour is byte-identical, so they only bite with heterogeneous KV block
sizes, which we do not have with MTP off), and
[#55122](https://github.com/vllm-project/vllm/pull/55122) (make
`persistent_topk` deterministic).

That thread is stuck on the mechanism: the reporter sees a clean
`indexer_budget` threshold, but the independent reproduction in the comments
diverges at **582** tokens (below budget) and **survives disabling MTP**. Our
data agrees with the second reading — divergence everywhere, no threshold — and
adds a second vendor and a second sparse-attention implementation (GLM DSA, not
Qwen QSA), which argues the top-k switch is not the mechanism on any platform.

Related open issues:
[#53257](https://github.com/vllm-project/vllm/issues/53257) (DeepSeek-V4-Flash,
temperature 0, rate scales with concurrency),
[#47069](https://github.com/vllm-project/vllm/issues/47069) (Hopper
batch-invariant path),
[#55131](https://github.com/vllm-project/vllm/issues/55131) (batch-invariant
matmul is not batch-invariant),
[#42259](https://github.com/vllm-project/vllm/issues/42259) (RFC on logprob /
determinism semantics).

**No mitigation available.** `VLLM_BATCH_INVARIANT` exists but `envs.py`
documents it as requiring NVIDIA compute capability >= 9.0, and #55131 reports
it does not fully work anyway.

**Reproducer:** `minrepro.py`, inlined in the upstream comment above. ~30 lines,
stdlib only, exits non-zero on divergence.

### 1.2 MTP corrupts structured output (ours to avoid)

With `glm53_mtp_tokens: 3`, requests die at **exactly 12 output tokens** — 3
MTP steps at k=3 — returning 13-character tool-call argument strings. Prompts
as small as ~4k tokens return completely empty responses (`finish_reason: stop`,
0 content, 0 tool calls).

Isolation, 5 identical requests per cell:

| config | 4,012-token prompt | 3,686-token prompt |
|---|---|---|
| MTP on, tbv_ar2 on | 4/5 died at 12 tokens | 4/5 gave 0 calls or 13–29 char args |
| MTP on, tbv_ar2 **off** | 5/5 real calls | 4/5 died at 12 tokens |
| MTP **off**, tbv_ar2 off | **5/5 real calls**, args 2009–4256 ch | **5/5 real calls**, args 1476–5144 ch |

So `tbv_ar2` is not implicated and MTP is. The upstream recipe for this model
states plainly that **MTP is not supported on ROCm**
(<https://recipes.vllm.ai/zai-org/GLM-5.3-Flash>); we run it via
`vsh-mtp-ropefree-triton-sparse.patch` regardless.

**Current setting: `glm53_mtp_tokens: 0`.** Cost is roughly **3x decode
throughput** (MTP was delivering 2.2–4.0 tokens/step). This is the right trade
for agent use — a harness that silently drops tool calls is useless at any
speed — but it is why decode is now ~4–5 tok/s instead of 10–20.

Related upstream: [#54928](https://github.com/vllm-project/vllm/issues/54928)
(spec decoding changes greedy output at token 30, even K=1 with
`--enforce-eager`), [#53436](https://github.com/vllm-project/vllm/issues/53436),
[#55357](https://github.com/vllm-project/vllm/issues/55357) (MTP repetition
collapse in the thinking block until `max_tokens` — matches our 12,387-token
case that burned 3072 tokens and returned nothing).

### 1.3 Tool calling collapses above ~10k prompt tokens

With MTP off, tool calling works up to ~4k and fails above ~10k. Fixed
unguessable tool schema, 3 tools, streaming, `reasoning_effort: low`:

| prompt_tokens | finish_reason | tool_calls | what happened |
|---:|---|---:|---|
| 295 | `tool_calls` | 1 | valid |
| 4,231 | `tool_calls` | 1 | valid |
| 12,387 | `length` | **0** | 3072 tokens generated, all buffered in an unterminated tool call |
| 24,737 | `stop` | **0** | announces the action, never calls |
| 49,655 | `length` | **0** | writes the file inline as chat content |

The mechanism is an **unterminated tool call**. Real captured output:

```
<tool_call>edit_file<arg_key>content</arg_key><arg_value><!DOCTYPE html>...
</parameter></invoke><parameter name="description">Create index.html game file</parameter></invoke></function>
```

`<tool_call>` x1, `</tool_call>` **x0**, and a drift into Anthropic-style
`<invoke>` / `<parameter>` markers mid-call. `glm47_moe_config` leaves
`ParserState.TOOL_NAME` only on `ARG_KEY_START` or `TOOL_END`, so the parser
buffers indefinitely and the client gets HTTP 200 with nothing after the
preamble. Another observed shape: `<tool_call>write# snake.html\n<!DOCTYPE
html>...` — correct tool name, then freeform text, no `<arg_key>` at all.

This is what made two independent coding harnesses (a DeepSeek harness and
opencode) appear to "just stop" after replying *"I'll create a complete Snake
game…"*.

**Working mitigation, built but NOT applied.** Repeating the tool-call format
at the *end* of the prompt restores a valid call at 12k:

| ~12.3k prompt | tool_calls | finish_reason |
|---|---:|---|
| format only at the start (template default) | 0 | `length` |
| format repeated just before `<\|assistant\|>` | 1, valid JSON | `tool_calls` |

`~/vsh-chat-template.glm53.jinja` on box1 implements this and is validated
offline: non-tool requests render **byte-identical** to the stock template, tool
requests gain the block a second time immediately before the assistant turn, at
a cost of **+76 tokens** per tool request. It also explicitly names the wrong
syntaxes (`<invoke>`, `<parameter>`, `<function_calls>`, markdown) that were
observed. **To apply:** add `--chat-template ~/vsh-chat-template.glm53.jinja`
to `vsh-manual-serve.sh` and restart. Not yet done, and only validated at 12k
with a single successful call — re-run the ladder at 12k/25k/50k before
trusting it.

### 1.4 What was ruled out, and how

Worth recording so it is not re-investigated:

| hypothesis | verdict | evidence |
|---|---|---|
| Tool/reasoning parser flags wrong | **no** | `glm45` and `glm47` are byte-identical registry entries for *both* flags (`Glm47MoeParserReasoningAdapter`, `Glm47MoeModelToolParser`). Recipe and our config are the same code path. |
| Chat template does not render tools | **no** | `/tokenize` shows the `<tools>` block and the `<tool_call>{name}<arg_key>…` instructions rendered correctly. |
| Model ignores the tool schema | **no** | It calls `zz_unguessable_writer(target_path, body_text)` correctly — an unguessable name with unguessable args. |
| Streaming-specific parser bug | **no** | Streaming and non-streaming both return valid JSON at 4k (311 SSE chunks reassembled cleanly). |
| Harness system prompt teaching a rival dialect | **no** | Injecting Anthropic-style `<invoke>`/`<parameter>` docs alongside a valid `tools` array still produced clean GLM-format calls. |
| Prefill chunking (`max_num_batched_tokens: 4096`) | **no** | 2-chunk prompts (4,279 / 5,427) work; a 1-chunk prompt (4,012) fails. |
| Our tuned MoE `SPLIT_K` | **no** | 20 identical `fused_experts_op(..., use_int4_w4a16=True)` calls at the real `E=288, K=4096, N=1024, topk=8`: 1 distinct result, max drift 0.0, for M=1/4/8 under `SPLIT_K=1` and `SPLIT_K=4`. |
| MoE top-k weighted combine (`moe_sum`) | **no** | Source-traced: `fused_experts_op` (`fused_moe.py:1454`) returns `fused_experts_impl(...)` at 1480, which calls `ops.moe_sum(...)` at 1854 before returning — so the combine is inside the op tested bitwise above. |
| Triton split-KV decode attention | **no** | `v1/attention/ops/triton_decode_attention.py` contains zero `atomic_add`; split-KV writes per-split partials indexed by `split_kv_id` and stage 2 reduces them in a fixed loop. Deterministic per shape. |
| Batch-shape / M-dependence channel | **no** (see 1.6) | ROCm BF16 dense is M-invariant through M=64; int4 MoE through M=256. |
| Prefix caching | **no** (now actually tested) | Re-measured with `--no-enable-prefix-caching`; see 1.1. |
| The collective (RCCL) | **no** | `NCCL_ALGO=Ring` + `NCCL_PROTO=Simple` pinned and **verified in both workers' `/proc/<pid>/environ`**, not merely passed as a flag. Still 5/5 distinct. This also clears our own §9 `NCCL_PROTO` unpinning of suspicion. |
| Decode-side kernels as the *origin* | **no** | Prefill alone diverges; see 1.7. |
| `tbv_ar2` custom all-reduce | **no** | Disabling it leaves the divergence unchanged (see 1.2 table). |
| CUDA graphs | **no** | `--enforce-eager` throughout. |
| Prefix caching | **no** | Nonce-prefixed uncacheable prompts throughout. |
| ROCm MRV1 accuracy regression [#54924](https://github.com/vllm-project/vllm/issues/54924) | **not applicable** | Keys on `GlmMoeDsaForCausalLM`; our checkpoint is `Glm5NextForConditionalGeneration`, which is not in `ROCM_DEFAULT_MRV1_ARCHITECTURES`. Verified in the installed source. |
| Reasoning-effort / token budget | **not the cause** | Real but separate; see 3.1. |

**Still not excluded:** `vsh-mtp-ropefree-triton-sparse.patch`, which routes
GLM's rope-free sparse-MLA decode to the Triton ragged kernel because aiter's
asm kernel aborts on gfx1151. It is in the path for every measurement above.
[#50603](https://github.com/vllm-project/vllm/issues/50603) reports
non-determinism **plus long-sequence corruption** on gfx1100/RDNA3 traced to a
Triton paged-attention fallback — a different kernel, same shape of problem
("RDNA takes the Triton path, the Triton path is not reproducible"). **This is
the first thing to test when picking the work back up.**

---

### 1.5 A one-off GPU fault in the sparse indexer (not reproducible)

Running the upstream-suggested "sequential, `--max-num-seqs 1`, nothing else on
the server" cell **killed the engine on the first request**:

```
Memory Fault Error [host: toolbx, GPU index: 0, faulting addr: 0x7ede9e94c000,
  kernel: _deepgemm_fp8_paged_mqa_logits
Warning: Queue error - HSA_STATUS_ERROR_MEMORY_FAULT
... hipErrorIllegalAddress
```

Immediately after `jit_monitor` logged the JIT compile of
`_deepgemm_fp8_paged_mqa_logits` during inference.

> ⚠️ **RETRACTED: not reproducible, and `--max-num-seqs 1` is not the trigger.**
> I told upstream it was isolated to that flag. Retested deliberately: with
> `--max-num-seqs 1` and prefix caching still off, a fresh restart serves a
> trivial 8-token prompt fine, and one request per size at 3 / 97 / 160 / 244 /
> 306 / 368 / 493 / 740 / 1421 / 2806 prompt tokens **all succeed with the
> engine alive after each**. The 244-token prompt that killed it the first time
> now passes at the same setting.
>
> The distinguishing feature of the crash run is in the log line above: it was
> the **first, cold, JIT-compiled launch** of that kernel. The compile is now
> cached on disk and in-process, so the cold path is gone. That makes it a
> plausible **cold-JIT** fault rather than an allocation-size one — which also
> kills the "same allocation-size dependence would read wrong values silently"
> mechanism I had floated. Compare
> [#50603](https://github.com/vllm-project/vllm/issues/50603) Symptom A on
> gfx1100: *"the first `generate()` call in a process produces different output
> from subsequent identical calls… run 0 (the cold/JIT-compiled first call)
> diverges"*, cured by a warmup call.
>
> **Untested and the way to settle it:** clear the Triton cache for that kernel
> and restart, so the first inference launch is cold again. Retracted publicly
> in [issuecomment-5546779876](https://github.com/vllm-project/vllm/issues/54521#issuecomment-5546779876).

That kernel is the sparse-attention indexer's paged MQA logits kernel — the
component [#54521](https://github.com/vllm-project/vllm/issues/54521) is about —
and `max_num_seqs=1` mainly shrinks the allocations it indexes into. Not
established as the mechanism for 1.1 (the recurring-hash observation argues
against a plain OOB read), but an indexing bug there that only faults when the
access leaves a mapped page would be consistent with silent wrong-value reads at
larger `max_num_seqs`. Given the retraction above, treat `glm53_max_seqs: 1`
as untrusted rather than known-fatal — it faulted once, then survived ten sizes
on retry.

Next step if resumed: clear the Triton cache and restart to make the launch cold
again, and re-run under `AMD_SERIALIZE_KERNEL=3` to confirm the fault is in that
kernel rather than asynchronously reported from an earlier one. Only file
separately once it reproduces.

### 1.6 M-invariance (batch-shape channel) — measured, and inactive here

Adapted from `tools/gemm_m_invariance.py` in
<https://github.com/jschmied/qwen38-flash-next-gb10>. The FP8 rows in that table
do not apply: this checkpoint is AWQ/compressed-tensors W4A16, which leaves
`self_attn.*` and the KDA projections in BF16 (hipBLASLt/rocBLAS, every decode
step) and quantises only the routed experts. Per-rank shapes at TP=2:

| path (gfx1151) | row 0 identical to M=1 | max abs diff |
|---|---|---|
| BF16 `kda in_proj_qkvbfg_a` 12576x4096 | M = 1…**64**; differs from 128 | 5.0e-01 |
| BF16 `kda o_proj` 4096x4096 | M = 1…**64**; differs from 128 | 1.25e-01 |
| BF16 `mla q_b` 3072x4096 | M = 1…**32**; differs from 64 | 5.0e-01 |
| int4 W4A16 Triton MoE, E=288 N=1024 K=4096 | **every M tested, 1…256** | **0** |

Two conclusions:

1. **The batch-shape channel is inactive at decode widths on ROCm.** BF16 is
   invariant across the whole decode and verification range and only switches
   kernel at M>=128 — markedly better than the sm120 rows upstream, where
   blockwise FP8 and BF16 cuBLAS both diverged from M=2. So the E == V != A
   pattern of [#54928](https://github.com/vllm-project/vllm/issues/54928) should
   not apply here, consistent with disabling MTP not restoring determinism.
2. **The int4 MoE is M-invariant even with per-M tuned tiles** that change
   `BLOCK_SIZE_K` and `SPLIT_K` across buckets. Divergence was expected and
   measured at zero.

Methodology note worth keeping: the first run of this silently used the **stock**
MoE config because `VLLM_TUNED_CONFIG_FOLDER` is not set in a standalone
process. Always confirm the log line `Using configuration from
.../E=288,N=1024,device_name=AMD_Radeon_8060S,dtype=int4_w4a16.json` before
trusting a result about the tuned tiles.

### 1.7 The divergence is in PREFILL — the tightest localisation so far

`prompt_logprobs` with `max_tokens=1` exposes the prompt forward pass on its
own, with nothing generated. Five byte-identical greedy requests, prefix caching
off, sequential, idle, MTP off, tbv_ar2 off:

| prompt | prefill (prompt logprob values) | first sampled token | decode, 64 tok |
|---:|---:|---:|---:|
| 740 tok (739 values) | **5 distinct / 5** | 2 / 5 | 4 / 5 |
| 5571 tok (5570 values) | **5 distinct / 5** | 2 / 5 | 5 / 5 |

**The prompt forward pass itself is nondeterministic.** Decode-side kernels are
not the origin; they can only compound it. This validates the *phase* named in
[#54521](https://github.com/vllm-project/vllm/issues/54521)'s title
("`persistent_topk` in prefill") even though the `indexer_budget` threshold part
did not hold on either platform.

The gradient across the columns matters: logprobs differ 5/5 while the argmax
survives more often (2/5 distinct first tokens). Small perturbations that flip
the ranking only at near-ties — which is exactly why this reads as occasional
silent corruption rather than obvious garbage, and why rigid structures like
tool-call syntax shatter while prose looks fine.

Also excluded for this measurement: chunking (740 tokens is one chunk at
`max_num_batched_tokens=4096`, 5571 is two; both diverge 5/5) and MTP (off).

**What is left, all on the prefill side:** the chunked **KDA/GDN scan** (34 of
45 layers here are linear-attention), the **sparse-indexer prefill path** (kpool
compress + top-k), and ragged prefill attention. GDN was one of the three causes
the upstream reporter had already fixed locally, which is suggestive given how
much of this model is GDN. Posted as
[issuecomment-5546997128](https://github.com/vllm-project/vllm/issues/54521#issuecomment-5546997128),
with a request for a lever to separate "indexer top-k" from "KDA scan" — forcing
the dense (non-top-k) indexer path, or a build with
[#55122](https://github.com/vllm-project/vllm/pull/55122) applied.

## 2. Performance work (done, and still valid)

Prefill improvements are unaffected by the correctness defects — they are
prefill-path changes that do not touch speculative decoding.

| | bring-up | now | |
|---|---:|---:|---|
| prefill | 151 tok/s | 284–334 tok/s | **~+100%** |
| TTFT, 2,841-token prompt | 18.05 s | 8.7 s | **−52%** |
| cold 128K prefill (extrapolated) | 13.9 min | ~7.1 min | |
| decode step | 359 ms | 200–231 ms | ~1.6x |

Four changes, each measured independently (detail in `PATCHES.md` §6, §8–§10):

| change | knob | effect |
|---|---|---|
| tuned fused-MoE tile configs | `host/moe-configs/` | decode step 359 → 225 ms; stock int4 path hardcodes `BLOCK_SIZE_K=32` (38 → 95 GB/s) |
| prefill chunk 512 → 4096 | `glm53_max_batched` | prefill 151 → 196 tok/s |
| unpin `NCCL_PROTO` (was `LL`) | `VSH_NCCL_PROTO` | prefill 196 → 277 tok/s |
| RCCL over sockets, not the RoCE rail | `transport` | prefill 271 → 305 tok/s |

**Caveat on decode throughput.** The 1.6x step-time gain is real, but the tok/s
figures were measured with MTP on. With MTP off for correctness, decode is
~4–5 tok/s rather than 10–20.

### 2.1 Negative results — do not re-run

- **MoE tile/occupancy tuning is exhausted.** A second sweep over
  `waves_per_eu` and `SPLIT_K` gained 1.04–1.18x in the kernel and nothing end
  to end. M=4 sits at ~95 GB/s against ~214 GB/s for the model's own BF16
  GEMVs. Closing that needs a different kernel (a hand int4 grouped-GEMV for
  the `M*topk` rows-over-experts shape), not different parameters.
- **CUDA graphs are blocked.** Rank 0 captures (11.43 GiB), rank 1 dies in a
  Triton launch hook with `SystemError` from `triton/knobs.py:432`. Even if
  fixed, 11.43 GiB/rank is unaffordable against ~18.6 GiB free.
- **RDMA is not the win the repo name implies.** `transport: tcp` measures the
  same as `hybrid` within noise, single-stream — a fresh rig can skip the tbv
  kernel-module build entirely. RDMA may still matter at concurrency, untested.

### 2.2 Largest remaining performance prize

The AWQ checkpoint leaves `self_attn.*` (including all 34 KDA layers) and
`layers.45.*` (the MTP block) in **BF16**: ~6.1 GB/rank of reads every step,
~48 ms, **24% of decode**, which would be ~7 ms at int4. Needs a re-quantised
checkpoint, not a serving change — cheapest route is asking `wtdcode` for a
variant that quantises `self_attn` and layer 45. See `PATCHES.md` §12.

---

## 3. Other findings worth keeping

### 3.1 Reasoning effort is `Max` by default and it is expensive

`chat_template.jinja` line 2 accepts `reasoning_effort` only as `'low'` or
`'high'`; **anything else, including unset and `'medium'`, becomes `Max`**. At
`Max` on a bare prompt the model produced 2048 tokens of pure reasoning with
zero content, and with no `max_tokens` cap ran **22 minutes / ~16,200 tokens
without terminating**.

| `reasoning_effort` | finish | total tokens | reasoning | content | runnable output |
|---|---|---:|---:|---:|---|
| default (`Max`) | `length` | 2048 | 2048 | 0 | no |
| **`low`** | `stop` | 1246 | **0** | 4310 ch | **yes** |
| `high` | `length` | 2048 | 2048 | 0 | no |

`low` disables thinking entirely and is 2.3x faster end to end. `high` is no
better than `Max`. This is **not** what broke the harnesses (with tools in the
request, reasoning collapses to ~74 tokens on its own) but it is a large,
free win for chat use.

- Client-side: `"chat_template_kwargs": {"reasoning_effort": "low"}`
- Server-side: `--default-chat-template-kwargs '{"reasoning_effort":"low"}'`
  (merges with request-level kwargs; request wins). **Not applied.**
- `thinking_token_budget` also works on this build (needs `--reasoning-parser`,
  which we have) and is a cleaner lever — it caps the think block and forces
  `</think>` rather than switching reasoning off. Verified: `256` → 74 reasoning
  tokens, valid tool call.

### 3.2 The response field is `reasoning`, not `reasoning_content`

On this build the think block comes back in `message.reasoning`, with the count
under `usage.completion_tokens_details.reasoning_tokens`. A client looking for
`reasoning_content` silently sees nothing. Note `vsh-manual-serve.sh` still has
a comment claiming `reasoning_content` — **that comment is wrong and should be
fixed.** Upstream has a related open issue:
[#54744](https://github.com/vllm-project/vllm/issues/54744) (parser gates on
`enable_thinking`/`thinking` kwargs the GLM-5.3 template never reads).

### 3.3 Documentation errors that were corrected

- Context labels in `PATCHES.md` §6–§10 were wrong: the decode harness sized
  prompts as `round(target / 230)` paragraphs when a paragraph is 71 tokens, so
  its "512 / 4k / 8k" prompts were really ~155 / ~1.3K / ~2.6K tokens.
  Before/after comparisons were unaffected; labels were fixed and the flatness
  claim re-measured at true token counts (flat 523 → 32,830).
- The Thunderbolt rail negotiates **20 Gbps**, not the 40 Gbps originally
  assumed (`ethtool thunderbolt0` → `Speed: 20000Mb/s`).
- `README.md` had gone stale, still claiming TTFT was "unchanged at ~18 s …
  dominated by the RCCL all-reduces (next target)" after that had been fixed.

---

## 4. Current machine state (local, uncommitted)

`~/vsh-config.yaml` on box1 differs from the repo template. Backups:
`~/vsh-config.yaml.bak-128k-8gib`, `~/vsh-config.yaml.bak-tbvon`.

| setting | current | repo template | why |
|---|---|---|---|
| `glm53_max_ctx` | 262144 | 131072 | user request; 256K prefill is ~16 min cold |
| `glm53_kv_bytes` | 16 GiB | 8 GiB | 1,183,680 tokens = 4.52x concurrency at 256K |
| `glm53_mtp_tokens` | **0** | 3 | correctness (1.2) |
| `glm53_tbv_ar2` | **0** | 1 | left off from the isolation test; **not implicated, safe to restore** |
| `glm53_prefix_cache` | 1 | (new knob) | restored after the determinism measurement |
| `glm53_log_requests` | 0 | (new knob) | logging is OFF; set 1 only while debugging, it writes full prompts to the journal |
| `glm53_max_seqs` | 256 | 256 | briefly set to 1 for a test; see the retraction in 1.5 |
| `transport` | hybrid | hybrid | |

> **Plumbing lesson, hit twice.** Env-gated knobs in `vsh-manual-serve.sh` do
> nothing unless the variable is also on the `ENVPASS` whitelist in
> `vsh-cluster-restart.sh` — that string is what carries variables into the
> container on both ranks. `VSH_GLM53_LOG_REQUESTS` and `VSH_NCCL_PROTO` were
> both gated but unforwarded, so setting them in the caller's shell silently did
> nothing; I spent a restart believing logging was off when it was not. Both are
> now forwarded and config-driven, so the state is reproducible from
> `~/vsh-config.yaml` alone. **Check `ENVPASS` when adding a knob.**

`~/vsh-manual-serve.sh` also carries three uncommitted local additions:

1. `--override-generation-config '{"temperature":0}'` — server-side default
   sampling temperature 0. Note this is a **default, not a clamp**: a client
   sending `"temperature": 0.7` still gets 0.7, and vLLM has no flag to force
   otherwise.
2. `--enable-log-requests --enable-log-outputs` — **still on**, and writing
   full prompt text to the journal. Set `VSH_GLM53_LOG_REQUESTS=0` and restart
   to turn it off.
3. Neither of these is in the repo, so `host/deploy.sh` would overwrite them.
   The `vsh-config.yaml` values survive, since `deploy.sh` does not install
   site config.

Memory headroom is thin at 256K/16 GiB: `MemAvailable` ~7.6 GiB on box1 and
~8.7 GiB on box2, down from 15.7/18.0 at 128K/8 GiB. Box1 was already touching
swap. Doubling `max_model_len` also doubled the sparse-indexer prefill
workspace (`max_model_len * 40 * 132 B`: 0.65 → 1.29 GiB/rank).

---

## 5. Where to pick this up

In priority order:

1. **Watch [vllm#54521](https://github.com/vllm-project/vllm/issues/54521).**
   Re-run `minrepro.py` after any upstream fix lands. If greedy becomes
   reproducible, re-test MTP (1.2) and long-context tool calling (1.3) — both
   may be downstream of it.
2. **Test the sparse-decode patch as a cause** (1.4, "still not excluded"). It
   is the only local patch left in the path. Compare against any alternative
   routing, and cross-reference
   [#50603](https://github.com/vllm-project/vllm/issues/50603).
3. **Apply and validate the chat-template mitigation** (1.3). Built and
   validated offline; needs one restart and a ladder re-run at 12k/25k/50k.
4. **Bisect the prefill path** (1.7) — this is now the live question. Separate
   the sparse-indexer top-k from the chunked KDA/GDN scan: force the dense
   (non-top-k) indexer path if a knob exists, and try a build with
   [#55122](https://github.com/vllm-project/vllm/pull/55122) applied. Upstream
   has been asked for the right lever.
5. **Cold-JIT test for the fault** (1.5) — clear the Triton cache, restart, and
   see whether `_deepgemm_fp8_paged_mqa_logits` faults on its first cold launch.
6. **Redo the MoE numerics verification end to end** once the engine is
   deterministic. The kernel-level check is solid, but the original end-to-end
   quality check was circular — it dismissed text divergence as ambient noise.
7. **Ask `wtdcode` for a checkpoint that quantises `self_attn` and layer 45**
   (2.2). Largest remaining performance prize, ~24% of decode.
8. Consider filing a **separate gfx1151 issue** for the long-context tool-call
   collapse (1.3) if it turns out not to be downstream of #54521.

## 6. Reproducers

All on box1 under `/tmp` (disposable — copy anywhere worth keeping):

| script | what it does |
|---|---|
| `minrepro.py` | minimal greedy-determinism check, stdlib only, non-zero exit on divergence |
| `det_sweep.py` | determinism vs prompt length, hashes per rep |
| `ladder.py` | tool-call success vs prompt length, unique non-repeating filler |
| `needle.py` | needle recall at length + the recency probe |
| `splitk_determinism.py` | kernel-level bitwise check of the tuned MoE configs |
| `gemm_m_invariance_gfx1151.py` | M-invariance of BF16 dense + int4 MoE (needs `VLLM_TUNED_CONFIG_FOLDER` set); upstreamed as `gemm_m_invariance_rocm.py` |
| `quality_probe.py` | five verifiable stems — **weak: uses raw /v1/completions on a chat model, see 9.5** |
| `prefill_vs_decode.py` | **the localiser** — `prompt_logprobs` to separate prefill from decode divergence |
| `fault_probe.py` | ascending prompt sizes, checking engine liveness after each, for the fault condition |
| `determinism.py` | 5x identical chat requests with tools, reports distinct outcomes |
| `ctx_sweep.py` | prefill tok/s and ms/step at real token counts |
| `prefill_rate.py`, `measure.py`, `ar_bench.py`, `tool_test.py` | the performance harnesses behind `PATCHES.md` §6–§10 |
---

## 7. Addendum — evening 2026-09-04: elimination sweep around the prefill divergence

Follow-up to §1.7 (divergence localises to prefill). Three of the four planned
discriminators ran; the findings below tighten the suspect set considerably.
Hunt scripts are in `/tmp/hunt-*.sh` and `/tmp/*determinism*.py` on box1
(disposable), with local copies under `D:\Projects\vllmds4\hunt\`.

### 7.1 Static sweep — no nondeterministic primitive anywhere in the prefill path

Every kernel family the prefill forward launches was read from the installed
tree in vllm-glm:

| kernel family | where | nondeterministic primitives |
|---|---|---|
| ragged sparse-MLA prefill attention | `v1/attention/ops/rocm_aiter_mla_sparse.py` (`_sparse_attn_prefill_ragged_kernel`) | none — online softmax, masked loads, `-1` slots guarded (`other=-1` → `valid = slot>=0` → `safe_slot=0` masked load, score → -inf, p → 0) |
| indexer: kpool compress + pool top-k + expansion | `models/glm5next/amd/ops/kpool_compress.py`, `layers/sparse_attn_indexer_kpool.py` | none (pure gather/arithmetic; no sorting kernels) |
| indexer: fp8 MQA logits | `aiter/ops/triton/attention/pa_mqa_logits.py` | none (no split-KV atomics; aiter's atomics live only in its fp4/mxfp4 MoE, unused here) |
| KDA chunked scan | `models/glm5next/amd/ops/third_party/kda/{kernels,fused_recurrent}.py` (via `kimi_gdn_linear_attn`, backend forced `"triton"` on ROCm) | none |
| GDN ops | `model_executor/layers/mamba/ops/*` | none |
| vllm distributed | `vllm/distributed/*` | none |
| tuned-MoE Triton kernel | (§1.4, earlier) | bitwise stable incl. `moe_sum` |
| `sparse_utils.py` atomic slot allocator | `v1/attention/backends/mla/sparse_utils.py:175,191` | **not in our path** — importers are FLASHMLA_SPARSE / FLASHINFER_* (NVIDIA) plus a JIT-warmup hook in `mla_attention.py` gated on those backend names |

Also verified at the code level:

- `topk_indices_buffer` is re-initialised to -1 for every token every step
  (ops file ~line 929) — no stale indices leak across requests.
- The persistent `paged_kv_indices` buffer is only read inside the ranges the
  convert kernel freshly wrote (indptr-defined); the ragged kernel walks
  indptr, never the buffer width.
- The `allreduce_rms` fusion is a torch.compile custom pass — **inert under
  `--enforce-eager`** (mode NONE never runs the pass manager), so the prefill
  collective is plain RCCL.

**Conclusion:** every kernel the prefill launches is deterministic by
construction (fixed reduction order, no atomics, no value-dependent
reordering), buffers are clean, everything is single-stream. The divergence
cannot originate *inside* any kernel we can read. Remaining candidates:
(a) cross-rank ordering inside RCCL itself, (b) a runtime-only effect invisible
statically (autotune selection varying, init-order effect), (c) a kernel not
yet identified.

### 7.2 BF16 dense GEMMs are run-to-run STABLE (closes a §1.6 gap)

The M-invariance harness answered *shape*-invariance, not run-to-run
reproducibility of the same call. Repeated the identical `torch.mm` 20× at the
real shapes (M=64/740/5571 × N=12576/4096/3072, K=4096, bf16): **1 distinct
hash / 20 at every shape, max drift 0.0** (`/tmp/gemm_repeat_determinism.py`,
standalone alongside the server).

### 7.3 `AMD_SERIALIZE_KERNEL=3` deadlocks TP=2 bring-up — do not use

Tried as the race discriminator (§5 old item 4's successor): with the var first
in `ENVPASS`, the serve hangs at the c10d/RCCL bootstrap — 30+ min no progress,
GPU spinning at 100%, VRAM at the desktop baseline, c10d socket warnings as the
last log lines. Host-synchronous serialization deadlocks the cross-rank init.
Aborted; the rig was restored afterwards. **Known-incompatible knob on this
stack.** (The torch-level `CUDA_LAUNCH_BLOCKING=1` remains untested at TP=2 and
may not share the deadlock — it synchronizes after launch instead of making the
runtime globally synchronous.)

### 7.4 TP=1 does not currently come up on this build

Attempted as a no-RCCL discriminator:

- ray + TP=1: the head-node Worker actor dies with "Worker unexpectedly exits
  with a connection error code 2, End of file" at engine init — twice,
  including on a freshly recreated container — then EngineCore falls back to a
  "local Ray instance" and idles. Unresolved; if TP=1 matters later, chase the
  worker's stderr under the ray actor first.
- mp + TP=1 via direct `podman exec`: no log output, API never up; the
  container entered a crash-loop-ish state during these attempts.

### 7.5 Ops incidents during the hunt (both fixed)

- A stray watcher from the 00:55 session, polling for
  `vsh-cluster-restart.sh` to finish, was armed to append
  `glm53_enforce_eager: 0` to `~/vsh-config.yaml` after the next restart —
  CUDA graphs, the known rank-1 crasher (§2.1/§11). It never fired: its
  `pgrep -f` pattern matched its own command line and the loop self-deadlocked.
  Killed; config verified. **Lesson: never arm one-shot config edits behind
  pgrep watchers.**
- The vllm-glm toolbox lost its supervisor during the force-cleanup (exec
  failed with "unable to find user ... no matching entries in passwd");
  recreated per the documented container-heal quirk. Triton/aiter caches live
  on the host home and survived; torch verified post-recreate.

### 7.6 Updated next steps (supersedes §5 item 4 for this thread)

1. **Offline two-pass prefill harness** (no server, no ray, no RCCL):
   `LLM(tp=1, enforce_eager, prefix-caching off)` in one process, the same
   740/5571-token prompt 5×, hash `prompt_logprobs`. Divergent ⇒ the defect is
   intra-rank and reproduces without the serving stack — then rerun with
   `CUDA_LAUNCH_BLOCKING=1` (torch-level serialization that cannot deadlock a
   single-rank run) to split race-vs-kernel. Deterministic ⇒ the cross-rank
   path is implicated.
2. If cross-rank: server-level A/B on the collective — transport rdma vs tcp
   with `NCCL_ALGO`/`NCCL_PROTO` pinned both ways (the §1.4 pin ran on one
   transport only), plus `RCCL_DEBUG=INFO` traces comparing the selected
   algorithm per size.
3. Cold-JIT fault test (§1.5) unchanged; rides along any cold restart.
4. §5.1 upstream watch stands; post today's eliminations as a follow-up
   comment on #54521 once the offline harness result is in.
---

## 8. Addendum 2 — evening 2026-09-04 (session 2): the hunt goes to the input layer

Continued from §7 the same evening. The discrimination chain ran live on the
rig; every step is reproducible from `/tmp/*.sh` (scripts kept, local copies
in `D:\Projects\vllmds4\hunt\`).

### 8.1 Race hypothesis: killed

`CUDA_LAUNCH_BLOCKING=1` (torch-level per-launch sync; unlike
`AMD_SERIALIZE_KERNEL=3` it does NOT deadlock TP=2 bring-up — API up in
~5.5 min): divergence **persists** — prefill logprobs still 5/5 distinct at
740 and 5571 tokens. Inter-kernel ordering is not the mechanism.

### 8.2 Allocator-page-reuse hypothesis: killed

`PYTORCH_NO_CUDA_MEMORY_CACHING=1` (every allocation a fresh, driver-zeroed
`cudaMalloc`): divergence **persists** (5/5 distinct). Not PyTorch allocator
reuse. (The KV pool / mamba state pools are persistent application tensors
anyway — this flag never covered them.)

### 8.3 Static sweep extended: everything clean

- `vllm/third_party/flash_linear_attention/`, `models/kimi_k3/**`, GDN ops,
  `vllm/distributed/`: zero atomics.
- `kimi_k3/nvidia/ops/cute_dsl/gemm_rs_ar.py` DOES have `cute.arch.atomic_add`
  — but it is SM100/NVLink-only (tcgen05, multimem, symm_mem), default
  `run_gemm_rs_ar: False`, and never runs on gfx1151.
- The `allreduce_rms` fusion is a torch.compile pass — inert under
  `--enforce-eager` (mode NONE never runs the pass manager).
- `gather_initial_states` honours `has_initial_state=False` (zeroes rows);
  `causal_conv1d_fn` gates the initial-state load per row
  (`if load_init_state:`). The KDA/conv state gates read correct.

### 8.4 TP=1: two independent blockers (documented, unresolved)

- Offline/served TP=1 OOMs on one APU: loader peak 119 GiB allocated against
  ~106 GiB usable (desktop resident). Single-APU TP=1 does not fit.
- ray + TP=1: head-node Worker actor dies at engine init
  ("connection error code 2, End of file"), twice — including on a freshly
  recreated container — then EngineCore falls back to a local Ray instance and
  idles. Unresolved; its own bug if TP=1 ever matters.

### 8.5 Ops incident: stray config-injection watcher

A leftover watcher from the 00:55 session was polling for
`vsh-cluster-restart.sh` to finish, armed to append `glm53_enforce_eager: 0`
(the known rank-1-crashing CUDA-graph knob) to `~/vsh-config.yaml`. It never
fired — its `pgrep -f` matched its own command line and the loop
self-deadlocked. Killed. Lesson stands: never arm one-shot config edits behind
pgrep watchers. (The vllm-glm toolbox also had to be recreated after the
force-cleanup; Triton/aiter caches live on the host home and survived.)

### 8.6 ~~BREAKTHROUGH 1 — corruption at the embedding input~~ — RETRACTED, tap off-by-one

`prompt_logprobs`-level probing was upgraded to per-layer and per-segment
taps (in-container hot-patches of `glm5next/nvidia/{model,kda,attention}.py`,
the MLA backend, and `VocabParallelEmbedding`; all reverted from `.orig`
afterwards — **note: in-container edits do NOT survive toolbox recreation**).

Token-id dump at `VocabParallelEmbedding.forward` for three byte-identical
requests in one process:

```
V ids n=886 head=[374, 2833, 291, 13057, 911, 264, 43333, 2038] ...
V ids n=886 head=[330, 2833, 291, 13057, 911, 264, 43333, 2038] ...
V ids n=886 head=[1096, 2833, 291, 13057, 911, 264, 43333, 2038] ...
```

The original reading was: *the first prompt token id differs on every
request, so the input itself is nondeterministic.*

> ⚠️ **RETRACTED on review (2026-09-05). This is an off-by-one in the tap, not
> a defect. The input is not corrupted.**
>
> Align the logged sequence against the real one. The tap logged
> `[374, 2833, 291, 13057, 911, 264, 43333, 2038]`; the engine's actual prompt
> ids for that prompt shape begin `[2833, 291, 13057, 911, 264, 43333, 2038,
> 3152]`. **Tap positions 1–7 are exactly true positions 0–6** — one extra
> leading element, with the real sequence intact behind it. The varying values
> (374 / 330 / 1096) are a stale slot *preceding* the request's range, and the
> "885-vs-886 scheduled counts" wobble is the same off-by-one in the slice
> length.
>
> Verified independently, with no hot-patching: `/v1/completions` with
> `return_token_ids: true`, 5 identical requests — **1 distinct id sequence /
> 5** (`sha 01455b72`, first id `2833`, `n=823` every time).
>
> The "fresh engine's first request is deterministic (3/3)" observation
> supports the artifact reading rather than the corruption one: a freshly
> zeroed buffer read out of range yields the *same* stale value every boot.
>
> Everything derived from this is withdrawn with it: position-0 logprob
> shifts, the harbor/harpoon flip attribution, and "defect (1)" below. **Do not
> post the token-id dump upstream** — see the removed §8.8 item 4.
>
> If the runner-side path is ever re-probed, fix the tap first: log
> `input_ids[pos : pos + n]` with `pos`/`n` taken from the same scheduler
> metadata the forward uses, and assert the first id against the frontend's
> `return_token_ids` output before drawing any conclusion.

### 8.7 Divergence with IDENTICAL inputs — bisection stands, mHC attribution does not

On one boot, two of the three requests had **byte-identical token ids
(`2f177003` twice) and byte-identical embeddings (`e995fed6` twice)**
(`embed_input_ids` method tap) — yet their full outputs still diverged.
Layer-entry taps localised it: entries 0–3 (KDA 0,1,2 + MLA 3) match;
**layer 4's entry differs** ⇒ the nondeterminism sits inside model layer 3's
post-attention segment: `o_proj` → `hc_fused_post_pre` (**the mHC fallback we
force onto gfx1151 via the TileLang-disable patch**) → MoE. The first MLA
layer's indexer top-k AND attention output matched.

**The bisection is the valuable part and it survives §8.6's retraction** — it
was measured on requests with byte-identical ids *and* embeddings, which we now
know is simply the normal case. It remains the tightest localisation we have:
the divergence appears between layer 3's entry and layer 4's entry.

> ⚠️ **The "prime suspect: mHC" attribution is not supported** (reviewed
> 2026-09-05). It rests on elimination, and the suspect was checked:
> `model_executor/kernels/mhc/{triton,aiter,tilelang}.py`, `kernels/mhc/
> __init__.py`, `_aiter_ops.py` and `layers/mhc.py` contain **zero**
> nondeterministic primitives (no `atomic*`, `scatter_add`, `index_add`,
> `scatter_reduce`, `index_put`). We are confirmed on the fallback
> (`HAS_TILELANG_MHC: False`; our patch adds `not on_gfx1151()`), so the right
> code was inspected. That makes mHC no better supported than every other
> kernel in §7.1's clean sweep.
>
> The elimination also leaned on `o_proj` and the MoE being stable, both tested
> at **decode** shapes while the divergence is in **prefill**. The one real gap
> there is now closed: the tuned MoE config uses `SPLIT_K>1` only in the M =
> 1 / 4 / 8 buckets; **every prefill-shape bucket (M = 16…512) is
> `SPLIT_K=1`**, so our tuned config introduces no split reduction in prefill.
>
> So the segment is localised but the culprit inside it is not. Re-verify the
> bisection with fresh taps (the previous ones did not survive the toolbox
> recreation, and one of them had the §8.6 off-by-one) before acting on it.

Note that with §8.6 withdrawn there is **one** defect under investigation here,
not two: a forward-pass nondeterminism somewhere in layer 3's post-attention
segment. The request input assembly is exonerated.

### 8.8 Updated next steps (supersedes §7.6)

1. **Re-verify the §8.7 bisection with correct taps.** This is the only live
   lead. Assert the tap's first token id against the frontend's
   `return_token_ids` output before trusting any per-layer dump — that check
   alone would have caught §8.6. Then re-confirm "layer 3 entry matches, layer
   4 entry differs" on the same boot.
2. **mHC A/B** (still worth running, as a *test* rather than a confirmation):
   restart with mHC disabled (hf-override `{"mhc": false}`; quality is wrong
   but determinism is the question). Deterministic ⇒ mHC despite the clean
   static read; divergent ⇒ mHC is excluded and the segment narrows to
   `o_proj`/MoE **at prefill shapes**, which is where neither has been
   repeat-tested. Repeat-test those two at M ≈ 823 first — it is cheaper than
   a restart and closes the gap the §8.7 elimination left open.
3. Re-test MTP (§1.2) and long-context tool calling (§1.3) **after** (2) —
   both may be downstream of this defect rather than #54521's kernels.
4. Upstream: the token-id dump is **withdrawn** and must not be posted (§8.6).
   Post the layer-level bisection instead, once re-verified — and note that
   #54521 has since moved on: `jahnclawdmonet` found a scale-layout bug in the
   reporter's M-invariance harness, the blockwise-FP8 row was withdrawn, and
   the script is now at v2. Use more repetitions than we have been: their 1080
   identical calls bound the per-call divergence rate at ~0.3% (95%), where our
   20-call tests bound it at ~14%.

## 9. Addendum 3 — 2026-09-05: the DS4 comparison

Asked of this: is the greedy nondeterminism a property of this rig, or of
GLM-5.3 / the 0.29 build? Answered halfway, and the other half is blocked for a
structural reason worth recording.

### 9.1 DS4 on its own stack IS deterministic — the rig is not the problem

DeepSeek-V4-Flash, same two Strix Halo boxes, same Thunderbolt fabric, TP=2,
`--enforce-eager`, **DSpark MTP on with 5 draft tokens**, via the deployed
`ds4-vllm` unit (vLLM `0.22.1rc1.dev499+g470229c37`, fp8 KV, disk KV transfer).
`det_any.py`, `temperature=0`, `seed=1234`, `ignore_eos`, 5 identical requests
per row:

| prompt_tokens | DS4 distinct/5 | GLM-5.3 (MTP off) |
|---:|---:|---:|
| 224 | **1** | 5 |
| 566 | **1** | 5 |
| 1098 | **1** | 5 |
| 2181 | **1** | 5 |
| 4328 | **1** | 5 |
| 8603 | 2 → **1 warmed** | 5 |
| 12878 | 2 → **1 warmed** | 5 |

The two long rows first showed 2 distinct with rep1 differing and reps 2-5
agreeing — the cold-path signature, since DS4 runs with prefix caching **on**
and the transition sits where prompts start filling a full cache block. A single
warmup request before the measured reps makes all five agree, confirming rep1
was the cold/uncached path rather than run-to-run nondeterminism. (Same
confound class as 1.1 — check it before calling anything nondeterministic.)

**The load-bearing conclusion: this hardware, fabric, ROCm stack and TP=2/RCCL
configuration are capable of bit-reproducible greedy decoding.** Any "gfx1151
over Thunderbolt cannot be deterministic" hypothesis is dead. GLM's defect is
fixable in principle.

It also makes DS4 the pragmatic serving choice while GLM is unresolved:
reproducible, ~15 tok/s (roughly 3x GLM at MTP-off), already deployed, up in
about two minutes.

### 9.2 DS4 on the 0.29 build — BLOCKED, and why

9.1 is not a controlled experiment: DS4 there runs vLLM 0.22 vs GLM's 0.29, fp8
vs bf16 KV, no KDA/GDN layers vs 34 of 45, a different mHC implementation, and
spec decode on vs off. The seven-version gap is the biggest confound — if the
defect arrived in vLLM between 0.22 and 0.29, DS4 would look clean whatever
model it ran.

The 0.29 build does register `DeepseekV4ForCausalLM`, and the DS4 HF checkpoint
(`~/models/DeepSeek-V4-Flash-0731-hf`, fp8 block-128, 43 layers) is on disk, so
running DS4 on the GLM build looked tractable. Wiring added for it, and worth
keeping:

- `vsh-cluster-restart.sh` now honours **`VSH_SERVE_SCRIPT`** to override the
  serve script, and forwards `VLLM_USE_V2_MODEL_RUNNER` to both ranks.
- `~/vsh-ds4-029-serve.sh` serves the DS4 checkpoint from the vllm-glm
  container with flags matched to the GLM determinism runs.

**It does not run.** Two failures, in order:

1. `AssertionError: DeepseekV4 fp8_ds_mla layout only supports fp8 kv-cache, got
   auto` — the DS4 MLA path **requires** fp8 KV; bf16 is not an option. Fixed by
   adding `--kv-cache-dtype fp8`.
2. `AssertionError: Unsupported dtype: torch.float8_e4m3fn`, raised from
   `aiter/utility/dtypes.py:55` (`torch_to_aiter_pybind`), repeated across the
   cluster. gfx1151 has `float8_e4m3fnuz`, not `float8_e4m3fn`, and 0.29's
   `kv-cache-dtype` choices contain no fnuz variant (`auto, float16, bfloat16,
   fp8, fp8_e4m3, fp8_e5m2, fp8_inc, fp8_ds_mla, nvfp4_ds_mla, turboquant_*,
   int4/int8/fp8_per_token_head, nvfp4*`). Retried with `VSH_GLM53_AITER=0` —
   same failure (note: that run did not echo the flag, so aiter-off was not
   confirmed applied; the DS4 MLA path may call aiter directly regardless of
   the vLLM gate).

So the blocker is structural: **DS4 on 0.29 needs fp8-fnuz support ported into
the DeepSeekV4 MLA path**, which is the same class of work as this repo's
`vsh-fp8-fnuz-mqa.patch` and is exactly what the ds4-vllm 0.22 stack carries its
own patch series for. Not a flag.

### 9.3 The tractable alternative: flip GLM's model runner instead

Swapping models on one build is blocked, so control the variable on GLM
instead. On ROCm 0.29, `ROCM_DEFAULT_MRV1_ARCHITECTURES` contains
`DeepseekV4ForCausalLM` but **not** `Glm5NextForConditionalGeneration`, so DS4
defaults to model-runner **V1** while GLM runs **V2** — confirmed in the log:
`Defaulting to V1 model runner on ROCm for model architectures:
DeepseekV4ForCausalLM`.

That makes the runner a live candidate that needs no new model bring-up: run
**GLM with `VLLM_USE_V2_MODEL_RUNNER=0`** (the forwarding is now in place) and
re-run `det_sweep.py`.

Interpretation, noting [#54924](https://github.com/vllm-project/vllm/issues/54924)
reports MRV1 producing *corrupted* generations for GLM-family models on ROCm
(GSM8K 91.6% → 14.9%):

- **deterministic on MRV1** ⇒ the runner is the channel. Highly informative even
  if MRV1 output is wrong, and a clean upstream story.
- **still divergent on MRV1** ⇒ the runner is excluded and the suspect set stays
  where 1.7 / 8.7 left it — GLM's own prefill path, layer 3's post-attention
  segment.

### 9.4 Ops note

Three self-inflicted breakages while wiring 9.2, all fixed, all the same class:

- `ENVPASS` forwarded `VLLM_USE_V2_MODEL_RUNNER=` unconditionally, i.e. an empty
  string, which makes vLLM throw `int('')` at config build. **This would have
  broken the next GLM restart too**, not just DS4.
- The repair went through an **unquoted heredoc over ssh**, so the shell expanded
  both the search and replacement strings and stripped the `VAR=` prefix,
  leaving a dangling `${VAR:-}` that would expand to a bare value inside an
  `export` list.
- Correct form is conditional expansion, now in place and verified on both
  branches: `${VLLM_USE_V2_MODEL_RUNNER:+VLLM_USE_V2_MODEL_RUNNER=$VLLM_USE_V2_MODEL_RUNNER}`.

**Rule: never edit shell files through an unquoted heredoc over ssh — write the
patch script locally and `scp` it.** And when adding a knob, check both that
`ENVPASS` forwards it and that it forwards *nothing* when unset.

### 9.5 Model runner EXCLUDED — GLM diverges identically on MRV1 and MRV2

The tractable substitute for the blocked 9.2 experiment (see 9.3): hold the
model and build fixed, flip the runner. GLM with `VLLM_USE_V2_MODEL_RUNNER=0`,
MTP off, `--enforce-eager`. **Correction (see 10.5): prefix caching was ON for this
sweep**, not off — `VSH_GLM53_*` env prefixes are overridden by the config
eval. The conclusion is unchanged (5/5 distinct either way; the cache-off
baseline is in 1.1) but the stated condition was wrong. Runner confirmed in the log by
which module loads — 14 hits on `gpu_model_runner` (the V1 path) and none on
`v1/worker/gpu/model_runner` (V2), the same discriminator
[#54924](https://github.com/vllm-project/vllm/issues/54924) uses.

| prompt_tokens | MRV2 distinct/5 | **MRV1 distinct/5** |
|---:|---:|---:|
| 244 | 5 | **5** |
| 614 | 5 | **5** |
| 1196 | 3 | **5** |
| 2373 | 5 | **5** |
| 4718 | 5 | **5** |
| 9705 | 5 | **5** |
| 14782 | 5 | **5** |

**The runner is not the channel.** Divergence is identical on both.

**Runner contrast verified per boot**, not assumed — the two boots really did
use different runners:

| boot | window | evidence in the journal |
|---|---|---|
| forced `VLLM_USE_V2_MODEL_RUNNER=0` | 09:27-09:39 | `gpu_model_runner.py` x5, no "Using V2" line |
| restored (arch default) | 09:39- | `Using V2 Model Runner` x2 (both ranks), no `gpu_model_runner.py` |

> ⚠️ **WITHDRAWN: "MRV1 also corrupts output."** The first version of this
> section reported `17 * 23 = ` → `529` on MRV1 and called it independent
> corroboration of #54924. **I ran that probe only on MRV1 and never ran the
> MRV2 control.** MRV2 scores **identically — 3/5, same two failures**
> (`17 * 23 = ` → `23 * 23 * 23 * ...`). So the probe does not distinguish the
> runners at all and says nothing about #54924.
>
> The failures are almost certainly my probe design rather than either runner:
> it drives `/v1/completions` with bare stems ("17 * 23 = ") on a
> chat/reasoning model, and degenerate looping is the expected behaviour for
> that shape. A quality comparison here needs the chat endpoint with the
> template applied, and a real eval rather than five hand-written stems.
>
> What survives: nothing is known about MRV1 output quality for
> `Glm5NextForConditionalGeneration` from this test. #54924 stands on its own
> evidence for `GlmMoeDsaForCausalLM`. Forcing MRV1 is still not advisable —
> GLM is not in `ROCM_DEFAULT_MRV1_ARCHITECTURES` and the default exists for a
> reason — but this session produced no evidence for that either way.

### 9.6 Consolidated exclusion list for the prefill divergence

Everything ruled out by measurement, across all sessions. Worth keeping in one
place so nothing is re-investigated:

| candidate | how it was excluded |
|---|---|
| MoE combine (`moe_sum`) | 20 calls bitwise identical; source-traced into the tested op (1.4) |
| MoE `SPLIT_K` (our tuning) | bitwise identical at real E=288; all prefill buckets are `SPLIT_K=1` (8.7) |
| Split-KV decode attention | zero `atomic_add`; fixed-order two-stage reduction (1.4) |
| Collective / RCCL | `ALGO=Ring` + `PROTO=Simple`, verified in both workers' `/proc/<pid>/environ` (1.4) |
| `NCCL_PROTO` unpinning (our change) | same test |
| Inter-kernel ordering / races | `CUDA_LAUNCH_BLOCKING=1`, divergence persists (8.1) |
| Allocator page reuse | `PYTORCH_NO_CUDA_MEMORY_CACHING=1`, persists (8.2) |
| Prefix caching | re-measured with `--no-enable-prefix-caching` (1.1) |
| Request input assembly | tap off-by-one; frontend ids identical 5/5 (8.6) |
| Batch-shape / M-dependence | BF16 invariant to M=64, int4 MoE to M=256 (1.6) |
| Hardware / fabric / ROCm / TP=2 | **DS4 is bit-reproducible on the same rig** (9.1) |
| Model runner (MRV1 vs MRV2) | identical divergence on both (9.6) |
| Every kernel in the prefill path | static sweep: no nondeterministic primitives anywhere (7.1, 8.3) |

> **SUPERSEDED by section 10.** The layer-3 divergence is root-caused: the MoE
> router's `torch.topk(..., sorted=False)` returns different expert *sets* run
> to run on ROCm. Two rows of the table above need amending:
> **the fused MoE is NOT excluded** — only its expert GEMM and combine were
> tested, with fixed `topk_ids`, so the selection layer was never exercised; and
> **DSA sparse attention IS now excluded in situ** (identical at every call,
> section 10.2). What remains open is the smaller layer-21/34 defect in
> section 10.3, not the layer-3 one.

Still not excluded: `vsh-mtp-ropefree-triton-sparse.patch` and
`vsh-fp8-fnuz-mqa.patch` (ours), though both are decode-path patches and the
divergence is in prefill, which makes them poor fits.

### 9.7 Current server state

GLM restored to its production config after the runner test: **MRV2** (the
arch default — do not force MRV1, see 9.5), prefix caching **on**, MTP **off**
(correctness, 1.2), request logging **off**, 256K ctx, 16 GiB KV, API :1235.

DS4 is one command away and is the deterministic option:
`./vllm-strix-halo.sh ds4 start` (API :1234). The two stacks are mutually
exclusive — they both claim all four GPUs.

> **Teardown bug worth fixing:** `./vllm-strix-halo.sh stop` reports success but
> leaves ray running — `gcs_server` keeps port 6379, which makes the *next*
> DS4 bring-up fail with `!! box1 ray start failed`. Workaround:
> `podman exec vllm-glm bash -lc "ray stop --force"` before switching stacks,
> then check `ss -ltn | grep 6379` is empty. Fix belongs in
> `vsh-cluster-down.sh`.

## 10. ROOT CAUSE (partial) — 2026-09-05: the MoE router's unsorted top-k

Layer-3 tap investigation, run from `HANDOFF-layer3-tap.md`; full report in
**`HANDOFF-layer3-result.md`** (read that for the raw tables). Verified
independently afterwards. **This root-causes the layer-3 divergence completely
and leaves a separate, smaller defect at layers 21/34.**

### 10.1 The cause, named

`vllm/model_executor/layers/fused_moe/router/grouped_topk_router.py`:

```python
# For batch invariance, use sorted=True to ensure deterministic expert selection
use_sorted = envs.VLLM_BATCH_INVARIANT          # default False
group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=use_sorted)[1]
...
topk_ids  = torch.topk(tmp_scores,  k=topk,       dim=-1, sorted=use_sorted)[1]
```

**On ROCm/gfx1151 `torch.topk(..., sorted=False)` returns a different top-k
_set_ run to run for bit-identical input** — not merely a different order.
Measured in-container on **synthetic** logits, M=740 / E=288 / groups=1,1 /
k=8 / sigmoid+bias, 20 identical calls:

> ⚠️ **This table does not reproduce on real router tensors — see 11.4.** The
> in-situ layer-3 result below stands; this op-level replay does not generalise,
> and must not lead an upstream report.

| arm | distinct raw /20 | distinct selected SETS /20 |
|---|---|---|
| in-situ `grouped_topk` (as deployed) | **20/20** | **20/20** |
| raw `torch.topk(sorted=False)` | **20/20** | **20/20** |
| raw `torch.topk(sorted=True)` | **1/20** | **1/20** |

So the MoE router picks different experts on every call. The fused
`ops.grouped_topk` branch is not taken here (`RocmPlatform.is_cuda()` is False,
and `VLLM_ROCM_USE_AITER_MOE=0`), so this Python path is the live one.

### 10.2 The in-situ chain, and why earlier exonerations missed it

Six hash points inside `Glm5NextDecoderLayer.forward`, layer 3 only, 8
identical prefill-only requests, **identical on both ranks at every point**:

| point | what | result |
|---|---|---|
| pt=1 | layer entry | = |
| pt=2 | after pre-attn `hc_fused_post_pre` (into self_attn) | = |
| pt=3 | **after `self.self_attn(...)`** (DSA attention out) | **=** |
| pt=4 | after post-attn `hc_fused_post_pre` (MoE input) | = |
| pt=5 | **after `self.mlp(...)`** (MoE out) | **X — 8/8 distinct** |
| pt=6 | at return | x mirrors pt=5; r/p/c all = |

Then: `router_logits` identical (8/8) but `select_experts` returns different
`topk_ids`/`topk_weights` (8/8) ⇒ **routing, not the expert GEMM**.

**Control arm:** with `sorted=True` forced, layer 3 became bit-deterministic at
every point. Mechanism confirmed end to end.

> **Why our own MoE exoneration (1.4, 8.7) missed this.**
> `moe_prefill_det.py` fed **fixed** `topk_ids`/`topk_weights` into
> `fused_experts_op`. It tested the expert GEMM and the combine; it never
> exercised the **selection** layer. The GEMM exoneration stands; the
> conclusion drawn from it ("MoE excluded") did not. **Lesson: an exoneration
> is only as wide as the inputs you varied.**

This also exonerates the DSA sparse attention in situ (pt=3 identical every
call) — **but only below `index_topk`**, which is where that measurement was
taken. Above the budget the DSA attention IS the source; see §12.3. The
original claim that "the component #54521 is titled after is clean on this
stack" was too broad and is corrected there.

### 10.3 What the fix does NOT cure — a second defect remains

With `sorted=True`, 24 identical forwards: layers 0-20 fully bit-identical
(including every DSA layer: 3, 7, 11, 15, 19), but the first divergence
**moves to layer 21** (also seen at 34) — both KDA+MoE layers — and the first
sampled token still flips (8/24 vs 16/24). Two effects, both measured:

1. **Exact-fp32-tie top-k flips survive `sorted=True`.** At layer 21 the router
   logits were identical in all 24 forwards, yet top-k took a second value in
   8/24. Standalone: with exact ties at the k-boundary,
   `torch.topk(sorted=True)` is nondeterministic 20/20; without ties, 0/20.
   **CONFIRMED — see 11.1.** Real layer-21 scores contain 2 exact
   k-boundary ties (rows 40, 549); the layer-5 control has none; op replay on
   the real tensors flips exactly row 40. The caveat is discharged.
2. **Combine-stage deviations — localised, NOT isolated.** In 3/24 forwards at
   layer 21 (2/24 at 34) the *final* MoE output deviated while the
   runner-internal `post`, top-k, logits and inputs were all bit-identical.
   That puts it between the runner's returned tensor and `Glm5NextMoE.forward`'s
   return: unfinalized-output materialisation, shared-expert combine,
   `routed_scaling`, or the TP combine of expert partials. Not attributed to an
   op. ~~**This is the open defect.**~~
   > ⚠️ **RETRACTED — see 11.2.** This was a rank0-only analyser plus a
   > TP-local partial mistaken for the final tensor. There is no combine-stage
   > defect in this data; the deviating forwards are rank1's independent tie
   > flips.

**Net: layer 3 is fully explained; end-to-end determinism is not achieved.**

### 10.4 `VLLM_BATCH_INVARIANT=1` is NOT a usable workaround

It fixes the router at op level (in-situ `grouped_topk` goes 20/20 → 1/20,
measured), but **GLM cannot boot with it**: it invalidates
`ROCM_AITER_MLA_SPARSE`, the only viable sparse-MLA backend here, and bring-up
dies. The rejection string is at `v1/attention/backend.py:322`
(`"batch invariance not supported"`).

So the only usable lever today is a **targeted patch** setting `use_sorted =
True` in `grouped_topk` (or gating it on platform rather than on
`VLLM_BATCH_INVARIANT`). That is a partial fix per 10.3 — worth it for layer 3,
not sufficient for reproducibility.

### 10.5 Ops correction: `VSH_GLM53_*` env prefixes are silently overridden

`vsh-cluster-restart.sh:22` runs `eval "$($HOME/vsh-config ...)"`, which
re-exports every yaml key **over** whatever the caller set. So
`VSH_GLM53_PREFIX_CACHE=0 ~/vsh-cluster-restart.sh` does nothing — the yaml is
the only effective lever for those keys. `VSH_TAP_FILE` and
`VLLM_USE_V2_MODEL_RUNNER` survive only because they are not yaml keys.

**Consequence for our own records:** the MRV1/MRV2 sweep in 9.5 was run with
prefix caching **ON**, not off as originally labelled. The conclusion is
unchanged — 5/5 distinct either way, and the cache-off baseline is independently
established in 1.1 — but the stated condition was wrong. Corrected in 9.5.

### 10.6 Where to go next

1. ~~Close the tie question~~ — **DONE: confirmed, see 11.1.** Real layer-21
   scores hold 2 exact k-boundary ties (rows 40, 549); the layer-5 control has
   none; op replay flips exactly row 40.
2. ~~Isolate the combine-stage defect~~ — **DONE: there is no such defect, see
   11.2.** It was a rank0-only analyser reading a TP-local partial. The combine
   stretch, TP all-reduce, shared-expert add and routed scaling are exonerated
   on real data at real shapes, so the `moe_sum` real-shaped control is no
   longer needed for this purpose.
3. **Upstream** (human decides): `grouped_topk`'s `sorted=False` default plus
   the tie race affects every noaux_tc router (DeepSeek/GLM family) on ROCm.
   The in-tree comment shows the hazard is known, but the gate is a flag that
   cannot be enabled on this platform.
4. Layer-3 fix is available as a one-line patch if determinism at that layer is
   worth a non-stock tree; note it does not deliver reproducible output.

### 10.7 Reproducers (all on box1)

| what | how |
|---|---|
| op-level top-k A/B (the decisive one) | `podman exec vllm-glm python3 /tmp/topk_det_test2.py` |
| per-layer bisection | `python3 /tmp/bisect_run.py` |
| six-point / router / runner taps | `/tmp/l3_tap_patch{,2,3}.py`, driver `/tmp/l3_run.py <N>` |
| sorted=True control | `/tmp/l3p4.py` |
| all-layer taps (rounds 5/6) | `/tmp/l3p5.py`, `/tmp/l3p6.py`, analysers `/tmp/l3a5.py`, `/tmp/l3a6.py` |
| raw logs | `~/tap-layer3/` on both boxes |

All patches were revert-verified in both containers; the previous session's
`.orig` files were left untouched.

## 11. 2026-09-05 (later): tie mechanism CONFIRMED, combine-stage defect RETRACTED

Two delegated investigations, both reported in full:
**`HANDOFF-tie-result.md`** and **`HANDOFF-combine-result.md`**. Their central
claims were spot-checked independently before being recorded here. Net effect:
**everything now traces to one mechanism**, and one of the two open defects
turned out not to exist.

### 11.1 The tie explanation is CONFIRMED (upgrades 10.3 item 1)

Real router score tensors dumped from live forwards (6 identical 740-token
prefill-only requests, `sorted=True` in situ, prefix caching off, warmed), then
checked for bitwise-exact fp32 ties at the top-8 boundary:

| layer | rows with exact k-boundary tie | op replay, 20 calls |
|---|---|---|
| 3 (anchor) | **0** / 740 | deterministic 1/20 |
| 5 (control) | **0** / 740 | deterministic 1/20 |
| **21** | **2** / 740 (rows 40, 549) | **2/20 distinct — the varying row is exactly row 40** |
| **34** | 1 in 3 of 4 upstream states, 0 in the 4th | tie state varies at row 651; tie-free state 1/20 |

Every tie-bearing tensor is nondeterministic **at a tie row**; every tie-free
tensor is deterministic. The ties are genuine fp32 rounding collisions, not
duplicated parameters — layer 21 row 40 is experts 12 and 202 with *different*
logits and *different* biases whose sums both round to `0x414467e2`. The
enabling structure is that `e_score_correction_bias` values cluster tightly
(three within 1 ULP at layer 34 row 651).

So "sorted=True is insufficient" is now **measured, not inferred**, and 10.3
item 1's caveat is discharged.

**Instrumentation cross-check that makes this trustworthy:** the layer-3
router-logits hash `bc3d628ff22b` was produced independently by two different
investigators with separately written taps and matches exactly; cross-rank
dumps are bitwise identical; the frontend `id0=2833` matches earlier
measurements; and the first-token split (' Which' 5/6 vs ' Then' 1/6)
reproduces the earlier 16/24 vs 8/24 bimodality.

> **A discarded round worth recording.** The first measurement round silently
> measured the *stock* arm: an env-gated `use_sorted` constant was read at
> **module import** inside the Ray workers, before the environment RPC
> delivered the variable. It was caught because layer-5 logits hashes came back
> 4-distinct, which is impossible under `sorted=True`. **Lesson: gate at
> forward time, not import time, and verify the arm behaviourally before
> trusting any number.**

### 11.2 The combine-stage defect is RETRACTED — it was a rank0-only analyser

10.3 item 2 reported that in 3/24 forwards the final MoE output deviated while
the runner-internal tensor, top-k, logits and inputs were all bit-identical.
That is logically impossible, and it was an analysis artifact:

- `/tmp/l3a6.py:85` calls only `analyse("rank0", ...)`. Its own predecessor
  `/tmp/l3a5.py` had **both** rank0 (line 101) and `analyse("rank1",
  read_lines(BOX2))` (line 103). The rank1 call was dropped in round 6.
  *(Verified directly.)*
- `moe2=post` hashes a **TP-local expert partial** — never rank-identical (0/24
  at all 42 layers). `moe=out` is the **all-reduced** result. So "identical
  intermediate" was identical *on rank0 only*.
- The `torch.topk` tie race resolves **independently per rank**. `out` is an
  exact function of both ranks' router states:

| rank0 topk | rank1 topk | layer-21 `moe=out` | n |
|---|---|---|---|
| steady | steady | `2d571580dc87` | 13 |
| flip | flip | `f7c49bee3edd` | 3 |
| flip | steady | `c347cae5337f` | 5 |
| **steady** | **flip** | `0fd4b33c31c3` | **3** |

Four keys, four outputs, zero ambiguity. The reported "3/24" is exactly the
last row — the forwards a rank0-only analyser cannot see. Across all 1008
(forward, MoE-layer) blocks `out` is a perfect function of (input, rank0
partial, rank1 partial), 0 ambiguous cases; under rank0 alone, 8 keys are
ambiguous.

Also found in the audit: `moe_runner.py:420` does `fused_output *= 2.5`
**after** the tap hashes it — a real in-place mutation, benign (constant scale)
but it means `moe2=post` and `moe=out` values are not directly comparable.

**Consequently exonerated on real data at real shapes:** the combine path, the
shared-expert add, the 2.5x routed scaling, and the TP all-reduce. The
"re-do `moe_sum` with real-shaped partials" control (10.6 item 2) is no longer
needed for this purpose.

Two further corrections to earlier write-ups: "the ranks are in lockstep at
every point" holds only for the *layer-level* taps, not the runner-internal
ones; and the layer-34 companion figure "2/24" does not reproduce from the
retained log (7/24 under the same modal definition, 2 ambiguous keys under the
functional test — the modal test is lineage-incoherent at layer 34).

### 11.3 One mechanism, and what a real fix requires

Both the layer-3 and the layer-21/34 divergence now reduce to a single op:
**`torch.topk(tmp_scores, k=8)` in `grouped_topk`, on exact fp32 k-boundary
ties, resolved independently on each TP rank.** With `topk_group=1` and
`n_group=1` on this model, that is the only nondeterministic op in the routing
path.

Router logits are bit-identical across ranks (layer 21: `9baf850ea3f8` on both,
24/24), so the divergence is entirely the per-rank tie race.

> **One claim in `HANDOFF-combine-result.md` §7 overstates.** It says no
> per-rank-deterministic fix "including a tie-break inside `torch.topk`" can
> make TP=2 reproducible. Since the ranks' inputs are bit-identical, a **fully**
> deterministic per-rank top-k *would* make them agree automatically — and its
> own escape clause ("or made a deterministic function of the score tensor")
> concedes this. The substantive point, which does hold: redundant per-rank
> selection **doubles the exposure** — two independent chances to flip per layer,
> which is why 11 of 24 forwards are affected at layer 21 rather than ~8 — and a
> *partial* fix such as `sorted=True` cannot help, because it only addresses
> non-tie rows.

Candidate fixes, in order of how convincing they are:

1. **A deterministic total order** in the selection: score, then expert index as
   the tie-break. Fixes it per rank and therefore across ranks.
2. **Compute the selection once and broadcast** `topk_ids`/`topk_weights`.
   Removes the redundancy as well as the race.
3. `sorted=True` alone — fixes layers 0-20 only. Necessary, not sufficient.

### 11.4 Caveat that weakens 10.1's headline measurement

The op-level table in 10.1 (`sorted=False` → 20/20 distinct) was measured on
**synthetic** logits. On **real** router tensors it does **not** reproduce:
standalone `sorted=False` behaved the same as `sorted=True` on every real
tensor tested, including tie-free layer 3.

So the in-situ every-call divergence at layer 3 under `sorted=False` — the
original layer-3 root cause — is **not** explained by ties, and is not
reproduced by a standalone replay either. That root cause rests on the in-situ
evidence (forcing `sorted=True` made layer 3 bit-deterministic, observed
twice), not on a demonstrated op-level mechanism. The mechanism is likely
engine-concurrency-dependent and remains unexplained.

**This matters for any upstream report:** do not lead with the synthetic 20/20
table. A maintainer running it on real router data will not reproduce it.

### 11.5 Where the investigation stands

**Explained:** layers 3 and 21/34, one mechanism, named op, with a concrete fix.
**Exonerated on real data:** the combine stretch, TP all-reduce, shared-expert
add, routed scaling — plus everything in the 9.6 table.
**Still unexplained:** why `sorted=False` diverges on *every* call in situ at a
tie-free layer (11.4). This is the last open question, and it is smaller than it
sounds — the fix direction in 11.3 does not depend on answering it.

Not attempted: nothing was re-run live for 11.2 (it was an analysis-only task by
design); the combine stretch is exonerated as a whole rather than op-by-op; the
specific flipped experts at the rank1-only forwards were not dumped.

Raw data: `~/tie-dump/` and `~/tie-dump-rank1/` (score tensors),
`~/tap-layer3/tap-round6-sortedTrue-24reps.rank{0,1}` (24-forward all-layer
logs, both ranks). Helper scripts on box1: `~/tie_tap_patch.py`, `~/tie_run.py`,
`~/tie_analyse.py`, `~/tie_followup.py`, `/tmp/l3a5.py`, `/tmp/l3a6.py`.

## 12. 2026-09-05 (evening): fix implemented and shipped; TWO bugs, cleanly separated

Supersedes §11.5's "one mechanism explains all of it". There are **two**
independent top-k bugs. One is ours and is now fixed; the other is upstream's
and is already being fixed there.

### 12.1 The fix

`container/patches/vsh-moe-router-deterministic-topk.patch`, documented in
`PATCHES.md` §14 and **wired into `container/Dockerfile`** (COPY, `git apply`,
`rm`) so it survives an image rebuild. It replaces the three
`torch.topk(..., sorted=use_sorted)` selection sites in `grouped_topk` with a
stable descending sort — a total order of *value descending, then expert index
ascending*, which is the convention vLLM's own fused CUDA kernel already uses.

Also applied live in both containers during validation. `git apply --check -p1
--whitespace=nowarn` (the Dockerfile's exact invocation) passes against stock
`da7b32ec80cd79291a2b58e3df8ccbbe`.

### 12.2 Validation

| check | result |
|---|---|
| op-level, patched vs stock control in one process | **1/30** vs **30/30** distinct — the test is sensitive, not merely passing |
| determinism sweep, 244 / 614 / 1196 / 1421 / 1797 tok | **1/5** (was 5/5) |
| per-layer taps, 740 tok, both ranks | **all 45 layers bit-identical** |
| quality | 4 chats correct and coherent; tool calling returns `tool_calls` with valid JSON, streaming and not |
| prefill | 340 / 328 / 317 tok/s at 2K / 8K / 32K — inside the 284-334 baseline |
| tie-break direction | row 40 → expert 12 (not 202), row 549 → 243, L34 row 651 → 58 — ascending index, as specified |

Independently re-verified afterwards on the restored production server:
614-token prompt, 32 generated tokens, **1 distinct / 5**.

### 12.3 The second bug — DSA indexer top-k, above `index_topk` only

The fix does **not** make the platform reproducible. Above `index_topk = 2048`
divergence returns, and a six-point tap inside layer 3 (both ranks, one boot,
below- and above-threshold groups measured together) isolates it:

| tap point | 719 tok (below) | 2497 tok (above) |
|---|---|---|
| layer entry | 1 / 5 | 1 / 6 |
| after pre-attention norm (into `self_attn`) | 1 / 5 | 1 / 6 |
| **after `self.self_attn(...)` — DSA attention** | **1 / 5** | **6 / 6** |
| after post-attention norm | 1 / 5 | 6 / 6 (cascade) |
| after `self.mlp(...)` — MoE | 1 / 5 | 6 / 6 (cascade) |
| at return | 1 / 5 | 6 / 6 (cascade) |

The divergence **enters at the attention output with its input bit-identical**,
so this is the origin, not propagation, and the MoE is clean post-fix. It
engages only once sequence length exceeds the indexer's top-k budget.

**This vindicates the upstream reporter.** #54521 is titled *"non-deterministic
from persistent_topk in prefill when prompt length nears indexer_budget"* —
which is exactly this, and PR #55122 fixes that op. Our earlier data appeared
to contradict them only because every measurement was taken **below** the
budget. §10.2's "DSA sparse attention is exonerated in situ" must be read as
*below `index_topk` only* — corrected there.

Threshold behaviour: below 2048, deterministic at 244/614/1196/1421/1797. Above
it, the layer diverges on **every** call, while *output-level* divergence is
intermittent (2146, 2434, 2848, 3302 flip; 2310, 2560 do not) — a perturbed
attention output does not always change the sampled argmax. Layer-level
measurement is the sharper instrument; output-level intermittency is argmax
robustness, not an intermittent defect.

### 12.4 Upstream package

In `upstream/`: `PR-BODY.md` (end-to-end numbers filled in; only the DCO
`Signed-off-by` remains) and `test_grouped_topk_determinism.py` — three tests
with a *constructed* bitwise tie, deliberately **not** CUDA-gated because the
Python fallback is the code under test. Not a duplicate of anything open as of
2026-09-05; the bug is also **not ROCm-only** (CUDA takes the same Python path
when `e_score_correction_bias is None` or the aiter flag is off).

`fused_topk_bias_router.py` carries the same hazard and was patched in the
validated build, but is excluded from the shipped patch and the PR: unreachable
for GLM-5.3, no behavioural coverage, and it holds a further unanalysed `topk`.

### 12.5 What is still open

1. **The DSA indexer top-k** (12.3) — upstream's, tracked at #54521 / #55122.
   Worth testing #55122 here once it lands; that is the last piece for full
   reproducibility on this rig.
2. **Post the router finding to #54521.** It is the missing explanation for the
   sub-budget divergence that thread never accounted for, and it confirms their
   diagnosis for the above-budget regime. Not yet posted.
3. The stale upstream draft at `/tmp/upstream_draft.md` predates all of this and
   should be discarded in favour of `upstream/PR-BODY.md`.
4. Unchanged from before: MTP stays off (§1.2), and the long-context tool-call
   collapse (§1.3) is a *separate* problem with an unapplied chat-template
   mitigation — determinism work does not address it.
