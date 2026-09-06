# TASK: tap inside GLM-5.3 decoder layer 3 to isolate the nondeterminism source

## Your job in one sentence

GLM-5.3-Flash on this 2-box gfx1151 rig produces **different greedy output for
byte-identical input**. The divergence has been localised to **layer 3's output
`hidden_states`**. Layer 3 contains exactly two unexcluded components. Tap
*inside* layer 3 and report which sub-module's output first differs.

Write your findings to **`~/vllm-strix-halo/HANDOFF-layer3-result.md`** on box1
(`davidcanar@192.168.0.102`). Raw logs to `~/tap-layer3/`. Someone will read
that file; nothing else you produce will be seen.

## Environment

- box1 `davidcanar@192.168.0.102` (has GitHub access); box2 reachable from it as
  `davidcanar@10.0.2.2`. TP=2 over Thunderbolt, rank 0 on box1, **rank 1 on box2**.
- Container `vllm-glm` on **both** boxes, vLLM `0.29.0.dev0+git.8bf39632`,
  ROCm 10.0, torch 2.11, gfx1151.
- GLM API on `:1235`. Start/stop: `cd ~/vllm-strix-halo && ./vllm-strix-halo.sh
  start|stop`. Bring-up takes ~4-6 min.
- Read `~/vllm-strix-halo/PENDINGWORK.md` first — sections 1.7, 8.7, 9.6 and 9.7
  are the relevant ones. It is long; the exclusion table in 9.6 will save you
  hours.

## What is already established — do NOT re-investigate

Divergence is real: 5 byte-identical `temperature=0` `/v1/completions` requests
return 5 different completions, at every prompt length from 244 tokens up,
sequentially on an idle server with prefix caching **off**.

Already excluded by measurement (details and methods in PENDINGWORK 9.6):

- MoE combine (`moe_sum`), and the fused MoE kernel itself — **20/20 bitwise
  identical at M=4/64/512/740/2048** with the deployed tuned config loaded.
- Dense BF16 GEMMs (`o_proj` etc.) — 20/20 bitwise identical.
- KDA / linear attention — layers 0,1,2 are KDA and are **bit-identical**.
- Dense MLP — layers 0,1,2 use dense MLP (`first_k_dense_replace: 3`) and are
  bit-identical.
- mHC post/pre — `post` and `comb` are **identical at layer 3**, and layers 0-2
  carry mHC too. (mHC was the previous prime suspect. It is not supported.)
- The collective (RCCL, `ALGO=Ring` + `PROTO=Simple` verified in both workers'
  `/proc/<pid>/environ`), inter-kernel ordering (`CUDA_LAUNCH_BLOCKING=1`),
  allocator reuse (`PYTORCH_NO_CUDA_MEMORY_CACHING=1`), prefix caching, request
  input assembly, batch-shape/M-dependence, the model runner (MRV1 and MRV2
  diverge identically), and the hardware itself (DeepSeek-V4-Flash is
  bit-reproducible on this same rig).
- Static reads found **no** nondeterministic primitive (no atomics,
  `scatter_add`, `index_add`, `scatter_reduce`) anywhere in the prefill path,
  including all mHC kernel files.

## The current localisation (reproduce this first as a sanity check)

Taps are **already installed** in
`vllm/models/glm5next/nvidia/model.py` in both containers, env-gated on
`VSH_TAP_FILE` (unset => byte-identical to stock). They hash all four tensors
the layer loop carries after every layer.

Result, both ranks agreeing exactly, 3 identical 740-token prefill-only requests:

| layer | hidden | residual | post | comb |
|---|---|---|---|---|
| 0,1,2 | = | = | = | = |
| **3** | **X** | = | = | = |
| 4+ | X | X | X | X |

Input embeddings were identical across all three (`embed=9b0d90d30511`).

Reproduce with: `python3 /tmp/bisect_run.py` on box1 (needs the server up with
`VSH_TAP_FILE` set — see "How to run" below). If you cannot reproduce this,
stop and say so in the result file; everything below assumes it holds.

## What layer 3 is, and the two remaining suspects

From the checkpoint config: `layer_types[3] = "deepseek_sparse_attention"`,
`full_attn_layers: [3, 7, 11, ...]`, `first_k_dense_replace: 3`. So layer 3 is
**both** the first DSA sparse-attention layer **and** the first MoE layer. The
MoE kernel is already exonerated standalone, but not *in situ* with real routing.

Remaining candidates, in order of prior probability:

1. **DSA sparse attention** — the indexer (kpool compress + top-k, `index_topk:
   2048`, `index_kpool: 4`) and the sparse MQA. This is what upstream
   vllm#54521 is titled about (`persistent_topk` in prefill).
2. **The MoE in situ** — exonerated standalone, but real routing / expert
   distribution differs from the synthetic test.

## The actual task

`Glm5NextDecoderLayer.forward` in
`/opt/venv/lib/python3.12/site-packages/vllm/models/glm5next/nvidia/model.py`
has this structure (mHC branch, which is the live one):

```python
    residual = x
    ...
    residual, post, comb, x = self.hc_fused_post_pre(...)   # pre-attention
    x = self.self_attn(positions=..., hidden_states=x, ...)  # <-- DSA attention
    residual, post, comb, x = self.hc_fused_post_pre(...)   # post-attn + pre-FFN
    if self._mlp_is_moe:
        x = self.mlp(x, already_sequence_parallel=self.is_sequence_parallel)
    else:
        x = self.mlp(x)
    ...
    return x, residual, post, comb
```

Add taps **only when `self.layer_idx == 3`** (keeps the log small and avoids
perturbing 45 layers), hashing `x` at each of these six points:

1. layer entry (`hidden_states` as received)
2. after the pre-attention `hc_fused_post_pre`, i.e. the tensor fed to `self_attn`
3. **after `self.self_attn(...)`** — the discriminating one
4. after the post-attention `hc_fused_post_pre`
5. **after `self.mlp(...)`** — the other discriminating one
6. at return

Then run 3+ identical prefill-only requests and report **which point is the
first to differ across runs**.

- If (3) differs => DSA sparse attention. Go deeper: tap inside
  `self_attn.forward` to separate the indexer top-k indices from the MQA output.
  The indexer lives in `vllm/model_executor/layers/sparse_attn_indexer_kpool.py`
  and `vllm/models/glm5next/amd/ops/kpool_compress.py`; the backend is
  `vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py`.
- If (5) differs while (3) matches => the MoE in situ, despite the standalone
  result. Dump `topk_ids`/`topk_weights` and check whether **routing** differs
  run to run (that would make the router, not the expert GEMM, the culprit).
- If neither differs but the layer output does => the divergence is in the mHC
  post/pre or the residual arithmetic after (5); tap the return values.

## Tooling already in place — reuse it

- `/tmp/tap_patch.py` (also inside both containers at `/tmp/`) — `apply` /
  `revert`. Backs up to `.taporig`. **Do not touch the `.orig` files**, they are
  a previous session's backups.
- Helper functions available in the patched `model.py`: `_VSH_TAP_FILE`,
  `_vsh_tap(msg)`, `_vsh_hash(tensor)`, `_vsh_rank()`. `_vsh_hash` returns a
  12-char sha256 of the fp32 CPU bytes and handles `None`.
- `~/vsh-cluster-restart.sh` forwards `VSH_TAP_FILE` to both ranks already
  (conditional expansion — adds nothing when unset).
- `/tmp/bisect_run.py` — driver + analyser for the per-layer version. Adapt it.
- `/tmp/det_sweep.py` — the determinism sweep, for confirming state.

### How to run

```bash
# both containers must be patched -- rank 1 is on box2
podman cp /tmp/tap_patch.py vllm-glm:/tmp/ && podman exec -i vllm-glm python3 /tmp/tap_patch.py apply
scp /tmp/tap_patch.py davidcanar@10.0.2.2:/tmp/ && ssh davidcanar@10.0.2.2 \
  "podman cp /tmp/tap_patch.py vllm-glm:/tmp/ && podman exec -i vllm-glm python3 /tmp/tap_patch.py apply"

cd ~/vllm-strix-halo && ./vllm-strix-halo.sh stop
VSH_TAP_FILE=/home/davidcanar/tap.log VSH_GLM53_PREFIX_CACHE=0 \
  setsid nohup ~/vsh-cluster-restart.sh > /tmp/restart.log 2>&1 &
# wait for :1235 to answer, then drive requests; tap output lands at
# ~/tap.log.rank<N> on the box that rank runs on (box2 has its own file)
```

## Methodological guards — these were all learned the hard way here

Two previous "breakthroughs" in PENDINGWORK were retracted. Both were
instrumentation errors, not model behaviour. Please do not add a third.

1. **Assert your tap against ground truth before believing it.** A previous tap
   read one element *before* the sequence start and "showed" the first prompt
   token id changing every request. It was an off-by-one; the ids are identical.
   Cross-check any id/tensor tap against `/v1/completions` with
   `"return_token_ids": true`. Note `input_ids` is `None` in this code path —
   the wrapper passes `inputs_embeds` — so hash the embedding, not ids.
2. **Always run the control arm.** A quality claim about the model runner was
   withdrawn because the probe was run on MRV1 and never on MRV2 (they score
   identically). If you claim X causes Y, measure without X.
3. **Prefix caching must be OFF** (`VSH_GLM53_PREFIX_CACHE=0`) for any
   determinism measurement. Repeated identical prompts otherwise hit cached KV
   and reps 2..N take a different path from rep 1. This produced a false
   "nondeterministic" reading on DeepSeek-V4.
4. **Warm the engine.** The first request after boot can differ from subsequent
   ones (cold JIT). Send a warmup request before the measured reps.
5. **Patch BOTH containers.** rank 1 runs on box2. A box1-only patch gives you
   half the picture silently.
6. **Never edit shell files through an unquoted heredoc over ssh.** The shell
   expands your search and replacement strings and mangles the file. Write the
   patch script locally and `scp` it.
7. **`pkill -f` over ssh will kill your own session** if the pattern appears
   anywhere in your command line — including inside an unrelated `pgrep` in the
   same command. Prefer explicit PIDs.
8. **Do not set `glm53_max_seqs: 1`** — it once hard-faulted the engine in
   `_deepgemm_fp8_paged_mqa_logits` (not reproducible since, see PENDINGWORK 1.5).
9. **Do not force `VLLM_USE_V2_MODEL_RUNNER=0`** for anything but a runner test;
   it is not the default for this architecture for a reason.
10. **`AMD_SERIALIZE_KERNEL=3` deadlocks TP=2 bring-up.** Use
    `CUDA_LAUNCH_BLOCKING=1` instead if you need launch serialization.

## Constraints

- **Do not post anything to GitHub** (vllm#54521 or elsewhere). Report to the
  result file; a human decides what goes upstream.
- **Do not `git commit`.** `PATCHES.md`, `README.md`, `PENDINGWORK.md` and
  `vllm-strix-halo.sh` have uncommitted changes that must stay uncommitted.
- **Revert your taps when done** (`tap_patch.py revert` in both containers) and
  **leave the server running** in its normal config: `./vllm-strix-halo.sh start`
  with no `VSH_TAP_FILE` and prefix caching back on (it is `glm53_prefix_cache: 1`
  in `~/vsh-config.yaml`). Confirm `:1235` answers before you finish.
- Switching to the DS4 stack is not needed for this task. If you do, note that
  the two stacks are mutually exclusive and `./vllm-strix-halo.sh stop` now
  tears ray down properly (it did not before; see PENDINGWORK 9.7).

## What to put in `~/vllm-strix-halo/HANDOFF-layer3-result.md`

1. Whether you reproduced the layer-3 localisation (table above). If not, stop
   there and say so.
2. The six-point table: which tap points matched and which differed, across how
   many identical requests, for **both ranks**.
3. The first differing point, and what that implicates.
4. If you went deeper (indexer vs MQA, or routing vs expert GEMM): the same
   treatment — what you tapped, what differed, what it implicates.
5. **Anything you could not exclude, stated plainly**, and any control you did
   not run. Under-claiming is much more useful here than over-claiming.
6. Exact commands and file paths for anything you want reproduced, plus where
   you left the raw logs.
7. Server state when you finished.
