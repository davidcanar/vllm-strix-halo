# Handover: GLM-5.3-Flash tool-call degeneration on the 2× Strix Halo rig

> **Resolved 2026-10-03 (PATCHES.md §27, §27.1).** The degeneration was neither
> the model nor the AWQ checkpoint. Four stacked sparse-attention bugs left the
> model unable to read anything past ~4K tokens, and opencode's tool-format
> instruction sits ~5K tokens into its prompt. With the fixes, the captured
> opencode request passes 8/8 (re-checked on 2026-10-06, §44). Kept as history.
> The rig details below are dated: production now decodes with DFlash2 k=3
> under PIECEWISE CUDA graphs, not MTP k=3 eager.

**Audience:** an engineer/agent with SSH access to the two boxes. Everything below was measured on
this rig; commands are copy-pasteable. Read PATCHES.md §25–§26 for the long form.

---

## 0. The rig, in one paragraph

Two AMD Strix Halo boxes (gfx1151, 128 GB unified each), `box1 = 192.168.0.102` (ray head, serves
the API on **:1234**, container **`vllm-glm`**), `box2 = 10.0.2.2` (ray worker, reachable from box1
by `ssh 10.0.2.2`). TP=2 over a Thunderbolt-4/USB4 cable via the OdinLink driver (`odl_tb5`;
`odl_ar2` decode all-reduce, `odl_mq` control plane, RCCL net plugin for prefill collectives).
vLLM pinned at `73859fec` (0.31.0.dev0, ROCm 10.0 / torch 2.11) inside the container; site-packages
live at `/opt/venv/lib/python3.12/site-packages` **inside** the container — always
`podman exec vllm-glm …` for those paths. Model: `GLM-5.3-Flash-AWQ-W4A16` (int4 routed experts;
KDA/MLA/`lm_head` in bf16) at `/home/davidcanar/models/GLM-5.3-Flash-AWQ-W4A16`. Serving flags of
interest: `--enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm47`,
`--max-model-len 262144`, `--max-num-batched-tokens 8192`, MTP speculative decoding k=3.

Production config lives in `~/vsh-config.yaml` (flat keys → `VSH_*`), env in
`~/vsh-cluster-env.odl.sh` (must be byte-identical on both boxes), launch via
`bash ~/vsh-cluster-restart.sh` (~5–8 min to API 200). The repo is `~/vllm-strix-halo`
(patch scripts in `container/patches/`, byte-exact patched files in `container/pinned-vllm/`,
measurement harnesses in `scripts/`, history in `PATCHES.md`).

Client under test: **opencode 1.18.18** (`~/.opencode/bin/opencode` on box1), configured in
`~/.config/opencode/opencode.jsonc` with a `strix-halo` provider pointing at
`http://127.0.0.1:1234/v1`, model `glm-5.3-flash`, `limit: {context: 262144, output: 131072}`.

---

## 1. Issue A — agent turns silently produced nothing **(FIXED, verify and keep)**

### Symptom

Asking opencode to "create a small snake game" produced one sentence of assistant text and **no
file**. Server-side, roughly **three turns in five** ended with `finish_reason=stop`, **empty
content, no `tool_calls`**, and between 100 and **12 795** generated tokens that appeared in **no
field of the response**. A verbatim replay of the same request succeeded the rest of the time
(one `write` call, 3 240 chars of valid JSON, the full HTML).

### How it was reproduced

A logging reverse proxy between opencode and the server captured the real request body
(`~/proxy.py`, listens on **1235**, forwards to 1234, bodies to `~/proxy_capture/NNN.json` +
`index.txt`). Point opencode's provider `baseURL` at `http://127.0.0.1:1235/v1`, run a prompt, then

```bash
cd ~ && python3 replay.py          # replay the captured request verbatim, dump finish/usage/calls
python3 replay_n.py 6              # same request N times -> failure rate
```

Captured facts worth knowing: opencode sends **`stop=None`, `max_tokens=32000`, `stream=True`, 10
tools, 2 messages** (7 055 prompt tokens, a 9 558-char system prompt). So the output budget was
never the cause.

### Root cause of the *loss*

The GLM tool parser buffers everything from the tool-call opener onward and, when the turn ends
without a complete, well-formed call, **discards the buffer**. Nothing is surfaced: not content,
not reasoning, not `tool_calls`. The reasoning *before* the block is all the client sees.

### Fix (shipped, live)

`container/patches/vsh-toolcall-failopen.py` → applied to
`vllm/parser/abstract_parser.py`. It tracks the raw stream text and whether a tool call ever
reached the client, and at the existing end-of-turn hook
`DelegatingParser.finalize_generation()` emits the text from `<tool_call>` onward as **content**
when no call was surfaced. Fail-open: a correct call short-circuits on the flag and is untouched.
Gate `VSH_TOOLCALL_FAILOPEN` (default **on**; `=0` reverts).

Measured after the fix, same 6 replays: 4 salvages, each recovering the file the model wrote
(2 805–4 348 chars of HTML as content), with matching journal receipts:

```bash
journalctl --user -u vsh-glm-manual -o cat --since '-15min' | grep -c vsh-toolcall-failopen
```

There is also log-only instrumentation, `container/patches/vsh-toolcall-drop-log.py`
(`VSH_TOOLCALL_DROP_LOG=1`, dir `VSH_TOOLCALL_DROP_DIR`, default `/tmp/vsh-toolcall-drops`),
which dumps text the parser was about to discard.

### What to check first

1. `podman exec vllm-glm grep -c vsh-toolcall-failopen \
   /opt/venv/lib/python3.12/site-packages/vllm/parser/abstract_parser.py` → expect **6**.
2. Re-run `python3 replay_n.py 6`: you should never see `content_chars=0` on a `stop` that
   generated thousands of tokens.
3. The upstream equivalent is **vLLM PR #55759** ("Extract tool calls emitted within reasoning
   blocks", fixes issue #39056, **open/unmerged** as of writing) — it patches only the
   *non-streaming* `parse()` path, which is not the path that broke here. Consider porting it
   anyway for completeness.

---

## 2. Issue B — the model mangles the tool-call format **(OPEN — this is the real problem)**

### Symptom

Only ~**30–40 %** of turns produce a usable tool call. The reasoning text is always coherent; the
*format* collapses. Collected samples (opencode prompt, trivial user request "create hello.txt
containing the word hello"):

```
<tool_call>write</arg_key><arg_value>content</arg_key><arg_value>hello\n</arg_value><arg_key>filePath</arg_key>
<tool_call>bash$$\n<parameter=name>bash</parameter>\n<command>echo hello > /tmp/snake-test/…
<tool_call>writeXML\u200f=null_file\rplaceholderscript>
write("/tmp/snake-test/hello.txt", "hello")
bash -c 'echo hello > hello.txt' cwd=/tmp/snake-test
```bash\necho "hello" > /tmp/snake-test/hello.txt\n```
<tool_call>write# Write hello.txt\n\nvoid main() {\n    print("hello");\n}
<tool_call>Write借助内容/file</think>文件已创建：hello.txt 包含单词 "hello"
<tool_call>write|null|file_path: …|content: hello
```

i.e. **missing opening `<arg_key>` tags**, invented dialects (Python call, C function, shell
one-liner, markdown fence, `<parameter=name>` XML, `key=value` and `|`-separated), mixed scripts
(Chinese/Japanese fragments), hallucinated parameter names (`diffprotected`, `robberNone`,
`writeJson5`), out-of-order `</think>`, and false success claims with no call at all.

### What has already been tested and ruled out

Each row is a measured arm on the **same** captured request, ~4–6 repetitions:

| hypothesis | experiment | result |
|---|---|---|
| output budget truncates the call | inspected the captured body | `max_tokens=32000`, `stop=None` — **not it** |
| the reasoning span swallows it | `chat_template_kwargs: {"thinking": false}` | unchanged (2/6) |
| too many tools overload the model | 2 tools vs 10 tools (`tool_count.py`) | 0/4 vs 2/4 — not the lever |
| spec-decode depth corrupts output | live `{"force": 1}` in `~/vsh-adaptive-k.json`, verified **1.00** draft tokens/step | **worse**: 1/6 usable |
| thinking effort | `reasoning_effort` low / high (`effort_test.py`) | 1/5 / 2/5 |
| the prompt never states the format | rendered the checkpoint's own `chat_template.jinja` (`render_prompt.py`) | it states it exactly: `<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key><arg_value>{arg-value-1}</arg_value>…` |
| the template was substituted/patched | file mtimes in the model dir | `chat_template.jinja` is the checkpoint's own (Sep 27 15:08); only `tokenizer_config.json` was edited at bring-up (Sep 27 15:59) to point vLLM at that file (its inline `chat_template` is empty) |

Conclusion so far: **a checkpoint-level deficiency in emitting this tool format.** At temperature 0,
with the format spelled out in its own prompt, the model still substitutes other dialects. Nothing
in the serving stack selects those tokens.

### Open questions / suggested next steps

1. **Separate "AWQ damage" from "this checkpoint is weak at tool calls."** The obvious control is a
   different quantization of the same model — **not possible on this rig** (fp8/bf16 checkpoints do
   not fit; see PATCHES.md §3). Do this on any box that can hold another build, or compare against a
   hosted GLM endpoint if one is available.
2. **Unfinished check:** whether `<tool_call>`, `<arg_key>`, `<arg_value>` are *single tokens* in
   this tokenizer. A one-token-per-tag format means the model only has to get one token right per
   tag, and a degraded head would show up as exactly this kind of near-miss. Run inside the
   container (the earlier attempt from an `exec` shell returned `-1`, i.e. the tokenizer did not
   load in that context):
   ```bash
   podman exec -i vllm-glm python3 - <<'PY'
   from transformers import AutoTokenizer
   t = AutoTokenizer.from_pretrained("/home/davidcanar/models/GLM-5.3-Flash-AWQ-W4A16", trust_remote_code=True)
   for s in ("<tool_call>", "</tool_call>", "<arg_key>", "</arg_key>", "<arg_value>", "</arg_value>"):
       print(repr(s), t.encode(s, add_special_tokens=False), t.convert_ids_to_tokens(t.encode(s, add_special_tokens=False)))
   PY
   ```
3. **Logprobs on a failing turn.** Add `"logprobs": 5, "top_logprobs": 5` to the replayed body and
   inspect what the model thought it was doing at the first malformed token. High probability on the
   malformed token ⇒ model/checkpoint; low probability but still chosen ⇒ look for something
   interfering with decoding.
4. **Port the markup-tolerance PRs** if a well-formed-but-malformed variant turns out to dominate:
   vLLM **#47190** ("Recover GLM-4.7 tool calls missing arg_key start") and **#51364** ("glm47 tool
   parser: tolerate missing opening `<arg_value>` tag"). The most common defect in our samples is
   exactly the missing *opening* tag.
5. **Compare against the DS4 engine's approach.** `~/ds4-src` has a working implementation of the
   same recovery idea: `complete_tool_call_inside_thinking()` + `parse_generated_message_ex()` in
   `ds4_agent.c`, tested by `tests/ds4_test.c::test_think_tool_recovery` (the detector must wait for
   the complete block; the parser keeps only the preceding prose as `reasoning_content`).

### Client-side workarounds available today

* Cap opencode's output: `provider.strix-halo.models."glm-5.3-flash".limit.output` and
  `options.maxTokens` at **8192** (ample for edits). This bounds a runaway turn to ~5 minutes
  instead of ~20.
* Retry on `finish_reason=stop` with no `tool_calls` (the failure is intermittent, so retries land).
* Keep the fail-open patch: the file the model wrote arrives as content and can be used as-is.

---

## 3. Issue C — MTP-off is unusable as a control **(OPEN)**

Disabling MTP (`glm53_mtp_tokens: 0` in `~/vsh-config.yaml`) is the natural control for "does
speculative decoding degrade output quality". **It cannot be used as-is:**

1. **The worker crashes.** Boot succeeds, the API answers, then the first real generation kills
   rank 1 with `c10::AcceleratorError … illegal memory access` inside `gpu_worker.execute_model`,
   followed by `SIGABRT` and `container exec_died` (see `journalctl --user -u vsh-glm-manual`).
2. **The retention interval must follow the scheduler block size.** With MTP off the coordinator
   refuses to start:
   ```
   ValueError: prefix_cache_retention_interval (2304) must be non-negative and a
   multiple of scheduler_block_size (2176).
   ```
   `VSH_GLM53_APC_RETENTION` was chosen as a multiple of the **MTP k=3** block size (2304). Set
   `export VSH_GLM53_APC_RETENTION=2176` in `~/vsh-cluster-env.odl.sh` on **both** boxes before
   booting without MTP. (Restore 2304 when MTP comes back.)

Fixing the crash would unlock the cleanest available experiment for Issue B (run
`python3 replay_n.py 6` with MTP off and compare). Suspects worth checking: KDA/MLA kernel
selection differences when `num_spec_tokens == 0`, and the `vsh-mtp-ropefree-triton-sparse` path
in PATCHES.md §1.5, whose notes say the worker aborts on speculative batches without it — the
inverse combination has apparently never been exercised.

---

## 4. Smaller things fixed along the way (already committed)

* **Adaptive-k live override was unreachable.** `Scheduler.schedule()` gated on
  `_VSH_ADAPTIVE_K.boot_enabled or _VSH_ADAPTIVE_K.force_k` (both *env*), while the JSON force is
  consulted only inside `force()` — so with `VSH_ADAPTIVE_K=off` and `VSH_ADAPTIVE_K_FORCE=0` the
  documented no-restart k-sweep silently did nothing. Fixed
  (`container/patches/vsh-adaptive-k-json-gate.py`); verified 1.00 draft tokens/step under
  `{"force": 1}`.
* **opencode output limit** in `~/.config/opencode/opencode.jsonc` set to 131 072 (128 k) on
  request; the server accepts that up to ~131 k of input (prompt + max_tokens ≤ 262 144, else
  HTTP 400).

## 5. Repo state

`~/vllm-strix-halo`, clean tree, 11 commits ahead of `origin` (not pushed). Relevant patches:
`vsh-toolcall-failopen.py`, `vsh-toolcall-drop-log.py`, `vsh-adaptive-k-json-gate.py`, with
byte-exact copies under `container/pinned-vllm/` and the analysis in `PATCHES.md` §25–§26.
Probe scripts: `scripts/{replay,replay_n,loop_probe,tool_count,effort_test,render_prompt}.py`
plus `proxy.py` for capture. Production is **MTP k=3, retention 2304, fail-open on**.
