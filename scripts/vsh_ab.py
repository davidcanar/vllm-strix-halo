#!/usr/bin/env python3
"""vsh_ab.py -- matched-length A/B harness for drafters (MTP vs DFlash2).

Fixes the metric the earlier comparison got wrong: the engine step rate is not
"HTTP chunks per second" (a step can deliver several accepted tokens in one
chunk). Steps are counted from the engine's own draft counter, so

    step_ms  = decode_window * 1000 / drafts_delta
    tok/step = accepted/drafts + 1            (engine-side)

Also supports:
  * --out-text FILE   dump the completion so two arms can be compared verbatim
  * --tools           send an OpenAI tool list and report whether tool_calls
                      came back (the agent-workload functional gate)

usage:
  python3 vsh_ab.py --label mtp-json-1 --mode json --reps 5 --max-tokens 220
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import time
import urllib.request

WORDS = (
    "harbor lantern granite meadow cipher kettle ember thistle quarry orchard "
    "basalt willow cinder prism fern mosaic tundra gable spire reef cobalt"
).split()

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                    "unit": {"type": "string", "enum": ["c", "f"]},
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": "Search the internal documentation",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}, "top_k": {"type": "integer"}},
                "required": ["query"],
            },
        },
    },
]


def metrics(port: int) -> dict:
    out: dict = {}
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=10) as r:
        for line in r.read().decode().splitlines():
            if line.startswith("#") or " " not in line:
                continue
            name, _, val = line.rpartition(" ")
            try:
                f = float(val)
            except ValueError:
                continue
            base = name.split("{", 1)[0]
            out[base] = out.get(base, 0.0) + f
            if base.endswith("_per_pos_total"):
                out[name] = f
    return out


def build(mode: str, nonce: str, prompt_tokens: int) -> tuple[str, list | None]:
    rng = random.Random(4242)
    body = " ".join(rng.choice(WORDS) for _ in range(int(prompt_tokens * 4.2 / 6.2)))
    if mode == "json":
        ask = (
            "Return ONLY a JSON object with keys id, title, tags (5 items), "
            "summary, score and source for the document above. No prose, no fences."
        )
    elif mode == "tools":
        ask = (
            "First call search_docs with a query naming all six topics in the "
            "archive, then call get_weather for Paris and for Berlin. Emit both "
            "calls."
        )
    else:
        ask = "Write a 180-word story about a lighthouse keeper and a stray cat."
    text = f"Archive document {nonce} follows.\n\n{body}\n\nEnd of archive. {ask}"
    return text, (TOOLS if mode == "tools" else None)


def run_one(args, mode: str, nonce: str, idx: int) -> dict:
    text, tools = build(mode, nonce, args.prompt_tokens)
    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": text}],
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"reasoning_effort": "low"},
        "cache_salt": f"{mode}{idx}",
    }
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"
    before = metrics(args.port)
    t0 = time.perf_counter()
    req = urllib.request.Request(
        f"http://127.0.0.1:{args.port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    ttft = None
    times: list[float] = []
    usage = None
    tool_calls: list = []
    content: list[str] = []
    with urllib.request.urlopen(req, timeout=1800) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            if line == "data: [DONE]":
                break
            d = json.loads(line[6:])
            if d.get("usage"):
                usage = d["usage"]
                continue
            ch = (d.get("choices") or [{}])[0]
            delta = ch.get("delta") or {}
            if delta.get("tool_calls"):
                tool_calls.extend(delta["tool_calls"])
            for k in ("content", "reasoning_content"):
                if delta.get(k):
                    content.append(delta[k])
            # any chunk counts as activity for TTFT: the first one carries only
            # {"role": "assistant", "content": ""} for tool-call responses
            if delta:
                now = time.perf_counter()
                if ttft is None:
                    ttft = now - t0
                times.append(now)
    after = metrics(args.port)

    dd = lambda k: after.get(k, 0.0) - before.get(k, 0.0)  # noqa: E731
    drafts = dd("vllm:spec_decode_num_drafts_total")
    acc = dd("vllm:spec_decode_num_accepted_tokens_total")
    dek = dd("vllm:spec_decode_num_draft_tokens_total")
    dec_s = (times[-1] - times[0]) if len(times) > 1 else 0.0
    rec = {
        "label": args.label,
        "mode": mode,
        "rep": idx,
        "prompt": usage["prompt_tokens"] if usage else None,
        "gen": usage["completion_tokens"] if usage else None,
        "ttft_s": ttft,
        "tool_calls": len(tool_calls),
        "tool_name": (tool_calls[0].get("function", {}).get("name") if tool_calls else None),
        "drafts": drafts,
        "draft_tokens": dek,
        "accepted": acc,
        "tok_per_step": (acc / drafts + 1) if drafts else float("nan"),
        "step_ms": (dec_s * 1000 / drafts) if drafts else float("nan"),
        "decode_tps": ((usage["completion_tokens"] - 1) / dec_s) if (usage and dec_s) else float("nan"),
        "text": "".join(content)[:4000],
    }
    if args.out_text:
        with open(args.out_text, "a") as fh:
            fh.write(f"=== {args.label} rep{idx} tool_calls={rec['tool_calls']} ===\n")
            fh.write(rec["text"] + "\n")
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=1234)
    ap.add_argument("--model", default="glm-5.3-flash")
    ap.add_argument("--label", required=True)
    ap.add_argument("--mode", default="json", choices=("json", "tools", "prose"))
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--prompt-tokens", type=int, default=2000)
    ap.add_argument("--max-tokens", type=int, default=220)
    ap.add_argument("--out-text", default="")
    ap.add_argument("--jsonl", default="/home/davidcanar/ab_results.jsonl")
    args = ap.parse_args()

    recs = []
    for i in range(args.reps):
        r = run_one(args, args.mode, "AB%04d" % i, i)
        recs.append(r)
        ttft_s = r["ttft_s"] if r["ttft_s"] is not None else float("nan")
        print(
            f"[{r['label']} r{i}] gen={r['gen']} ttft={ttft_s:.2f}s "
            f"tok/step={r['tok_per_step']:.2f} step={r['step_ms']:.0f}ms "
            f"decode={r['decode_tps']:.2f}tok/s drafts={r['drafts']:.0f} "
            f"tools={r['tool_calls']}"
            + (f"({r['tool_name']})" if r["tool_name"] else ""),
            flush=True,
        )
    with open(args.jsonl, "a") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")

    ok = [r for r in recs if r["drafts"] and r["gen"]]
    if ok:
        med = lambda k: statistics.median([r[k] for r in ok])  # noqa: E731
        print(
            f"\n=== {args.label} ({args.mode}, n={len(ok)}) ===\n"
            f"  tokens/step : {med('tok_per_step'):.2f}   (median of reps)\n"
            f"  step        : {med('step_ms'):.0f} ms\n"
            f"  decode      : {med('decode_tps'):.2f} tok/s\n"
            f"  tool_calls  : {sum(1 for r in ok if r['tool_calls'])}/{len(ok)} responses"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
