#!/usr/bin/env python3
"""vsh_bench.py -- decode/TTFT bench for the GLM-5.3 rig with spec-decode receipts.

Reports, per run:
  * prompt / cached / completion tokens, TTFT, prefill tok/s
  * decode tok/s excluding TTFT (median of per-chunk intervals is also printed)
  * spec-decode deltas from /metrics: drafts, draft tokens, accepted tokens,
    accepted-per-position (the acceptance curve), accepted/draft ratio
  * prefix-cache deltas: queries, hits, cached prompt tokens

Usage:
  python3 vsh_bench.py --label k3 --prompt-tokens 2000 --max-tokens 256
  python3 vsh_bench.py --force 2          # write the live adaptive-k override
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
    "basalt willow cinder prism fern mosaic tundra gable spire reef cobalt "
    "juniper delta lumen marlin peat quartz sable tussock vellum walnut zephyr "
    "anvil bramble cistern dulse escarpment"
).split()

SPEC_KEYS = (
    "vllm:spec_decode_num_drafts_total",
    "vllm:spec_decode_num_draft_tokens_total",
    "vllm:spec_decode_num_accepted_tokens_total",
)
POS_KEY = "vllm:spec_decode_num_accepted_tokens_per_pos_total"
PC_KEYS = (
    "vllm:prefix_cache_queries_total",
    "vllm:prefix_cache_hits_total",
    "vllm:prompt_tokens_cached_total",
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
)


def metrics(port: int) -> dict:
    out: dict = {}
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=10) as r:
            for line in r.read().decode().splitlines():
                if line.startswith("#") or " " not in line:
                    continue
                name, _, val = line.rpartition(" ")
                try:
                    fval = float(val)
                except ValueError:
                    continue
                out[name] = fval  # full key, labels included
                base = name.split("{", 1)[0]
                out[base] = out.get(base, 0.0) + fval
    except Exception as exc:  # noqa: BLE001
        print(f"[bench] /metrics unavailable: {exc}")
    return out


def snap(port: int) -> dict:
    m = metrics(port)
    d = {k: m.get(k, 0.0) for k in SPEC_KEYS + PC_KEYS}
    for name, val in m.items():
        if name.startswith(POS_KEY):
            d[name] = val
    return d


def build_prompt(target_tokens: int, nonce: str, mode: str) -> list[dict]:
    rng = random.Random(1234)
    n_words = max(int(target_tokens * 4.2 / 6.2), 20)
    body = " ".join(rng.choice(WORDS) for _ in range(n_words))
    if mode == "structured":
        ask = (
            "Return ONLY a JSON object with keys id, title, tags (5 items), "
            "summary and score for the document above. No prose, no markdown."
        )
    else:
        ask = (
            "Write a 180-word story about a lighthouse keeper and a stray cat. "
            "Do not mention the archive above."
        )
    text = (
        f"Archive document {nonce} follows.\n\n{body}\n\n"
        f"End of archive. Ignore its contents entirely. Now {ask}"
    )
    return [{"role": "user", "content": text}]


def run(args) -> int:
    nonce = args.nonce or ("%06X" % random.getrandbits(24))
    body = {
        "model": args.model,
        "messages": build_prompt(args.prompt_tokens, nonce, args.mode),
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"reasoning_effort": args.reasoning},
    }
    if args.salt:
        body["cache_salt"] = args.salt
    before = snap(args.port)
    t0 = time.perf_counter()
    req = urllib.request.Request(
        f"http://127.0.0.1:{args.port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    ttft = None
    times: list[float] = []
    usage = None
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
            delta = d["choices"][0].get("delta") or {}
            if delta.get("content") or delta.get("reasoning_content"):
                now = time.perf_counter()
                if ttft is None:
                    ttft = now - t0
                times.append(now)
    t_end = time.perf_counter()
    after = snap(args.port)

    if ttft is None or not usage or not times:
        print(f"[{args.label}] FAILED ttft={ttft} usage={usage} chunks={len(times)}")
        return 1

    ptoks = usage["prompt_tokens"]
    ctoks = usage["completion_tokens"]
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    dec_s = (times[-1] - times[0]) if len(times) > 1 else 0.0
    gaps = [b - a for a, b in zip(times, times[1:])]
    # tokens/s uses the token count from usage: speculative decoding delivers
    # several accepted tokens per engine step, so chunks/s != tokens/s.
    decode_tps = (ctoks - 1) / dec_s if dec_s else float("nan")
    steps_per_s = (len(times) - 1) / dec_s if dec_s else float("nan")

    dd = {k: after.get(k, 0) - before.get(k, 0) for k in SPEC_KEYS}
    drafts = dd[SPEC_KEYS[0]]
    draft_tokens = dd[SPEC_KEYS[1]]
    accepted = dd[SPEC_KEYS[2]]
    def _pos(key: str) -> str:
        try:
            return key.split('position="')[1].split('"')[0]
        except IndexError:
            return "?"

    pos = sorted(
        ((_pos(k), after.get(k, 0) - before.get(k, 0)) for k in set(after) | set(before) if k.startswith(POS_KEY)),
        key=lambda kv: kv[0],
    )
    pc_q = after.get(PC_KEYS[0], 0) - before.get(PC_KEYS[0], 0)
    pc_h = after.get(PC_KEYS[1], 0) - before.get(PC_KEYS[1], 0)

    print(
        f"[{args.label}] prompt={ptoks} cached={cached} gen={ctoks} "
        f"TTFT={ttft:.2f}s prefill={ptoks / ttft:.0f}tok/s "
        f"decode={decode_tps:.2f}tok/s steps={steps_per_s:.2f}/s "
        f"median_gap={statistics.median(gaps) * 1000 if gaps else float('nan'):.0f}ms "
        f"total={t_end - t0:.1f}s"
    )
    print(
        f"[{args.label}] spec: drafts={drafts:.0f} draft_tok={draft_tokens:.0f} "
        f"accepted={accepted:.0f} acc/draft={accepted / drafts if drafts else 0:.2f} "
        f"tok/step={accepted / drafts + 1 if drafts else 0:.2f} "
        f"pos={' '.join(f'{k}:{v:.0f}' for k, v in pos)}"
    )
    print(f"[{args.label}] prefix_cache: queries={pc_q:.0f} hits={pc_h:.0f}")
    if args.json_out:
        with open(args.json_out, "a") as fh:
            fh.write(
                json.dumps(
                    {
                        "label": args.label,
                        "mode": args.mode,
                        "prompt": ptoks,
                        "cached": cached,
                        "gen": ctoks,
                        "ttft_s": ttft,
                        "decode_tps": decode_tps,
                        "median_gap_ms": statistics.median(gaps) * 1000 if gaps else None,
                        "drafts": drafts,
                        "draft_tokens": draft_tokens,
                        "accepted": accepted,
                        "pos": {k: v for k, v in pos},
                        "prefix_queries": pc_q,
                        "prefix_hits": pc_h,
                    }
                )
                + "\n"
            )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=1234)
    ap.add_argument("--model", default="glm-5.3-flash")
    ap.add_argument("--label", default="run")
    ap.add_argument("--mode", default="prose", choices=("prose", "structured"))
    ap.add_argument("--prompt-tokens", type=int, default=2000)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--reasoning", default="low")
    ap.add_argument("--salt", default="")
    ap.add_argument("--nonce", default="")
    ap.add_argument("--json-out", default="")
    ap.add_argument("--force", default="")
    ap.add_argument("--override", default="/home/davidcanar/vsh-adaptive-k.json")
    args = ap.parse_args()

    if args.force != "":
        payload = {} if args.force == "off" else {"force": int(args.force)}
        if args.force.startswith("{"):
            payload = json.loads(args.force)
        with open(args.override, "w") as fh:
            json.dump(payload, fh)
        print(f"[bench] override {args.override} <- {payload}")
        return 0
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
