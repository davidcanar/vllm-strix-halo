#!/usr/bin/env python3
"""Characterise the degenerate mode and the long-loop case.

Runs the captured opencode request N times with a selectable user prompt,
writes every full response to disk, and reports per run:

  finish, generated tokens, reasoning/content chars, the tool-call name if one
  was surfaced, and -- for the long generations -- how repetitive the text is
  (share of the output taken by its most repeated 60-char window), which is
  what distinguishes "the model is looping" from "the model wrote a lot".

usage: loop_probe.py <label> <n> [prompt]
"""
from __future__ import annotations

import json
import pathlib
import sys
import urllib.request
from collections import Counter

OUT = pathlib.Path.home() / "loop_cap"
OUT.mkdir(exist_ok=True)

LABEL = sys.argv[1] if len(sys.argv) > 1 else "snake"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 4
PROMPT = sys.argv[3] if len(sys.argv) > 3 else None

cap = [p for p in sorted(pathlib.Path.home().joinpath("proxy_capture").glob("0*.json"))
       if json.loads(p.read_text()).get("tools")]
base = json.loads(cap[-1].read_text())
if PROMPT:
    for m in base["messages"]:
        if m["role"] == "user":
            m["content"] = PROMPT


def repetition_score(text: str, win: int = 60) -> float:
    """Share of the text covered by its single most repeated window."""
    if len(text) < win * 3:
        return 0.0
    windows = Counter(text[i : i + win] for i in range(0, len(text) - win, win // 2))
    top, count = windows.most_common(1)[0]
    return (count * (win // 2)) / len(text)


def run(i: int) -> dict:
    body = dict(base, stream=True, stream_options={"include_usage": True})
    req = urllib.request.Request(
        "http://127.0.0.1:1234/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    content, reasoning, calls, finish, usage = [], [], {}, None, None
    with urllib.request.urlopen(req, timeout=3600) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            d = json.loads(line[6:])
            if d.get("usage"):
                usage = d["usage"]
            ch = (d.get("choices") or [{}])[0]
            if ch.get("finish_reason"):
                finish = ch["finish_reason"]
            delta = ch.get("delta") or {}
            if delta.get("content"):
                content.append(delta["content"])
            if delta.get("reasoning"):
                reasoning.append(delta["reasoning"])
            for tc in delta.get("tool_calls") or []:
                slot = calls.setdefault(tc.get("index", 0), {"name": None, "args": ""})
                fn = tc.get("function") or {}
                if fn.get("name"):
                    slot["name"] = fn["name"]
                if fn.get("arguments"):
                    slot["args"] += fn["arguments"]

    text = "".join(content)
    (OUT / f"{LABEL}-{i}.txt").write_text(
        f"finish={finish} usage={usage}\n--- reasoning ---\n{''.join(reasoning)}\n"
        f"--- content ---\n{text}\n--- calls ---\n{json.dumps(calls)[:4000]}\n"
    )
    gen = (usage or {}).get("completion_tokens") or 0
    valid = False
    names = []
    for v in calls.values():
        names.append(v["name"])
        try:
            json.loads(v["args"])
            valid = True
        except Exception:  # noqa: BLE001
            pass
    return {
        "i": i, "finish": finish, "gen": gen, "r": len("".join(reasoning)),
        "c": len(text), "calls": len(calls), "valid": valid,
        "names": names, "rep": repetition_score(text + "".join(reasoning)),
    }


rows = [run(i) for i in range(N)]
print(f"{'#':>2} {'finish':>10} {'gen':>6} {'reason':>7} {'content':>8} {'calls':>5} "
      f"{'valid':>5} {'rep%':>5}  names")
for r in rows:
    print(f"{r['i']:>2} {str(r['finish']):>10} {r['gen']:>6} {r['r']:>7} {r['c']:>8} "
          f"{r['calls']:>5} {str(r['valid']):>5} {r['rep'] * 100:>5.1f}  {r['names']}")

okc = sum(1 for r in rows if r["valid"])
print(f"\n{LABEL}: {okc}/{N} produced a usable tool call; "
      f"{sum(1 for r in rows if r['gen'] > 4000)}/{N} ran over 4000 tokens")
print("captures:", OUT)
