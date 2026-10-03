#!/usr/bin/env python3
"""How often does the same request end early with no tool call?

Replays opencode's captured build request N times and classifies each outcome:
  ok        -> finish=tool_calls with valid JSON arguments
  early     -> finish=stop with (near-)empty content and no tool call
  trunc     -> finish=length
"""
from __future__ import annotations

import json
import pathlib
import sys
import urllib.request

N = int(sys.argv[1]) if len(sys.argv) > 1 else 6
cap = [p for p in sorted(pathlib.Path.home().joinpath("proxy_capture").glob("0*.json"))
       if json.loads(p.read_text()).get("tools")]
body0 = json.loads(cap[-1].read_text())

rows = []
for i in range(N):
    body = dict(body0, stream=True, stream_options={"include_usage": True})
    req = urllib.request.Request(
        "http://127.0.0.1:1234/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    content, reasoning, calls, finish, usage = [], [], {}, None, None
    with urllib.request.urlopen(req, timeout=1800) as resp:
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

    valid = False
    for v in calls.values():
        try:
            json.loads(v["args"])
            valid = True
        except Exception:  # noqa: BLE001
            pass
    kind = ("ok" if finish == "tool_calls" and valid
            else "trunc" if finish == "length"
            else "early" if finish == "stop" and not calls
            else f"other:{finish}")
    gen = (usage or {}).get("completion_tokens")
    rows.append((kind, gen, len("".join(reasoning)), len("".join(content))))
    tail = "".join(reasoning)[-70:].replace("\n", " ")
    print(f"  run {i + 1}: {kind:6s} gen={gen} reasoning_chars={len(''.join(reasoning))} "
          f"content_chars={len(''.join(content))} | think tail: ...{tail!r}", flush=True)

bad = sum(1 for r in rows if r[0] != "ok")
print(f"\n{N - bad}/{N} ok, {bad}/{N} failed")
