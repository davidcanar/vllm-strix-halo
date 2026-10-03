#!/usr/bin/env python3
"""Does the mangled tool-call format depend on how many tools are offered?

Same short request ("create hello.txt containing the word hello"), same model,
same opencode system prompt -- only the size of the tool list changes. The
model's reasoning stays coherent in every failing case, so this asks whether
the *format* degrades with the amount of tool schema in the prompt.

usage: tool_count.py <n_tools> <reps>
"""
from __future__ import annotations

import json
import pathlib
import sys
import urllib.request

N_TOOLS = int(sys.argv[1]) if len(sys.argv) > 1 else 2
REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 4

cap = [p for p in sorted(pathlib.Path.home().joinpath("proxy_capture").glob("0*.json"))
       if json.loads(p.read_text()).get("tools")]
base = json.loads(cap[-1].read_text())
base["tools"] = base["tools"][:N_TOOLS]
for m in base["messages"]:
    if m["role"] == "user":
        m["content"] = "create hello.txt containing the word hello"

rows = []
for i in range(REPS):
    body = dict(base, stream=True, stream_options={"include_usage": True})
    req = urllib.request.Request(
        "http://127.0.0.1:1234/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    content, calls, finish, usage = [], {}, None, None
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
            for tc in delta.get("tool_calls") or []:
                slot = calls.setdefault(tc.get("index", 0), {"name": None, "args": ""})
                fn = tc.get("function") or {}
                if fn.get("name"):
                    slot["name"] = fn["name"]
                if fn.get("arguments"):
                    slot["args"] += fn["arguments"]
    text = "".join(content)
    valid = False
    for v in calls.values():
        try:
            json.loads(v["args"])
            valid = True
        except Exception:  # noqa: BLE001
            pass
    rows.append(valid)
    print(f"  n_tools={N_TOOLS} r{i + 1}: {'ok  ' if valid else 'FAIL'} finish={finish} "
          f"gen={(usage or {}).get('completion_tokens')} calls={len(calls)} "
          f"content={len(text)}c | {text[:110]!r}", flush=True)

print(f"n_tools={N_TOOLS}: {sum(rows)}/{REPS} usable")
