#!/usr/bin/env python3
"""Replay opencode's captured request verbatim and dump what the server emits.

This separates server behaviour from client behaviour with no guessing: same
body, same tools, same max_tokens -- only the transport differs.
"""
from __future__ import annotations

import json
import pathlib
import sys
import urllib.request

cap = sorted(pathlib.Path.home().joinpath("proxy_capture").glob("0*.json"))
target = None
for p in cap:
    d = json.loads(p.read_text())
    if d.get("tools"):
        target = (p, d)
if not target:
    sys.exit("no tool-bearing capture found")
path, body = target
print(f"replaying {path.name}: {len(body.get('tools') or [])} tools, "
      f"max_tokens={body.get('max_tokens')}, stream={body.get('stream')}, "
      f"msgs={len(body['messages'])}")
for m in body["messages"]:
    c = m.get("content")
    if isinstance(c, list):
        c = " ".join(str(x.get("text", "")) for x in c if isinstance(x, dict))
    print(f"  {m['role']}: {len(str(c))} chars | {str(c)[:110]!r}")

body = dict(body)
body["stream"] = True
body["stream_options"] = {"include_usage": True}
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
        if delta.get("reasoning_content"):
            reasoning.append(delta["reasoning_content"])
        if delta.get("reasoning"):
            reasoning.append(delta["reasoning"])
        for tc in delta.get("tool_calls") or []:
            slot = calls.setdefault(tc.get("index", 0), {"name": None, "args": ""})
            fn = tc.get("function") or {}
            if fn.get("name"):
                slot["name"] = fn["name"]
            if fn.get("arguments"):
                slot["args"] += fn["arguments"]

print(f"\nfinish={finish}")
print("usage:", usage)
r, c = "".join(reasoning), "".join(content)
print(f"reasoning: {len(r)} chars | tail: ...{r[-160:]!r}")
print(f"content  : {len(c)} chars | {c[:200]!r}")
print(f"tool_calls: {len(calls)}")
for i, v in sorted(calls.items()):
    ok = True
    try:
        json.loads(v["args"])
    except Exception:  # noqa: BLE001
        ok = False
    print(f"  [{i}] {v['name']} args={len(v['args'])} chars json_valid={ok}")
    print(f"       head: {v['args'][:120]!r}")
