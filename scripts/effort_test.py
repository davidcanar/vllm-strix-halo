#!/usr/bin/env python3
"""The template injects "Reasoning Effort: Max" by default (opencode sends no
chat_template_kwargs). Does turning thinking effort down fix the malformed
tool-call format?

Also reports whether the GLM tool-call tags are single tokens, i.e. whether
the model only has to get one token right per tag.
"""
from __future__ import annotations

import json
import pathlib
import sys
import urllib.request

MODEL = "/home/davidcanar/models/GLM-5.3-Flash-AWQ-W4A16"
cap = [p for p in sorted(pathlib.Path("/home/davidcanar/proxy_capture").glob("0*.json"))
       if json.loads(p.read_text()).get("tools")]
base = json.loads(cap[-1].read_text())
base["tools"] = base["tools"][:10]
for m in base["messages"]:
    if m["role"] == "user":
        m["content"] = "create hello.txt containing the word hello"


def tokens_of(s: str) -> int:
    try:
        from transformers import AutoTokenizer

        t = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
        return len(t.encode(s, add_special_tokens=False))
    except Exception as e:  # noqa: BLE001
        return -1


print("token counts (1 = single token):")
for s in ("<tool_call>", "</tool_call>", "<arg_key>", "</arg_key>",
          "<arg_value>", "</arg_value>", "filePath", "content"):
    print(f"  {s!r}: {tokens_of(s)}")


def run(effort, i: int) -> bool:
    body = dict(base, stream=True, stream_options={"include_usage": True})
    if effort:
        body["chat_template_kwargs"] = {"reasoning_effort": effort}
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
    txt = "".join(content)
    print(f"  effort={effort or 'default(Max)':14s} r{i + 1}: {'ok  ' if valid else 'FAIL'} "
          f"gen={(usage or {}).get('completion_tokens'):>5} "
          f"reason={len(''.join(reasoning)):>5}c calls={len(calls)} | {txt[:90]!r}", flush=True)
    return valid


for effort in (sys.argv[1] if len(sys.argv) > 1 else None,):
    res = [run(effort, i) for i in range(5)]
    print(f"effort={effort or 'default(Max)'}: {sum(res)}/5 usable\n")
