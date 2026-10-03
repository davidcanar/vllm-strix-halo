#!/usr/bin/env python3
"""What does the prompt actually look like once the chat template has run?

The model mixes output formats (XML, JSON5, key=value, markdown fences) and
sometimes emits garbage tokens while its reasoning stays coherent -- the
signature of a model that is *guessing* the tool-call format. This renders
opencode's captured request through this checkpoint's own chat template and
prints the tool section plus the generation prompt, so we can see whether the
tools are presented the way GLM-5.3 was trained to consume them.
"""
from __future__ import annotations

import json
import pathlib

from transformers import AutoTokenizer

MODEL = "/home/davidcanar/models/GLM-5.3-Flash-AWQ-W4A16"
cap = [p for p in sorted(pathlib.Path("/home/davidcanar/proxy_capture").glob("0*.json"))
       if json.loads(p.read_text()).get("tools")]
req = json.loads(cap[-1].read_text())

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
msgs = req["messages"]
tools = req.get("tools")

rendered = tok.apply_chat_template(
    msgs, tools=tools, tokenize=False, add_generation_prompt=True
)
print(f"rendered length: {len(rendered)} chars, ~{len(tok.encode(rendered))} tokens")
print("=" * 70)
print("FIRST 700 CHARS (system prompt + tool presentation):")
print(rendered[:700])
print("=" * 70)
print("LAST 900 CHARS (end of tools -> generation prompt):")
print(rendered[-900:])
print("=" * 70)
print("tool-format markers in the rendered prompt:")
for m in ("<tool_call>", "</tool_call>", "<arg_key>", "<arg_value>", "<tools>",
          "</tools>", "function", "\"parameters\"", "<|tools|>"):
    print(f"  {m!r}: {rendered.count(m)}")
