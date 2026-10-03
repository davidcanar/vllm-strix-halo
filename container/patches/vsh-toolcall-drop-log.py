#!/usr/bin/env python3
"""vsh-toolcall-drop-log: dump the text the parser discards when a tool call
is emitted inside a reasoning span that never closes.

Pure instrumentation -- it does not change what the client receives. Writes the
full accumulated generation to $VSH_TOOLCALL_DROP_DIR (default
/tmp/vsh-toolcall-drops) when either

  * the non-streaming parse() ends with no tool calls but the reasoning span
    contains the GLM tool-call markup, or
  * the streaming parse_delta() sees the final delta while still inside the
    reasoning phase and the accumulated text contains the markup.

Gate: VSH_TOOLCALL_DROP_LOG=1 (default off).

Anchors are this image's vllm/parser/abstract_parser.py (vLLM 0.31.0.dev0,
pin 73859fec). Fails closed if they drift.
"""
from __future__ import annotations

import sys
from pathlib import Path

P = Path("/opt/venv/lib/python3.12/site-packages/vllm/parser/abstract_parser.py")
MARK = "# [vsh-toolcall-drop-log]"

HELPER = '''

# [vsh-toolcall-drop-log] ------------------------------------------------------
def _vsh_drop_log_enabled() -> bool:
    import os

    return os.environ.get("VSH_TOOLCALL_DROP_LOG", "0") not in ("", "0", "off")


def _vsh_drop_log(text: str | None, where: str) -> None:
    """Persist text the parser is about to discard, for offline analysis."""
    if not text:
        return
    import os
    import time

    try:
        d = os.environ.get("VSH_TOOLCALL_DROP_DIR", "/tmp/vsh-toolcall-drops")
        os.makedirs(d, exist_ok=True)
        name = f"{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}-{where}.txt"
        with open(os.path.join(d, name), "w") as fh:
            fh.write(text)
    except Exception:  # never break a request on the debug path
        pass
# -----------------------------------------------------------------------------
'''

ANCHOR_PARSE = """        reasoning, content = self.extract_reasoning(model_output, request)
        tool_calls, content = self._extract_tool_calls(
            content=content,
            request=request,
            enable_auto_tools=enable_auto_tools,
        )
        return reasoning, content, tool_calls
"""

REPL_PARSE = """        reasoning, content = self.extract_reasoning(model_output, request)
        tool_calls, content = self._extract_tool_calls(
            content=content,
            request=request,
            enable_auto_tools=enable_auto_tools,
        )
        # [vsh-toolcall-drop-log]
        if not tool_calls and reasoning and _vsh_drop_log_enabled():
            if "<tool_call>" in reasoning or "<arg_key>" in reasoning:
                _vsh_drop_log(model_output, "parse")
        return reasoning, content, tool_calls
"""

ANCHOR_DELTA = """        current_text, current_token_ids = state.advance(delta_text, delta_token_ids)
        delta_message: DeltaMessage | None = None
        reasoning_transitioned = False
"""

REPL_DELTA = """        current_text, current_token_ids = state.advance(delta_text, delta_token_ids)
        # [vsh-toolcall-drop-log] the turn ended while still inside the
        # reasoning span: if the model emitted tool-call markup there, the
        # tool parser never saw it. Persist it before the request goes away.
        if finished and not state.reasoning_ended and _vsh_drop_log_enabled():
            if "<tool_call>" in current_text or "<arg_key>" in current_text:
                _vsh_drop_log(current_text, "stream")
        delta_message: DeltaMessage | None = None
        reasoning_transitioned = False
"""

IMPORT_ANCHOR = "from dataclasses import dataclass, field, replace\n"
IMPORT_REPL = "from dataclasses import dataclass, field, replace\n" + HELPER


def main() -> int:
    src = P.read_text()
    if MARK in src:
        print("already applied")
        return 0
    for name, anchor in (
        ("import", IMPORT_ANCHOR),
        ("parse", ANCHOR_PARSE),
        ("parse_delta", ANCHOR_DELTA),
    ):
        n = src.count(anchor)
        if n != 1:
            print(f"FAIL: anchor {name} occurs {n} times", file=sys.stderr)
            return 1
    out = src.replace(IMPORT_ANCHOR, IMPORT_REPL, 1)
    out = out.replace(ANCHOR_PARSE, REPL_PARSE, 1)
    out = out.replace(ANCHOR_DELTA, REPL_DELTA, 1)
    P.write_text(out)
    print(f"applied {MARK} to {P} ({len(src)} -> {len(out)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
