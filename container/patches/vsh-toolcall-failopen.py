#!/usr/bin/env python3
"""vsh-toolcall-failopen: don't let an unterminated tool call swallow the turn.

Observed on this rig (GLM-5.3-Flash, --tool-call-parser glm47): roughly three
turns in five end with finish_reason=stop, empty content, no tool_calls and
hundreds to twelve thousand generated tokens that appear in NO field of the
response. A verbatim replay of the same request succeeds the rest of the time,
so the model is calling the tool -- the text is being consumed by the tool
parser's buffer and discarded when the turn ends without a terminator.

What this patch does, at the existing end-of-turn hook
(DelegatingParser.finalize_generation):

  * tracks the raw text of the stream and whether a tool call was ever
    surfaced to the client;
  * if the turn ends with no tool call surfaced and the raw text contains the
    GLM tool-call opener, emits the text from that opener onwards as ordinary
    content instead of dropping it.

That is the semantics DS4 already ships (complete_tool_call_inside_thinking +
parse_generated_message_ex, tests/ds4_test.c::test_think_tool_recovery) and
what vLLM PR 55759 asks for on the non-streaming path. It is fail-open: a
correct tool call is untouched (the flag short-circuits), only the silent case
changes, and the alternative there is nothing at all.

Gate: VSH_TOOLCALL_FAILOPEN=0 disables it (default on).
Anchors are this image's vllm/parser/abstract_parser.py (pin 73859fec).
"""
from __future__ import annotations

import sys
from pathlib import Path

P = Path("/opt/venv/lib/python3.12/site-packages/vllm/parser/abstract_parser.py")
MARK = "# [vsh-toolcall-failopen]"

HELPER = '''
# [vsh-toolcall-failopen] ------------------------------------------------------
_VSH_TOOL_CALL_START = "<tool_call>"


def _vsh_failopen_enabled() -> bool:
    import os

    return os.environ.get("VSH_TOOLCALL_FAILOPEN", "1") not in ("", "0", "off")


def _vsh_salvage_unterminated_tool_call(
    delta_message, state, logger_
):
    """Emit a never-surfaced tool call as content instead of dropping it."""
    if not _vsh_failopen_enabled() or getattr(state, "vsh_tool_call_emitted", False):
        return delta_message
    raw = getattr(state, "vsh_raw_text", "") or ""
    idx = raw.find(_VSH_TOOL_CALL_START)
    if idx == -1:
        return delta_message
    salvaged = raw[idx:]
    if not salvaged.strip():
        return delta_message
    if delta_message is None:
        delta_message = DeltaMessage()
    if delta_message.tool_calls:
        # a partial call is already visible to the client; nothing is lost
        return delta_message
    delta_message.content = (delta_message.content or "") + salvaged
    logger_.warning(
        "vsh-toolcall-failopen: recovered %d chars of an unterminated tool "
        "call as content (no tool call was surfaced)",
        len(salvaged),
    )
    return delta_message
# -----------------------------------------------------------------------------
'''

# ---- 1. helper functions, after the existing instrumentation block ----------
HELPER_ANCHOR = """    except Exception:  # never break a request on the debug path
        pass
# -----------------------------------------------------------------------------
"""
HELPER_REPL = HELPER_ANCHOR + HELPER

# ---- 2. StreamState fields --------------------------------------------------
STATE_ANCHOR = "    function_name_returned: bool = False\n    engine_based: bool = False\n"
STATE_REPL = (
    "    function_name_returned: bool = False\n"
    "    engine_based: bool = False\n"
    "    # [vsh-toolcall-failopen] raw text of this stream, and whether a tool\n"
    "    # call was ever handed to the client\n"
    "    vsh_raw_text: str = \"\"\n"
    "    vsh_tool_call_emitted: bool = False\n"
)

# ---- 3. accumulate the raw text --------------------------------------------
ADV_ANCHOR = "        current_text, current_token_ids = state.advance(delta_text, delta_token_ids)\n"
ADV_REPL = (
    ADV_ANCHOR
    + "        # [vsh-toolcall-failopen] keep the raw text: engine-based parsers\n"
    "        # consume it and never hand it back, so this is the only copy left\n"
    "        # if the turn ends mid tool call.\n"
    "        state.vsh_raw_text = state.vsh_raw_text + (delta_text or \"\")\n"
)

# ---- 4. remember when a tool call actually reached the client ---------------
COMMIT_ANCHOR = "        state.commit(current_text, current_token_ids)\n"
COMMIT_REPL = (
    "        # [vsh-toolcall-failopen]\n"
    "        if delta_message is not None and delta_message.tool_calls:\n"
    "            state.vsh_tool_call_emitted = True\n\n"
    + COMMIT_ANCHOR
)

# ---- 5. salvage at end of turn ---------------------------------------------
FIN_ANCHOR = """        self._append_unstreamed_tool_args(delta_message)
        return delta_message
"""
FIN_REPL = """        # [vsh-toolcall-failopen]
        delta_message = _vsh_salvage_unterminated_tool_call(
            delta_message, state, logger
        )

        self._append_unstreamed_tool_args(delta_message)
        return delta_message
"""


def main() -> int:
    src = P.read_text()
    if MARK in src:
        print("already applied")
        return 0
    for name, anchor in (
        ("helper", HELPER_ANCHOR),
        ("state", STATE_ANCHOR),
        ("advance", ADV_ANCHOR),
        ("commit", COMMIT_ANCHOR),
        ("finalize", FIN_ANCHOR),
    ):
        n = src.count(anchor)
        if n != 1:
            print(f"FAIL: anchor {name} occurs {n} times", file=sys.stderr)
            return 1
    out = src.replace(HELPER_ANCHOR, HELPER_REPL, 1)
    out = out.replace(STATE_ANCHOR, STATE_REPL, 1)
    out = out.replace(ADV_ANCHOR, ADV_REPL, 1)
    out = out.replace(COMMIT_ANCHOR, COMMIT_REPL, 1)
    out = out.replace(FIN_ANCHOR, FIN_REPL, 1)
    P.write_text(out)
    print(f"applied {MARK} to {P} ({len(src)} -> {len(out)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
