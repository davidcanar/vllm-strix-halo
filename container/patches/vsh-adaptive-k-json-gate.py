#!/usr/bin/env python3
"""[vsh-adaptive-k] fix: the live JSON override never reached the cap.

schedule() gates on `boot_enabled or force_k` (both env), but the JSON force is
only consulted *inside* force(). With VSH_ADAPTIVE_K=off and
VSH_ADAPTIVE_K_FORCE=0 the gate never opens, so the documented no-restart
k-sweep silently did nothing. Include the JSON decision in the gate.
"""
from __future__ import annotations

import sys
from pathlib import Path

P = Path("/opt/venv/lib/python3.12/site-packages/vllm/v1/core/sched/scheduler.py")
OLD = "        if _VSH_ADAPTIVE_K.boot_enabled or _VSH_ADAPTIVE_K.force_k:  # [vsh-adaptive-k]\n"
NEW = (
    "        if (  # [vsh-adaptive-k] the live JSON override counts as enabled too\n"
    "            _VSH_ADAPTIVE_K.boot_enabled\n"
    "            or _VSH_ADAPTIVE_K.force_k\n"
    "            or _VSH_ADAPTIVE_K.force() is not None\n"
    "        ):\n"
)

src = P.read_text()
if "the live JSON override counts as enabled too" in src:
    print("already applied")
    raise SystemExit(0)
n = src.count(OLD)
if n != 1:
    print(f"FAIL: anchor occurs {n} times", file=sys.stderr)
    raise SystemExit(1)
P.write_text(src.replace(OLD, NEW, 1))
import ast

ast.parse(P.read_text())
print("patched the adaptive-k gate; syntax OK")
