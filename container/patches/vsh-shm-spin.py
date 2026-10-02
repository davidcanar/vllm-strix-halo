#!/usr/bin/env python3
"""vsh-shm-spin: make the SHM reader's busy-spin window tunable (default stock).

`SpinCondition.wait()` (vllm/distributed/device_communicators/shm_broadcast.py)
calls `sched_yield()` in a loop for `busy_loop_s` seconds after every read
before it falls back to the zmq poll. The reader is constructed with the
default `busy_loop_s = 1`, so the EngineCore<->worker control-plane reader
spins for a full second after every step -- i.e. it never idles while a request
is being served, burning a core that this rig needs (measured 2026-10-02: the
decode step is ~50% GPU-idle and CPU-bound).

The knob (mirrors MiaAI-Lab's `busy_loop_s 1 -> 0.016`, which they measured at
+0.95% decode and -85% EngineCore CPU):

    VSH_SHM_BUSY_LOOP_S   unset/empty -> 1.0 (stock, byte-identical behaviour)
                          0.016       -> spin 16 ms, then idle on the zmq poll

Env-gated, idempotent, fail-closed, backed up and AST-checked.
"""
from __future__ import annotations

import pathlib
import py_compile
import shutil
import time

P = pathlib.Path(
    "/opt/venv/lib/python3.12/site-packages/vllm/distributed/device_communicators/"
    "shm_broadcast.py"
)
MARK = "# [vsh-shm-spin]"

HELPER = f'''def _vsh_shm_busy_loop_s() -> float:  {MARK}
    """Reader busy-spin window in seconds (stock default 1.0)."""
    import os

    raw = os.environ.get("VSH_SHM_BUSY_LOOP_S", "").strip()
    if not raw:
        return 1.0
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 1.0


'''

OLD_SIG = """        notify_address: str,
        busy_loop_s: float = 1,
    ):
        self.is_reader = is_reader
"""

NEW_SIG = f"""        notify_address: str,
        busy_loop_s: float = -1,
    ):
        self.is_reader = is_reader
"""

OLD_ASSIGN = """            # Time to keep busy-looping on the shm buffer before going idle
            self.busy_loop_s = busy_loop_s
"""

NEW_ASSIGN = f"""            # Time to keep busy-looping on the shm buffer before going idle
            self.busy_loop_s = (  {MARK}
                _vsh_shm_busy_loop_s() if busy_loop_s < 0 else busy_loop_s
            )
"""


def main() -> int:
    text = P.read_text()
    if MARK in text:
        print("already patched")
        return 0
    for label, old in (("signature", OLD_SIG), ("assignment", OLD_ASSIGN)):
        if text.count(old) != 1:
            raise SystemExit(f"{label} anchor count {text.count(old)}")

    backup = P.with_suffix(f".py.bak-{time.strftime('%Y%m%d-%H%M%S')}")
    shutil.copy2(P, backup)

    anchor = "class SpinCondition:"
    if text.count(anchor) != 1:
        raise SystemExit("class anchor count")
    text = text.replace(anchor, HELPER + anchor, 1)
    text = text.replace(OLD_SIG, NEW_SIG, 1)
    text = text.replace(OLD_ASSIGN, NEW_ASSIGN, 1)

    compile(text, str(P), "exec")
    P.write_text(text)
    py_compile.compile(str(P), doraise=True, cfile="/tmp/_spin.pyc")
    print(f"vsh-shm-spin installed; backup={backup.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
