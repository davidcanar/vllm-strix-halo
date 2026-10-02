#!/usr/bin/env python3
"""Add the env-gated --prefix-cache-retention-interval to vsh-manual-serve.sh.

CacheConfig.prefix_cache_retention_interval defaults to 0 = "retain only
semantic checkpoints (the latest replay boundary and shared-prefix junctions)".
For the KDA/Mamba groups in GLM-5.3's hybrid KV that is the *sparsest* setting,
and the hybrid min() then ties every attention hit to a Mamba checkpoint that
may not exist yet -- measured on 2026-10-02: an identical repeat of a 13.9K
prompt missed, the next one hit. A positive value (a multiple of the scheduler
block size, 2304 here) keeps a checkpoint per segment; the flag is a stock
EngineArgs field (engine/arg_utils.py:576), so no vLLM patch is needed.

    VSH_GLM53_APC_RETENTION   unset/empty -> no flag (stock 0)
                              <int>       -> --prefix-cache-retention-interval <int>
"""
from __future__ import annotations

import pathlib
import shutil
import time

P = pathlib.Path("/home/davidcanar/vsh-manual-serve.sh")
MARK = "# [vsh-apc-retention]"

ANCHOR = """KVD=()
"""
NEW = """# [vsh-apc-retention] Retention interval for sliding-window / Mamba (KDA)
# prefix-cache checkpoints. Stock default (flag absent) is 0 = "keep only the
# latest replay boundary", the sparsest setting; on this hybrid model that ties
# every hit to a KDA checkpoint that may not exist yet, so a repeat of the same
# prompt can miss once before it hits. A positive multiple of the scheduler
# block size (2304) keeps one checkpoint per segment. Env-gated: unset = stock.
APCR=()
if [ -n "${VSH_GLM53_APC_RETENTION:-}" ]; then
  APCR=(--prefix-cache-retention-interval "${VSH_GLM53_APC_RETENTION}")
  echo "[vsh-serve] prefix-cache retention interval: ${VSH_GLM53_APC_RETENTION}"
fi

KVD=()
"""

SPOT_OLD = """  "${KVD[@]}" \\
"""
SPOT_NEW = """  "${KVD[@]}" "${APCR[@]}" \\
"""


def main() -> int:
    text = P.read_text()
    if MARK in text:
        print("already patched")
        return 0
    if text.count(ANCHOR) != 1:
        raise SystemExit(f"anchor count {text.count(ANCHOR)}")
    if text.count(SPOT_OLD) != 1:
        raise SystemExit(f"flag-site count {text.count(SPOT_OLD)}")
    backup = P.with_suffix(f".bak-{time.strftime('%Y%m%d-%H%M%S')}")
    shutil.copy2(P, backup)
    text = text.replace(ANCHOR, NEW, 1)
    text = text.replace(SPOT_OLD, SPOT_NEW, 1)
    P.write_text(text)
    import subprocess

    rc = subprocess.run(["bash", "-n", str(P)], capture_output=True, text=True)
    if rc.returncode != 0:
        shutil.copy2(backup, P)
        raise SystemExit(f"bash -n failed, reverted: {rc.stderr}")
    print(f"vsh-manual-serve.sh patched; backup={backup.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
