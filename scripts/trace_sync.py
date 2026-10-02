#!/usr/bin/env python3
"""Find blocking host ops (sync shape) in a torch trace: the long tail of
individual CPU op durations, which is where a .tolist()/.item() drain shows up.

usage: trace_sync.py <trace.json.gz>
"""
from __future__ import annotations

import collections
import gzip
import json
import sys

with gzip.open(sys.argv[1], "rt") as fh:
    data = json.load(fh)
ev = data["traceEvents"] if isinstance(data, dict) else data

cpu = [
    e
    for e in ev
    if e.get("ph") == "X"
    and e.get("cat") in ("cpu_op", "python_function", "cuda_runtime", "hip_runtime", "user_annotation")
    and (e.get("dur") or 0) > 0
]
print(f"cpu-ish events: {len(cpu)}")

by_name = collections.defaultdict(list)
for e in cpu:
    by_name[e.get("name", "?")].append(e.get("dur") or 0)

rows = []
for name, durs in by_name.items():
    durs.sort()
    rows.append((durs[-1], durs[len(durs) // 2], sum(durs), len(durs), name))
rows.sort(reverse=True)

print(f"\n{'max_us':>10} {'med_us':>9} {'total_ms':>10} {'n':>7}  name   (top 20 by worst-case)")
for mx, med, tot, n, name in rows[:20]:
    print(f"{mx:10.0f} {med:9.0f} {tot / 1000:10.1f} {n:7d}  {name[:70]}")

print("\nsync-shaped ops (to/copy/item/numpy/cpu), worst-case first:")
pat = ("to_copy", "aten::to", "aten::cpu", "item", "numpy", "tolist", "copy_", "_local_scalar")
sel = [r for r in rows if any(p in r[4] for p in pat)]
for mx, med, tot, n, name in sel[:15]:
    print(f"{mx:10.0f} {med:9.0f} {tot / 1000:10.1f} {n:7d}  {name[:70]}")

# how many ops exceed 5 ms (a drained pipeline) and what are they?
print("\nops with max duration > 5 ms:")
for mx, med, tot, n, name in rows:
    if mx > 5000:
        print(f"  max={mx / 1000:8.2f} ms  med={med / 1000:7.3f} ms  n={n:6d}  {name[:70]}")
