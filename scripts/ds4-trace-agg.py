#!/usr/bin/env python3
"""Aggregate the DS4 no-spec decode trace: top GPU kernels by total time."""
import gzip, json, collections, sys

path = sys.argv[1]
with gzip.open(path, "rt") as f:
    data = json.load(f)
events = data.get("traceEvents", [])
kernels = collections.defaultdict(lambda: [0, 0.0])
total_kernel = 0.0
for e in events:
    if e.get("ph") == "X" and e.get("cat", "") in ("kernel", "gpu_op", "Kernel"):
        name = e["name"][:110]
        dur = e.get("dur", 0)
        kernels[name][0] += 1
        kernels[name][1] += dur
        total_kernel += dur
print(f"total kernel time: {total_kernel/1e3:.1f} ms over the window")
rows = sorted(kernels.items(), key=lambda kv: -kv[1][1])[:22]
for name, (cnt, dur) in rows:
    print(f"{dur/1e3:9.1f} ms  x{cnt:6d}  {dur/cnt:8.1f} us/inv  {name}")
