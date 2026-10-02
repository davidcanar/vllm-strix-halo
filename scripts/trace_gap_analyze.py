#!/usr/bin/env python3
"""Where does a decode step actually spend its wall time?

Reads the torch-profiler chrome trace captured from the TP0 worker and reports,
per step: GPU-busy time, wall time, the idle gaps and which host operations were
running during them. Offline analysis - no server impact.
"""
from __future__ import annotations

import collections
import gzip
import json
import sys

path = sys.argv[1]
with gzip.open(path, "rt") as fh:
    data = json.load(fh)
ev = data["traceEvents"] if isinstance(data, dict) else data
print(f"events: {len(ev)}")

kernels = []
cpu_ops = []
api = []
ann = []
for e in ev:
    if e.get("ph") != "X":
        continue
    cat = e.get("cat", "")
    dur = e.get("dur") or 0
    if cat in ("kernel", "Kernel", "gpu_memcpy", "gpu_memset"):
        kernels.append((e["ts"], dur, e.get("name", "?"), cat))
    elif cat in ("cuda_runtime", "hip_runtime", "runtime"):
        api.append((e["ts"], dur, e.get("name", "?")))
    elif cat in ("cpu_op", "python_function"):
        cpu_ops.append((e["ts"], dur, e.get("name", "?")))
    elif cat in ("user_annotation",):
        ann.append((e["ts"], dur, e.get("name", "?")))

kernels.sort()
print(f"kernels={len(kernels)} cpu_ops={len(cpu_ops)} api={len(api)} annotations={len(ann)}")

if not kernels:
    sys.exit("no kernel events")

t0 = kernels[0][0]
t1 = max(ts + d for ts, d, _, _ in kernels)
wall = (t1 - t0) / 1e6
busy = sum(d for _, d, _, _ in kernels) / 1e6
print(f"kernel timeline: 0..{wall:.2f} s, kernel self time {busy:.2f} s "
      f"({busy / wall * 100:.1f}% of the span)")

# merge overlapping kernel intervals -> true GPU-busy time
merged = []
for ts, d, _, _ in kernels:
    if merged and ts <= merged[-1][1]:
        merged[-1][1] = max(merged[-1][1], ts + d)
    else:
        merged.append([ts, ts + d])
gpu_busy = sum(b - a for a, b in merged) / 1e6
print(f"GPU busy (merged): {gpu_busy:.2f} s -> idle {(wall - gpu_busy):.2f} s "
      f"({(1 - gpu_busy / wall) * 100:.1f}% idle)")

# gaps > 2 ms between merged kernel intervals
gaps = []
for (a1, b1), (a2, _) in zip(merged, merged[1:]):
    if a2 - b1 > 2000:
        gaps.append((b1, a2 - b1))
gaps.sort(key=lambda g: -g[1])
total_gap = sum(d for _, d in gaps) / 1e6
print(f"\ngaps >2 ms: {len(gaps)}, total {total_gap:.2f} s "
      f"({total_gap / wall * 100:.1f}% of the span); top 10:")

# what host work overlaps each gap
def overlap(ts, dur, table):
    end = ts + dur
    out = collections.Counter()
    for ots, odur, name in table:
        if ots < end and ots + odur > ts:
            o = min(end, ots + odur) - max(ts, ots)
            if o > 0:
                out[name] += o
    return out

agg = collections.Counter()
for gts, gdur in gaps[:200]:
    for name, o in overlap(gts, gdur, api).items():
        agg[f"API {name}"] += o
    for name, o in overlap(gts, gdur, cpu_ops).items():
        agg[f"CPU {name}"] += o
for name, us in agg.most_common(15):
    print(f"   {us / 1000:9.1f} ms  {name}")

print("\ntop API calls by total host time:")
api_tot = collections.Counter()
for _, d, name in api:
    api_tot[name] += d
for name, us in api_tot.most_common(12):
    print(f"   {us / 1000:9.1f} ms  {name}")

print("\ntop CPU ops by total host time:")
cpu_tot = collections.Counter()
for _, d, name in cpu_ops:
    cpu_tot[name] += d
for name, us in cpu_tot.most_common(12):
    print(f"   {us / 1000:9.1f} ms  {name}")
