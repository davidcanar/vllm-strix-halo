#!/usr/bin/env python3
"""Enumerate device->host copies (pipeline drains) in a torch trace -- fast path.

usage: trace_dtoh.py <trace.json.gz>
"""
from __future__ import annotations

import bisect
import collections
import gzip
import json
import sys

with gzip.open(sys.argv[1], "rt") as fh:
    data = json.load(fh)
ev = data["traceEvents"] if isinstance(data, dict) else data

memcpys = [
    e
    for e in ev
    if e.get("ph") == "X"
    and (e.get("cat") in ("gpu_memcpy", "gpu_memset") or "Memcpy" in (e.get("name") or ""))
]
print(f"memcpy-ish events: {len(memcpys)}")
kinds = collections.Counter()
for e in memcpys:
    a = e.get("args") or {}
    kinds[(e.get("name", "?"), a.get("kind"))] += 1
for (name, kind), n in kinds.most_common(10):
    print(f"  {n:6d}  {name:34s} kind={kind}")

kernels = [e for e in ev if e.get("ph") == "X" and e.get("cat") in ("kernel", "Kernel")]
t0 = min(e["ts"] for e in kernels)
t1 = max(e["ts"] + (e.get("dur") or 0) for e in kernels)
span = (t1 - t0) / 1e6
print(f"\nkernel span {span:.2f} s over {len(kernels)} kernels ({len(kernels) / span:.0f} kernels/s)")

dtoh = [e for e in memcpys if (e.get("args") or {}).get("kind") == 2 or (e.get("name") or "").startswith("Memcpy DtoH")]
print(f"\nDtoH events: {len(dtoh)}  ({len(dtoh) / span:.1f} per second of span)")
if dtoh:
    per_sec = collections.Counter(int((e["ts"] - t0) // 1_000_000) for e in dtoh)
    print("DtoH per second:", dict(sorted(per_sec.items())[:25]))
    print("DtoH sizes (bytes):", collections.Counter((e.get("args") or {}).get("bytes") for e in dtoh).most_common(6))

# issuer: nearest preceding CPU-side event, using bisect on sorted start times
cpu = sorted(
    (
        e
        for e in ev
        if e.get("ph") == "X"
        and e.get("cat") in ("cpu_op", "python_function", "cuda_runtime", "user_annotation", "hip_runtime")
    ),
    key=lambda e: e["ts"],
)
starts = [e["ts"] for e in cpu]
if dtoh and cpu:
    names = collections.Counter()
    for e in dtoh:
        i = bisect.bisect_right(starts, e["ts"]) - 1
        while i >= 0 and cpu[i]["ts"] + (cpu[i].get("dur") or 0) < e["ts"] - 2000:
            i -= 1
        if i >= 0:
            names[(cpu[i].get("cat"), cpu[i].get("name"))] += 1
    print("\ntop issuers (nearest preceding CPU event):")
    for (cat, name), n in names.most_common(12):
        print(f"  {n:6d}  [{cat}] {name}")
