"""Which host op launched a kernel? kernel -> correlation -> runtime call -> enclosing cpu_ops.

usage: python ds4who.py trace.json.gz SUBSTRING [SUBSTRING ...]
"""
import bisect
import collections
import gzip
import json
import sys

with gzip.open(sys.argv[1], "rt") as fh:
    ev = json.load(fh)["traceEvents"]
pats = sys.argv[2:]
rt_by_corr, ops = {}, collections.defaultdict(list)
kern = []
for e in ev:
    if e.get("ph") != "X":
        continue
    c = e.get("cat")
    a = e.get("args", {})
    if c == "cuda_runtime":
        rt_by_corr[a.get("correlation")] = e
    elif c == "cpu_op":
        ops[e.get("tid")].append((e["ts"], e["ts"] + e.get("dur", 0), e["name"], a))
    elif c == "kernel" and any(p in e["name"] for p in pats):
        kern.append(e)
for t in ops:
    ops[t].sort()
for p in pats:
    ks = [k for k in kern if p in k["name"]]
    if not ks:
        print(f"== {p}: no kernels"); continue
    stacks = collections.Counter()
    for k in ks[: 300]:
        r = rt_by_corr.get(k.get("args", {}).get("correlation"))
        if r is None:
            continue
        lst = ops.get(r.get("tid"), [])
        i = bisect.bisect_right(lst, (r["ts"], 1e30))
        chain = [o for o in lst[max(0, i - 400):i] if o[0] <= r["ts"] <= o[1]]
        names = " > ".join(o[2] for o in chain[-4:])
        dims = ""
        for o in reversed(chain):
            d = o[3].get("Input Dims") or o[3].get("Input dims")
            if d:
                dims = str(d)[:120]; break
        stacks[(names, dims)] += 1
    print(f"== {p}: {len(ks)} kernels, {sum(k.get('dur',0) for k in ks)/1e3:.1f} ms total")
    for (names, dims), cnt in stacks.most_common(5):
        print(f"   {cnt:4d}x  {names[:150]}  {dims}")
