"""DS4 decode trace: step wall vs GPU busy, top kernels, host self-time by op.

usage: python ds4trace.py rank0.pt.trace.json.gz
"""
import bisect
import collections
import gzip
import json
import statistics as st
import sys

with gzip.open(sys.argv[1], "rt") as fh:
    ev = json.load(fh)["traceEvents"]
K, ANN, OPS, RT = [], [], [], []
for e in ev:
    if e.get("ph") != "X":
        continue
    c = e.get("cat")
    d = e.get("dur", 0)
    if c in ("kernel", "gpu_memcpy", "gpu_memset"):
        K.append((e["ts"], d, e["name"]))
    elif c == "user_annotation":
        ANN.append((e["ts"], d, e["name"], e.get("tid")))
    elif c == "cpu_op":
        OPS.append((e["ts"], d, e["name"], e.get("tid"), e.get("pid")))
    elif c == "cuda_runtime":
        RT.append((e["ts"], d, e["name"], e.get("tid")))
K.sort(); ANN.sort(); OPS.sort(); RT.sort()
names = collections.Counter(a[2].split("(")[0] for a in ANN)
print("annotations:", names.most_common(8))
steps = [a for a in ANN if a[2].startswith("execute_context")]
if not steps:
    sys.exit("no execute_context annotations")
starts = [a[0] for a in steps]
per = [b - a for a, b in zip(starts, starts[1:])]
med = st.median(per)
sel = [(starts[i], starts[i + 1], steps[i][2]) for i in range(len(per)) if per[i] < 1.3 * med]
print(f"steps {len(steps)}, period median {med/1e3:.1f} ms; steady {len(sel)}; e.g. {sel[0][2][:60]}")
kts = [k[0] for k in K]
busy, nk = [], []
ktime = collections.Counter(); kcount = collections.Counter()
for a, b, _ in sel:
    i, j = bisect.bisect_left(kts, a), bisect.bisect_left(kts, b)
    iv = sorted((k[0], k[0] + k[1]) for k in K[i:j])
    m = []
    for x, y in iv:
        if m and x <= m[-1][1]:
            m[-1][1] = max(m[-1][1], y)
        else:
            m.append([x, y])
    busy.append(sum(y - x for x, y in m)); nk.append(j - i)
    for k in K[i:j]:
        nm = k[2].split("(")[0][:70]
        ktime[nm] += k[1]; kcount[nm] += 1
n = len(sel)
print(f"per step: wall {st.median([b-a for a,b,_ in sel])/1e3:.1f} ms, GPU busy {st.median(busy)/1e3:.1f} ms, kernels {st.median(nk):.0f}")
print("top kernels (ms/step, calls/step):")
for nm, t in ktime.most_common(14):
    print(f"  {t/n/1e3:7.2f}  {kcount[nm]/n:6.0f}  {nm}")
# host self time on the main thread (thread with most cpu_ops in the window)
lo, hi = sel[0][0], sel[-1][1]
win = [o for o in OPS if lo <= o[0] < hi]
tid = collections.Counter(o[3] for o in win).most_common(1)[0][0]
ops = sorted([o for o in win if o[3] == tid], key=lambda o: (o[0], -o[1]))
child = [0] * len(ops); stack = []
for i, o in enumerate(ops):
    while stack and ops[stack[-1]][0] + ops[stack[-1]][1] <= o[0]:
        stack.pop()
    if stack:
        child[stack[-1]] += o[1]
    stack.append(i)
selft = collections.Counter(); calls = collections.Counter()
for i, o in enumerate(ops):
    selft[o[2]] += o[1] - child[i]; calls[o[2]] += 1
rts = [r for r in RT if lo <= r[0] < hi and r[3] == tid]
rt_t = collections.Counter(); rt_c = collections.Counter()
for r in rts:
    rt_t[r[2]] += r[1]; rt_c[r[2]] += 1
tot_self = sum(selft.values())
print(f"main-thread cpu_op self time: {tot_self/n/1e3:.1f} ms/step over {len(ops)/n:.0f} ops/step; "
      f"runtime API {sum(rt_t.values())/n/1e3:.1f} ms/step ({len(rts)/n:.0f} calls/step)")
print("top host self time (ms/step, calls/step):")
for nm, t in selft.most_common(18):
    print(f"  {t/n/1e3:7.2f}  {calls[nm]/n:6.0f}  {nm[:80]}")
print("top runtime API (ms/step, calls/step):")
for nm, t in rt_t.most_common(6):
    print(f"  {t/n/1e3:7.2f}  {rt_c[nm]/n:6.0f}  {nm}")
