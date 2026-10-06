"""DS4 prefill trace: GPU time by kernel family over the whole profiled window (no step selection).

usage: python ds4prefill.py rank0.pt.trace.json.gz
"""
import collections
import gzip
import json
import sys

with gzip.open(sys.argv[1], "rt") as fh:
    ev = json.load(fh)["traceEvents"]
K = [e for e in ev if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
if not K:
    sys.exit("no kernels")
t0 = min(e["ts"] for e in K); t1 = max(e["ts"] + e.get("dur", 0) for e in K)
iv = sorted((e["ts"], e["ts"] + e.get("dur", 0)) for e in K)
busy, cur = 0.0, None
for a, b in iv:
    if cur is None or a > cur[1]:
        if cur:
            busy += cur[1] - cur[0]
        cur = [a, b]
    else:
        cur[1] = max(cur[1], b)
busy += cur[1] - cur[0]
FAM = [("moe (triton_kernels matmul / routing)", ("_matmul_ogs", "matmul_ogs", "_topk", "routing", "_combined_routing", "_finalize")),
       ("moe (vsh direct)", ("moe_mxfp4", "moe_swiglu", "moe_topk_sum")),
       ("fp8 linear (triton block-scaled)", ("_w8a8_triton_block_scaled_mm", "block_scaled")),
       ("fp8 gemv (vsh)", ("fp8_gemv",)),
       ("indexer / sparse attention", ("indexer", "sparse", "mqa", "_topk_kernel", "paged_logits", "attn")),
       ("hipBLASLt / rocBLAS", ("Cijk_", "wvSplitK")),
       ("all-reduce / comm", ("odl2", "nccl", "rccl", "allreduce")),
       ("mhc / norms / elementwise", ("mhc", "rms", "norm", "elementwise", "vectorized", "reduce", "copy", "cat")),
       ]
fam_t = collections.Counter(); name_t = collections.Counter(); name_c = collections.Counter()
for e in K:
    n = e["name"]; d = e.get("dur", 0)
    name_t[n.split("(")[0][:80]] += d; name_c[n.split("(")[0][:80]] += 1
    for f, keys in FAM:
        if any(k.lower() in n.lower() for k in keys):
            fam_t[f] += d
            break
    else:
        fam_t["other"] += d
tot = sum(name_t.values())
print(f"window {(t1 - t0) / 1e3:.0f} ms, GPU busy {busy / 1e3:.0f} ms ({100 * busy / (t1 - t0):.0f} %), kernel time {tot / 1e3:.0f} ms")
for f, t in fam_t.most_common():
    print(f"  {100 * t / tot:5.1f} %  {t / 1e3:8.1f} ms  {f}")
print("top kernels:")
for n, t in name_t.most_common(14):
    print(f"  {100 * t / tot:5.1f} %  {t / 1e3:8.1f} ms  {name_c[n]:6d}x  {n}")
