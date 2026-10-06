# MXFP4 direct MoE on REAL DS4 decode routing (moe_dump_ids.py dump): routing stats + per-phase v4/v5 timing.
# Each timed call uses the next real routing set (experts come from DRAM, as layer after layer in decode).
import glob, os, sys, types, collections, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import vsh_moe_int4 as vm
from vllm.model_executor.layers.quantization.utils.mxfp4_utils import _swizzle_mxfp4
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import _mx_scale_kwargs
from triton_kernels.matmul import PrecisionConfig, FlexCtx
# IDS: a moe_dump_ids.py dump (list of [M, top_k] tensors) or the compact
# scripts/data/ds4_dspark_routing.pt ({M: [n, M, top_k] int16}, DSpark k=5 decode, 2026-10-06)
dumps = sorted(glob.glob(os.environ.get("IDS", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                            "data", "ds4_dspark_routing.pt"))))
obj = torch.load(dumps[0])               # one rank is enough (TP ranks route identically)
calls = [c for m in sorted(obj) for c in obj[m]] if isinstance(obj, dict) else obj
by_m = collections.defaultdict(list)
skip = 0 if isinstance(obj, dict) else int(os.environ.get("SKIP", "600"))   # boot-time warmup calls
for c in calls[skip:]:
    by_m[c.shape[0]].append(c.int())
print("dump:", dumps[:1], "calls", len(calls), "by M:", {m: len(v) for m, v in sorted(by_m.items())})
for m, lst in sorted(by_m.items()):
    Ds = [torch.unique(c[c >= 0]).numel() for c in lst]
    mult = collections.Counter()
    for c in lst:
        u, n = torch.unique(c[c >= 0], return_counts=True); mult.update(n.tolist())
    tot = sum(mult.values())
    print(f"  M={m}: calls {len(lst)}, distinct experts D mean {sum(Ds)/len(Ds):.1f} (min {min(Ds)}, max {max(Ds)}); "
          f"tokens per expert: " + " ".join(f"{k}:{100*v/tot:.0f}%" for k, v in sorted(mult.items())))
torch.manual_seed(0); dev = "cuda"; E, H, I, TOPK = 256, 4096, 1024, 6
def mk(N, K):
    v = torch.randint(0, 256, (E, N, K // 2), dtype=torch.uint8, device=dev)
    s = torch.randint(119, 124, (E, N, K // 32), dtype=torch.uint8, device=dev)
    return _swizzle_mxfp4(v, s)
w1, f1, s1 = mk(2 * I, H); w2, f2, s2 = mk(H, I)
qc = types.SimpleNamespace(w1_bias=None, w2_bias=None, gemm1_clamp_limit=10.0,
    w1_precision=PrecisionConfig(**_mx_scale_kwargs(s1), flex_ctx=FlexCtx(rhs_data=f1)),
    w2_precision=PrecisionConfig(**_mx_scale_kwargs(s2), flex_ctx=FlexCtx(rhs_data=f2)))
plan = vm.mxfp4_plan(w1, w2, qc); a, b = plan["1"], plan["2"]; lib = vm._lib()
vm.mxfp4_direct_moe(torch.empty(1, H, dtype=torch.bfloat16, device=dev), torch.zeros(1, H, dtype=torch.bfloat16, device=dev), plan,
                    torch.ones(1, TOPK, device=dev), torch.arange(TOPK, device=dev, dtype=torch.int32).view(1, TOPK))
st = torch.cuda.current_stream().cuda_stream
def tm(fn, n):
    for i in range(n): fn(i)
    best = 1e9
    for _ in range(3):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for i in range(n): fn(i)
        e.record(); torch.cuda.synchronize(); best = min(best, s.elapsed_time(e) / n)
    return best
CF = [c.rsplit(":", 1) for c in os.environ.get("CFGS", "v4:4:2/4,v5:4:41/41,v5:4:141/141,v5:4:41/141,v5:8:141/141,v5:8:41/141").split(",")]
for m, lst in sorted(by_m.items()):
    if len(lst) < 50 or m > 10:
        continue
    lst = lst[:960]; n = len(lst); P = m * TOPK
    ids_all = torch.stack(lst).to(dev).contiguous()
    x = (torch.randn(m, H, device=dev) * 0.5).bfloat16(); tw = torch.rand(m, TOPK, device=dev)
    c1 = torch.empty(P, a["N"], dtype=torch.bfloat16, device=dev); act = (torch.randn(P, b["K"], device=dev) * 0.5).bfloat16()
    c3 = torch.empty(P, b["N"], dtype=torch.bfloat16, device=dev)
    res = []
    for ver_mt, r in CF:
        ver, mt = ver_mt.split(":")[0], int(ver_mt.split(":")[1])
        r1, r2 = [int(v) for v in r.split("/")]
        mt = min(mt, 1 if m == 1 else 2 if m == 2 else 8)
        fn = lib.vsh_moe_mxfp4_v5 if ver == "v5" else lib.vsh_moe_mxfp4_v4
        k1 = lambda i: fn(st, 1, mt, r1, x.data_ptr(), a["v"].data_ptr(), c1.data_ptr(), a["s"].data_ptr(), tw.data_ptr(),
                          ids_all[i].data_ptr(), P, a["N"], a["K"], TOPK, x.stride(0), a["sbe"], a["sbn"], c1.stride(0),
                          a["sse"], a["ssn"], a["ssg"], a["E"])
        k2 = lambda i: fn(st, 2, mt, r2, act.data_ptr(), b["v"].data_ptr(), c3.data_ptr(), b["s"].data_ptr(), tw.data_ptr(),
                          ids_all[i].data_ptr(), P, b["N"], b["K"], TOPK, act.stride(0), b["sbe"], b["sbn"], c3.stride(0),
                          b["sse"], b["ssn"], b["ssg"], b["E"])
        assert k1(0) == 0 and k2(0) == 0, (ver, mt, r1, r2)
        t1, t2 = tm(k1, n), tm(k2, n)
        res.append((t1 + t2, f"{ver} mt{mt} {r1}/{r2}", t1, t2))
    base = res[0][0]
    print(f"M={m} real routing ({n} calls):", flush=True)
    for tot, name, t1, t2 in res:
        print(f"    {name:18s} w13 {t1*1e3:6.1f} us  w2 {t2*1e3:6.1f} us  sum {tot*1e3:6.1f} us  ({base/tot:.2f}x vs v4)", flush=True)
