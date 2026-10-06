# time the 3 launches of the MXFP4 direct MoE separately (E=256, DS4 TP=2 shapes)
import ctypes, os, sys, time, types, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import vsh_moe_int4 as vm
from vllm.model_executor.layers.quantization.utils.mxfp4_utils import _swizzle_mxfp4
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import _mx_scale_kwargs
from triton_kernels.matmul import PrecisionConfig, FlexCtx
torch.manual_seed(0); dev = "cuda"; E, H, I, TOPK = 256, 4096, 1024, 6
def mk(N, K):
    v = torch.randint(0, 256, (E, N, K // 2), dtype=torch.uint8, device=dev)
    s = torch.randint(119, 124, (E, N, K // 32), dtype=torch.uint8, device=dev)
    return _swizzle_mxfp4(v, s)
w1, f1, s1 = mk(2 * I, H); w2, f2, s2 = mk(H, I)
qc = types.SimpleNamespace(w1_bias=None, w2_bias=None, gemm1_clamp_limit=10.0,
    w1_precision=PrecisionConfig(**_mx_scale_kwargs(s1), flex_ctx=FlexCtx(rhs_data=f1)),
    w2_precision=PrecisionConfig(**_mx_scale_kwargs(s2), flex_ctx=FlexCtx(rhs_data=f2)))
plan = vm.mxfp4_plan(w1, w2, qc); a, b = plan["1"], plan["2"]
lib = vm._lib(); out0 = torch.empty(1, H, dtype=torch.bfloat16, device=dev)
vm.mxfp4_direct_moe(out0, torch.randn(1, H, device=dev).bfloat16(), plan, torch.rand(1, TOPK, device=dev), torch.randint(0, E, (1, TOPK), device=dev, dtype=torch.int32))
st = torch.cuda.current_stream().cuda_stream
for M, overlap in ((1, False), (6, False), (6, True)):
    P = M * TOPK
    x = torch.randn(M, H, device=dev).bfloat16()
    base = torch.randperm(E, device=dev)[:TOPK]
    ids = torch.stack([base if overlap else torch.randperm(E, device=dev)[:TOPK] for _ in range(M)]).int().contiguous()
    tw = torch.rand(M, TOPK, device=dev)
    c1 = torch.empty(P, a["N"], dtype=torch.bfloat16, device=dev); c3 = torch.empty(P, b["N"], dtype=torch.bfloat16, device=dev); out = torch.empty(M, H, dtype=torch.bfloat16, device=dev)
    ph1 = lambda: lib.vsh_moe_mxfp4_direct(st, 1, x.data_ptr(), a["v"].data_ptr(), c1.data_ptr(), a["s"].data_ptr(), tw.data_ptr(), ids.data_ptr(), P, a["N"], a["K"], TOPK, 10.0, x.stride(0), a["sbe"], a["sbn"], c1.stride(0), a["sse"], a["ssn"], a["ssg"])
    ph2 = lambda: lib.vsh_moe_mxfp4_direct(st, 2, c1.data_ptr(), b["v"].data_ptr(), c3.data_ptr(), b["s"].data_ptr(), tw.data_ptr(), ids.data_ptr(), P, b["N"], b["K"], TOPK, 10.0, c1.stride(0), b["sbe"], b["sbn"], c3.stride(0), b["sse"], b["ssn"], b["ssg"])
    ph3 = lambda: lib.vsh_moe_topk_sum(st, c3.data_ptr(), out.data_ptr(), M, b["N"], TOPK, c3.stride(0), out.stride(0))
    uniq = len(set(ids.flatten().tolist()))
    res = []
    for nm, fn in (("w13", ph1), ("w2", ph2), ("sum", ph3)):
        for _ in range(5): fn()
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(100): fn()
        torch.cuda.synchronize(); res.append((nm, (time.perf_counter() - t) / 100 * 1e6))
    mb1 = uniq * a["N"] * a["K"] / 2 / 1e6; mb2 = uniq * b["N"] * b["K"] / 2 / 1e6
    print(f"M={M} pairs={P} unique experts={uniq} {'(overlap)' if overlap else ''}: " + ", ".join(f"{n} {u:.0f} us" for n, u in res)
          + f" | w13 {mb1/res[0][1]*1e3:.0f} GB/s, w2 {mb2/res[1][1]*1e3:.0f} GB/s")
