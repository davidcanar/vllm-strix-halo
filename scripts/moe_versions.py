# whole-call timing of MXFP4 direct MoE versions (E=256, DS4 TP=2 shapes) + correctness vs v1
import os, sys, time, types, torch
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
plan = vm.mxfp4_plan(w1, w2, qc)
for M, mode in ((1, "random"), (4, "random"), (6, "random"), (6, "half"), (6, "overlap")):
    x = (torch.randn(M, H, device=dev) * 0.5).bfloat16()
    base = torch.randperm(E, device=dev)[:TOPK]
    rows = []
    for t in range(M):
        if mode == "overlap" or (mode == "half" and t % 2 == 1): rows.append(base)
        else: rows.append(torch.randperm(E, device=dev)[:TOPK])
    ids = torch.stack(rows).int().contiguous(); tw = torch.rand(M, TOPK, device=dev)
    res, outs = {}, {}
    for ver in ("1", "3"):
        os.environ["VSH_MOE_MXFP4_V"] = ver
        out = torch.empty(M, H, dtype=torch.bfloat16, device=dev)
        fn = lambda: vm.mxfp4_direct_moe(out, x, plan, tw, ids)
        for _ in range(5): fn()
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(50): fn()
        torch.cuda.synchronize(); res[ver] = (time.perf_counter() - t) / 50 * 1e3
        outs[ver] = out.float().clone()
    rel = ((outs["3"] - outs["1"]).norm() / outs["1"].norm()).item()
    uniq = len(set(ids.flatten().tolist()))
    print(f"M={M} {mode:7s} uniq={uniq:2d}: v1 {res['1']:.3f} ms | v3 {res['3']:.3f} ms -> {res['1']/res['3']:.2f}x | v3 vs v1 rel {rel:.1e} | x43: {res['1']*43:.0f} -> {res['3']*43:.0f} ms/step")
