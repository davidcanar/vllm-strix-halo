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
plan = vm.mxfp4_plan(w1, w2, qc); os.environ["VSH_MOE_MXFP4_V"] = "3"
for mode in ("random", "half", "overlap"):
    M = 6; x = (torch.randn(M, H, device=dev) * 0.5).bfloat16(); base = torch.randperm(E, device=dev)[:TOPK]
    ids = torch.stack([base if (mode == "overlap" or (mode == "half" and t % 2)) else torch.randperm(E, device=dev)[:TOPK] for t in range(M)]).int().contiguous()
    tw = torch.rand(M, TOPK, device=dev); line = []
    for mt in ("1", "2", "4", "6"):
        os.environ["VSH_MX_MT"] = mt
        out = torch.empty(M, H, dtype=torch.bfloat16, device=dev); fn = lambda: vm.mxfp4_direct_moe(out, x, plan, tw, ids)
        for _ in range(5): fn()
        best = 1e9
        for rep in range(5):
            torch.cuda.synchronize(); t = time.perf_counter()
            for _ in range(30): fn()
            torch.cuda.synchronize(); best = min(best, (time.perf_counter()-t)/30*1e3)
        line.append(f"mt={mt}: {best:.3f} ms")
    print(f"M=6 {mode:7s}: " + " | ".join(line))
