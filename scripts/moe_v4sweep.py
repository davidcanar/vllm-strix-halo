# MXFP4 v3 vs v4 (row-pair passes per wave) at decode shapes, E=256; best-of-5 timing
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
def bench(fn):
    for _ in range(5): fn()
    best = 1e9
    for _ in range(5):
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(30): fn()
        torch.cuda.synchronize(); best = min(best, (time.perf_counter() - t) / 30 * 1e3)
    return best
configs = [("v3", {"VSH_MOE_MXFP4_V": "3"})] + [
    (f"v4 rp1={a} rp2={b}", {"VSH_MOE_MXFP4_V": "4", "VSH_MX_RP1": str(a), "VSH_MX_RP2": str(b)})
    for a in (1, 2, 4) for b in (1, 2, 4)]
for M, mode in ((1, "random"), (6, "random"), (6, "half"), (6, "overlap")):
    x = (torch.randn(M, H, device=dev) * 0.5).bfloat16(); base = torch.randperm(E, device=dev)[:TOPK]
    ids = torch.stack([base if (mode == "overlap" or (mode == "half" and t % 2)) else torch.randperm(E, device=dev)[:TOPK]
                       for t in range(M)]).int().contiguous()
    tw = torch.rand(M, TOPK, device=dev)
    res, ref = [], None
    for name, env in configs:
        os.environ.update(env)
        out = torch.empty(M, H, dtype=torch.bfloat16, device=dev)
        ms = bench(lambda: vm.mxfp4_direct_moe(out, x, plan, tw, ids))
        o = out.float().clone()
        if ref is None: ref = o
        res.append((ms, name, ((o - ref).norm() / ref.norm()).item()))
    best = min(res)
    print(f"M={M} {mode:7s}: v3 {res[0][0]:.3f} ms | best {best[1]} {best[0]:.3f} ms ({res[0][0]/best[0]:.2f}x, rel vs v3 {best[2]:.0e}) | "
          + " ".join(f"{n.replace('v4 ', '')}={m:.2f}" for m, n, _ in res[1:]))
