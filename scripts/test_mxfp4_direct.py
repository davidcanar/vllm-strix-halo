# vsh MXFP4 direct MoE vs the stock UnfusedOAITritonExperts 3.8 pipeline on the same swizzled weights
import os, sys, time, types, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import vsh_moe_int4 as vm
from vllm.model_executor.layers.quantization.utils.mxfp4_utils import _swizzle_mxfp4
import vllm.model_executor.layers.fused_moe.experts.gpt_oss_triton_kernels_moe as G
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import _mx_scale_kwargs
from vllm.model_executor.layers.fused_moe.utils import swiglu_limit_func
from triton_kernels.matmul import PrecisionConfig, FlexCtx
torch.manual_seed(0); dev = "cuda"
E, H, I, TOPK, L = 64, 4096, 1024, 6, 10.0
def mk(N, K):
    v = torch.randint(0, 256, (E, N, K // 2), dtype=torch.uint8, device=dev)
    s = torch.randint(119, 124, (E, N, K // 32), dtype=torch.uint8, device=dev)
    return _swizzle_mxfp4(v, s)
w1, f1, s1 = mk(2 * I, H)
w2, f2, s2 = mk(H, I)
p1 = PrecisionConfig(**_mx_scale_kwargs(s1), flex_ctx=FlexCtx(rhs_data=f1))
p2 = PrecisionConfig(**_mx_scale_kwargs(s2), flex_ctx=FlexCtx(rhs_data=f2))
qc = types.SimpleNamespace(w1_bias=None, w2_bias=None, w1_precision=p1, w2_precision=p2, gemm1_clamp_limit=L)
plan = vm.mxfp4_plan(w1, w2, qc)
assert plan is not None, "plan rejected the layout"
print("plan:", {k: (v["E"], v["K"], v["N"], v["sbe"], v["sbn"], v["sse"], v["ssg"], v["ssn"]) for k, v in plan.items() if k in "12"})

def stock(x, tw, ids):
    M = x.shape[0]
    rd, gi, si = G.make_routing_data(ids, tw, E)
    inter = G.matmul(x, w1, None, rd.ragged, None, rd.gather_tok, None, p1, gammas=None)
    act = torch.empty((rd.n_valid, I), dtype=x.dtype, device=dev)
    swiglu_limit_func(act, inter, L)
    down = G.matmul(act, w2, None, rd.ragged, None, None, None, p2, gammas=rd.gate_scal)
    acc = torch.zeros((M, H), dtype=torch.float32, device=dev)
    acc.index_add_(0, rd.gather_tok.to(torch.int64), down.to(torch.float32))
    return acc.to(torch.bfloat16)

worst = 0
for M in (1, 2, 4, 6, 8, 10):
    for trial in range(3):
        x = (torch.randn(M, H, device=dev) * 0.5).bfloat16()
        ids = torch.stack([torch.randperm(E, device=dev)[:TOPK] for _ in range(M)]).int()
        if trial == 1:                       # shared experts across tokens (spec-decode-like)
            ids[1:] = ids[0]
        tw = torch.rand(M, TOPK, device=dev)
        ref = stock(x, tw, ids).float()
        out = torch.empty(M, H, dtype=torch.bfloat16, device=dev)
        got = vm.mxfp4_direct_moe(out, x, plan, tw, ids).float()
        rel = ((got - ref).norm() / ref.norm()).item(); worst = max(worst, rel)
        if trial == 0: print(f"M={M:2d}: rel err vs stock {rel:.2e}")
print(f"worst rel err {worst:.2e}")
for M in (1, 6):
    x = (torch.randn(M, H, device=dev) * 0.5).bfloat16(); ids = torch.stack([torch.randperm(E, device=dev)[:TOPK] for _ in range(M)]).int(); tw = torch.rand(M, TOPK, device=dev)
    out = torch.empty(M, H, dtype=torch.bfloat16, device=dev)
    def tm(fn, it=30):
        for _ in range(3): fn()
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(it): fn()
        torch.cuda.synchronize(); return (time.perf_counter() - t) / it * 1e3
    a, b = tm(lambda: stock(x, tw, ids)), tm(lambda: vm.mxfp4_direct_moe(out, x, plan, tw, ids))
    print(f"timing M={M}: stock {a:.3f} ms | direct {b:.3f} ms -> {a/b:.1f}x  (x43 layers: {a*43:.0f} -> {b*43:.0f} ms/step)")
