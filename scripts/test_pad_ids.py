# padding rows: expert id -1 (VLLM_MOE_SKIP_PADDING) must contribute nothing and never fault
import os, sys, types, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import vsh_moe_int4 as vm
from vllm.model_executor.layers.quantization.utils.mxfp4_utils import _swizzle_mxfp4
import vllm.model_executor.layers.fused_moe.experts.gpt_oss_triton_kernels_moe as G
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import _mx_scale_kwargs
from vllm.model_executor.layers.fused_moe.utils import swiglu_limit_func
from triton_kernels.matmul import PrecisionConfig, FlexCtx
torch.manual_seed(3); dev = "cuda"; E, H, I, TOPK, L = 64, 4096, 1024, 6, 10.0
def mk(N, K):
    v = torch.randint(0, 256, (E, N, K // 2), dtype=torch.uint8, device=dev)
    s = torch.randint(119, 124, (E, N, K // 32), dtype=torch.uint8, device=dev)
    return _swizzle_mxfp4(v, s)
w1, f1, s1 = mk(2 * I, H); w2, f2, s2 = mk(H, I)
p1 = PrecisionConfig(**_mx_scale_kwargs(s1), flex_ctx=FlexCtx(rhs_data=f1)); p2 = PrecisionConfig(**_mx_scale_kwargs(s2), flex_ctx=FlexCtx(rhs_data=f2))
plan = vm.mxfp4_plan(w1, w2, types.SimpleNamespace(w1_bias=None, w2_bias=None, w1_precision=p1, w2_precision=p2, gemm1_clamp_limit=L))
def stock(x, tw, ids):
    M = x.shape[0]; rd, gi, si = G.make_routing_data(ids, tw, E)
    inter = G.matmul(x, w1, None, rd.ragged, None, rd.gather_tok, None, p1, gammas=None)
    act = torch.empty((rd.n_valid, I), dtype=x.dtype, device=dev); swiglu_limit_func(act, inter, L)
    down = G.matmul(act, w2, None, rd.ragged, None, None, None, p2, gammas=rd.gate_scal)
    acc = torch.zeros((M, H), dtype=torch.float32, device=dev); acc.index_add_(0, rd.gather_tok.to(torch.int64), down.to(torch.float32))
    return acc.to(torch.bfloat16)
for ver in ("1", "3"):
    os.environ["VSH_MOE_MXFP4_V"] = ver
    for M, npad in ((2, 1), (6, 2), (8, 3)):
        x = (torch.randn(M, H, device=dev) * 0.5).bfloat16()
        ids = torch.stack([torch.randperm(E, device=dev)[:TOPK] for _ in range(M)]).int()
        tw = torch.rand(M, TOPK, device=dev)
        ids[M - npad:] = -1; tw[M - npad:] = 0                 # padded rows, as the router marks them
        out = torch.empty(M, H, dtype=torch.bfloat16, device=dev)
        got = vm.mxfp4_direct_moe(out, x, plan, tw, ids).float(); torch.cuda.synchronize()
        ref = stock(x, tw, ids).float()
        real = slice(0, M - npad)
        rel = ((got[real] - ref[real]).norm() / ref[real].norm()).item()
        print(f"mxfp4 v{ver} M={M} pad={npad}: real rows rel {rel:.1e}, padded rows max |x| {got[M-npad:].abs().max().item():.1e}")
# int4 (GLM) direct path with -1 slots: zero contribution vs the same call with weights zeroed
Ei, Hi, Ii = 32, 4096, 1024
w13 = torch.randint(0, 256, (Ei, 2 * Ii, Hi // 2), dtype=torch.uint8, device=dev); w2i = torch.randint(0, 256, (Ei, Hi, Ii // 2), dtype=torch.uint8, device=dev)
s13 = (torch.rand(Ei, 2 * Ii, Hi // 128, device=dev) * 0.004 + 0.001).bfloat16(); s2i = (torch.rand(Ei, Hi, Ii // 128, device=dev) * 0.004 + 0.001).bfloat16()
x = torch.randn(4, Hi, device=dev).bfloat16(); ids = torch.stack([torch.randperm(Ei, device=dev)[:8] for _ in range(4)]).int(); tw = torch.rand(4, 8, device=dev)
ids_p = ids.clone(); tw_p = tw.clone(); ids_p[3] = -1; tw_p[3] = 0
a = vm.direct_moe(x, w13, w2i, s13, s2i, tw_p, ids_p, 10.0).float()
tw_z = tw.clone(); tw_z[3] = 0
b = vm.direct_moe(x, w13, w2i, s13, s2i, tw_z, ids, 10.0).float(); torch.cuda.synchronize()
print(f"int4 direct with -1 row: rows 0-2 rel {((a[:3]-b[:3]).norm()/b[:3].norm()).item():.1e}, padded row max |x| {a[3].abs().max().item():.1e}")
