# per-phase timing of the MXFP4 direct MoE (w13 / w2), v4 vs v5 configs, E=256. Each timed call uses a
# different routing set (48 sets cycled) so the experts come from DRAM, as in decode (layer after layer).
import os, sys, time, types, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import vsh_moe_int4 as vm
from vllm.model_executor.layers.quantization.utils.mxfp4_utils import _swizzle_mxfp4
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import _mx_scale_kwargs
from triton_kernels.matmul import PrecisionConfig, FlexCtx
torch.manual_seed(0); dev = "cuda"; E, H, I, TOPK, NS = 256, 4096, 1024, 6, 48
def mk(N, K):
    v = torch.randint(0, 256, (E, N, K // 2), dtype=torch.uint8, device=dev)
    s = torch.randint(119, 124, (E, N, K // 32), dtype=torch.uint8, device=dev)
    return _swizzle_mxfp4(v, s)
w1, f1, s1 = mk(2 * I, H); w2, f2, s2 = mk(H, I)
qc = types.SimpleNamespace(w1_bias=None, w2_bias=None, gemm1_clamp_limit=10.0,
    w1_precision=PrecisionConfig(**_mx_scale_kwargs(s1), flex_ctx=FlexCtx(rhs_data=f1)),
    w2_precision=PrecisionConfig(**_mx_scale_kwargs(s2), flex_ctx=FlexCtx(rhs_data=f2)))
plan = vm.mxfp4_plan(w1, w2, qc); a, b = plan["1"], plan["2"]
lib = vm._lib()
vm.mxfp4_direct_moe(torch.empty(1, H, dtype=torch.bfloat16, device=dev), torch.zeros(1, H, dtype=torch.bfloat16, device=dev), plan,
                    torch.ones(1, TOPK, device=dev), torch.arange(TOPK, device=dev, dtype=torch.int32).view(1, TOPK))
W13 = 2 * I * H // 2 + 2 * I * H // 32; W2 = H * I // 2 + H * I // 32
st = torch.cuda.current_stream().cuda_stream
def routing(M, mode, g):
    if mode == "random":
        return torch.stack([torch.randperm(E, generator=g)[:TOPK] for _ in range(M)])
    if mode == "overlap":
        return torch.randperm(E, generator=g)[:TOPK].repeat(M, 1)
    pool = torch.randperm(E, generator=g)[:int(mode[4:])]
    return torch.stack([pool[torch.randperm(pool.numel(), generator=g)[:TOPK]] for _ in range(M)])
def tm(fn):
    for i in range(NS): fn(i)
    best = 1e9
    for _ in range(5):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for i in range(NS): fn(i)
        e.record(); torch.cuda.synchronize(); best = min(best, s.elapsed_time(e) / NS)
    return best
CFGS = [(4, None, 2, 4)] + [(5, None, int(r1), int(r2)) for r1, r2 in
        (c.split("/") for c in os.environ.get("PH_CFGS", "41/41,141/141,141/142,122/122,142/141").split(","))]
for M, mode in ((1, "random"), (6, "random"), (6, "pool20"), (6, "pool12"), (6, "overlap")):
    g = torch.Generator(device="cpu").manual_seed(1234 + M)
    ids_all = torch.stack([routing(M, mode, g) for _ in range(NS)]).int().to(dev).contiguous()
    D = sum(torch.unique(ids_all[i]).numel() for i in range(NS)) / NS
    P = M * TOPK
    x = (torch.randn(M, H, device=dev) * 0.5).bfloat16(); tw = torch.rand(M, TOPK, device=dev)
    c1 = torch.empty(P, a["N"], dtype=torch.bfloat16, device=dev); act = (torch.randn(P, b["K"], device=dev) * 0.5).bfloat16()
    c3 = torch.empty(P, b["N"], dtype=torch.bfloat16, device=dev)
    for mts in ((1,) if M == 1 else (4, 8)):
        for ver, _, r1, r2 in CFGS:
            if ver == 4 and mts == 8:
                continue
            fn = lib.vsh_moe_mxfp4_v5 if ver == 5 else lib.vsh_moe_mxfp4_v4
            k1 = lambda i: fn(st, 1, mts, r1, x.data_ptr(), a["v"].data_ptr(), c1.data_ptr(), a["s"].data_ptr(), tw.data_ptr(),
                              ids_all[i].data_ptr(), P, a["N"], a["K"], TOPK, x.stride(0), a["sbe"], a["sbn"], c1.stride(0),
                              a["sse"], a["ssn"], a["ssg"], a["E"])
            k2 = lambda i: fn(st, 2, mts, r2, act.data_ptr(), b["v"].data_ptr(), c3.data_ptr(), b["s"].data_ptr(), tw.data_ptr(),
                              ids_all[i].data_ptr(), P, b["N"], b["K"], TOPK, act.stride(0), b["sbe"], b["sbn"], c3.stride(0),
                              b["sse"], b["ssn"], b["ssg"], b["E"])
            assert k1(0) == 0 and k2(0) == 0
            t1, t2 = tm(k1), tm(k2)
            print(f"M={M} {mode:7s} D~{D:4.1f} v{ver} mt{mts} {r1:3d}/{r2:3d}: w13 {t1*1e3:6.1f} us ({D*W13/t1/1e6:4.0f} GB/s)  "
                  f"w2 {t2*1e3:6.1f} us ({D*W2/t2/1e6:4.0f} GB/s)  sum {(t1+t2)*1e3:6.1f} us", flush=True)
