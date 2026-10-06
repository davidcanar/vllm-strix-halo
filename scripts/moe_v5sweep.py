# MXFP4 v4 vs v5 (fp16 v_perm decode + dot2, K-chunked staging) at DS4 TP=2 decode shapes, E=256; best-of-5
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
EXP_MB = (2 * I * H // 2 + 2 * I * H // 32 + H * I // 2 + H * I // 32) / 1e6   # bytes per expert (w13 + w2 + scales)
def bench(fn):
    for _ in range(5): fn()
    best = 1e9
    for _ in range(5):
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(30): fn()
        torch.cuda.synchronize(); best = min(best, (time.perf_counter() - t) / 30 * 1e3)
    return best
def ids_for(M, mode):
    if mode == "random":
        return torch.stack([torch.randperm(E, device=dev)[:TOPK] for _ in range(M)]).int()
    if mode == "overlap":
        b = torch.randperm(E, device=dev)[:TOPK]; return b.repeat(M, 1).int()
    if mode == "half":
        b = torch.randperm(E, device=dev)[:TOPK]
        return torch.stack([b if t % 2 else torch.randperm(E, device=dev)[:TOPK] for t in range(M)]).int()
    pool = torch.randperm(E, device=dev)[:int(mode[4:])]     # poolNN: experts drawn from NN candidates
    return torch.stack([pool[torch.randperm(pool.numel(), device=dev)[:TOPK]] for _ in range(M)]).int()
cases = [(1, "random"), (6, "random"), (6, "pool20"), (6, "half"), (6, "pool12"), (6, "overlap")]
only = os.environ.get("SWEEP_CASES")
if only:
    cases = [c for c in cases if f"{c[0]}{c[1]}" in only.split(",")]
cfgs = [("v4", {"VSH_MOE_MXFP4_V": "4"})]
for mt in os.environ.get("SWEEP_MT", "4,8").split(","):
    for rc1 in os.environ.get("SWEEP_RC1", "22,41,42").split(","):
        for rc2 in os.environ.get("SWEEP_RC2", "41,42,44").split(","):
            cfgs.append((f"v5 mt{mt} {rc1}/{rc2}", {"VSH_MOE_MXFP4_V": "5", "VSH_MX5_MT": mt, "VSH_MX5_RC1": rc1, "VSH_MX5_RC2": rc2}))
for M, mode in cases:
    x = (torch.randn(M, H, device=dev) * 0.5).bfloat16(); ids = ids_for(M, mode)
    D = torch.unique(ids).numel(); tw = torch.rand(M, TOPK, device=dev)
    res, ref = [], None
    for name, env in cfgs:
        if M == 1 and "mt8" in name:
            continue
        for k in ("VSH_MX5_MT", "VSH_MX5_RC1", "VSH_MX5_RC2"):
            os.environ.pop(k, None)
        os.environ.update(env)
        if M == 1 and env.get("VSH_MX5_MT"):
            os.environ["VSH_MX5_MT"] = "1"
        out = torch.empty(M, H, dtype=torch.bfloat16, device=dev)
        ms = bench(lambda: vm.mxfp4_direct_moe(out, x, plan, tw, ids))
        o = out.float().clone()
        if ref is None: ref = o
        res.append((ms, name, ((o - ref).norm() / ref.norm()).item()))
    v4 = res[0][0]; best = min(res)
    print(f"M={M} {mode:7s} D={D:2d} ({D*EXP_MB:5.0f} MB): v4 {v4:.3f} ms ({D*EXP_MB/v4:4.0f} GB/s) | best {best[1]} {best[0]:.3f} ms "
          f"({D*EXP_MB/best[0]:4.0f} GB/s, {v4/best[0]:.2f}x, rel vs v4 {best[2]:.0e})", flush=True)
    print("    " + " ".join(f"[{n.replace('v5 ', '')}]={m:.3f}" for m, n, _ in res[1:]), flush=True)
    worst = max(r[2] for r in res)
    print(f"    worst rel vs v4 across configs: {worst:.1e}", flush=True)
