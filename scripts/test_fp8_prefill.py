# vsh fp8 prefill path (dequant + hipBLASLt) vs fp32 reference and the stock Triton block GEMM
import os, sys, time, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import vsh_w8a16 as w8
from vllm.model_executor.layers.quantization.utils.fp8_utils import w8a8_triton_block_scaled_mm, per_token_group_quant_fp8
torch.manual_seed(0); dev = "cuda"
def tm(fn, it=10):
    for _ in range(3): fn()
    torch.cuda.synchronize(); best = 1e9
    for _ in range(3):
        t = time.perf_counter()
        for _ in range(it): fn()
        torch.cuda.synchronize(); best = min(best, (time.perf_counter() - t) / it)
    return best * 1e3
worst = 0
for (N, K, ld) in ((1536, 4096, 4096), (16384, 1024, 1024), (4096, 4096, 4096), (4096, 1024, 1024), (2048, 4096, 4096), (4096, 12288, 12544)):
    full = (torch.randn(N, ld, device=dev) * 0.5).to(torch.float8_e4m3fn); W = full[:, :K]
    S8 = torch.randint(118, 125, (N // 128, K // 128), dtype=torch.uint8, device=dev).view(torch.float8_e8m0fnu)
    Sf = S8.float()
    ref_w = (W.float().view(N // 128, 128, K // 128, 128) * Sf.view(N // 128, 1, K // 128, 1)).view(N, K)
    dq = w8.fp8_dequant(W, S8).float()
    exact = torch.equal(dq, ref_w)
    for M in (16, 64, 512):
        x = torch.randn(M, K, device=dev).bfloat16()
        assert w8.fp8_prefill_ok(x, W, S8, [128, 128])
        ref = x.float() @ ref_w.t()
        got = w8.fp8_prefill_mm(x, W, S8).float()
        rel = ((got - ref).norm() / ref.norm()).item(); worst = max(worst, rel)
        q, qs = per_token_group_quant_fp8(x, 128)
        st = lambda: w8a8_triton_block_scaled_mm(q, W, qs, Sf, [128, 128], torch.bfloat16)
        srel = ((st().float() - ref).norm() / ref.norm()).item()
        ta, tb, td = tm(st), tm(lambda: w8.fp8_prefill_mm(x, W, S8)), tm(lambda: w8.fp8_dequant(W, S8))
        print(f"N={N:5d} K={K:5d} M={M:4d}: dequant exact={exact} | rel {rel:.1e} (stock {srel:.1e}) | "
              f"stock {ta:5.2f} ms  vsh {tb:5.2f} ms (dequant {td:4.2f}) -> {ta/tb:4.1f}x", flush=True)
print(f"worst rel {worst:.1e}")
