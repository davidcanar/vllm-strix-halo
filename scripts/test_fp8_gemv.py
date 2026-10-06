# vsh_fp8_gemv vs exact reference (bf16 x @ dequant(W)^T) and vs stock W8A8 Triton path.
import os, sys, time, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import vsh_w8a16 as w8
from vllm.model_executor.layers.quantization.utils.fp8_utils import w8a8_triton_block_scaled_mm, per_token_group_quant_fp8
torch.manual_seed(0); dev = "cuda"

def make(N, K, dt, pad=256):
    wf = torch.randn(N, K, device=dev) * 0.6
    wf[:, ::97] *= 1e-3                                   # sprinkle subnormal-range values
    full = torch.empty(N, K + pad, device=dev, dtype=dt)  # padded rows like vLLM's
    w = full[:, :K]
    w.copy_(wf.to(dt))
    s = torch.rand((N + 127) // 128, (K + 127) // 128, device=dev) * 0.02 + 0.002
    return w, s

def deq(w, s):
    N, K = w.shape
    se = s.repeat_interleave(128, 0)[:N].repeat_interleave(128, 1)[:, :K]
    return w.float() * se

worst = 0.0
for dt in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
    for (N, K) in ((1536, 4096), (16384, 1024), (4096, 4096), (4096, 1024), (2048, 4096), (512, 4096),
                   (256, 512), (256, 1536), (256, 2048), (256, 3072), (256, 6144), (256, 7168), (256, 8192), (300, 1024)):
        w, s = make(N, K, dt)
        ref_w = deq(w, s)
        for M in (1, 4, 6, 8):
            x = torch.randn(M, K, device=dev).bfloat16()
            ref = x.float() @ ref_w.T
            got = w8.fp8_gemv(x, w, s).float()
            rel = ((got - ref).norm() / ref.norm()).item()
            worst = max(worst, rel)
            if M == 6 and dt == torch.float8_e4m3fn and N >= 512:
                xq, xs = per_token_group_quant_fp8(x, 128)
                stock = w8a8_triton_block_scaled_mm(xq, w.contiguous() if dt == torch.float8_e4m3fn else w, xs, s, [128, 128], output_dtype=torch.bfloat16).float()
                rel_stock = ((stock - ref).norm() / ref.norm()).item()
                print(f"{str(dt)[6:]:14s} N={N:5d} K={K}: gemv rel {rel:.2e} | stock W8A8 rel {rel_stock:.2e}")
print(f"worst rel err over all cases: {worst:.2e}")

for (N, K) in ((16384, 1024), (4096, 4096), (2048, 4096), (1536, 4096), (4096, 1024)):
    w, s = make(N, K, torch.float8_e4m3fn)
    wc = w.contiguous()
    for M in (1, 6):
        x = torch.randn(M, K, device=dev).bfloat16()
        xq, xs = per_token_group_quant_fp8(x, 128)
        def tm(fn, it=50):
            for _ in range(5): fn()
            torch.cuda.synchronize(); t = time.perf_counter()
            for _ in range(it): fn()
            torch.cuda.synchronize(); return (time.perf_counter() - t) / it * 1e3
        t_g = tm(lambda: w8.fp8_gemv(x, w, s))
        t_s = tm(lambda: w8a8_triton_block_scaled_mm(xq, wc, xs, s, [128, 128], output_dtype=torch.bfloat16), 10)
        print(f"timing N={N:5d} K={K} M={M}: gemv {t_g*1e3:7.1f} us ({N*K/(t_g*1e-3)/1e9:5.0f} GB/s) | stock {t_s:.3f} ms -> {t_s/t_g:.0f}x")
