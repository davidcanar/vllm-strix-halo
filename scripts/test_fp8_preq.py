# pre-quantized route: dequant_act_fp8 + fp8_gemv vs stock W8A8 on the same fp8 activations
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import vsh_w8a16 as w8
from vllm.model_executor.layers.quantization.utils.fp8_utils import w8a8_triton_block_scaled_mm, per_token_group_quant_fp8
torch.manual_seed(1)
for (N, K) in ((16384, 1024), (4096, 1024)):
    w = (torch.randn(N, K, device="cuda") * 0.6).to(torch.float8_e4m3fn)
    s = torch.rand(N // 128, K // 128, device="cuda") * 0.02 + 0.002
    for M in (1, 6):
        x = torch.randn(M, K, device="cuda").bfloat16()
        xq, xs = per_token_group_quant_fp8(x, 128)
        stock = w8a8_triton_block_scaled_mm(xq, w, xs, s, [128, 128], output_dtype=torch.bfloat16).float()
        got = w8.fp8_gemv(w8.dequant_act_fp8(xq, xs), w, s).float()
        print(f"N={N} K={K} M={M}: pre-quantized GEMV vs stock rel {((got-stock).norm()/stock.norm()).item():.2e}")
