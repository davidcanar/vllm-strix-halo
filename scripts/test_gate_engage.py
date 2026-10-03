# GateLinear.forward with the flags GLM-5.3's gate has on this pin (bf16 weight,
# fp32 out -> tier 4); TP init is not needed for forward.
import torch
from vllm.model_executor.layers import vsh_w8a16 as w8
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
g = GateLinear.__new__(GateLinear)
torch.nn.Module.__init__(g)
g.weight = torch.nn.Parameter((torch.randn(288, 4096, device="cuda") * 0.02).bfloat16(), requires_grad=False)
g.allow_ll_bf16_gemm = g.allow_fp32_router_gemm = g.allow_bf16x3_router_gemm = False
g.allow_cublas_router_gemm = True
g.return_bias = True
hits = []
orig = w8.w16_gemv
w8.w16_gemv = lambda x, w: (hits.append(1), orig(x, w))[1]
x = torch.randn(4, 4096, device="cuda").bfloat16()
y, _ = g(x)
ref = torch.mm(x, g.weight.T, out_dtype=torch.float32)
print("w16 engaged:", bool(hits), y.dtype, tuple(y.shape), "maxdiff vs tier4", (y - ref).abs().max().item())
