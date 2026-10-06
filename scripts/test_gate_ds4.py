# Does a DS4-shaped GateLinear (256 x 4096, bf16 weight, fp32 out) take the vsh w16 GEMV at M=6?
import os, tempfile, torch
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import init_distributed_environment, initialize_model_parallel
from vllm.model_executor.layers import vsh_w8a16 as w8
with set_current_vllm_config(VllmConfig()):
    init_distributed_environment(world_size=1, rank=0, local_rank=0,
                                 distributed_init_method=f"file://{tempfile.mktemp()}", backend="gloo")
    initialize_model_parallel(1, 1)
    from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
    torch.set_default_dtype(torch.bfloat16)
    g = GateLinear(input_size=4096, output_size=256, bias=False, out_dtype=torch.float32, prefix="model.layers.3.ffn.gate").cuda()
    torch.set_default_dtype(torch.float32)
    print("weight", g.weight.dtype, tuple(g.weight.shape), "contig", g.weight.is_contiguous(),
          "| allow_cublas_router_gemm", g.allow_cublas_router_gemm, "| out_dtype", g.out_dtype,
          "| ll_bf16", g.allow_ll_bf16_gemm, "fp32_router", g.allow_fp32_router_gemm)
    hits = []
    orig = w8.w16_gemv
    w8.w16_gemv = lambda x, w: (hits.append(tuple(x.shape)), orig(x, w))[1]
    for M in (1, 6):
        x = torch.randn(M, 4096, device="cuda").bfloat16()
        print(f"M={M}: w16_ok={w8.w16_ok(g.weight, x)}")
        y = g(x)
    print("w16 calls:", hits)
