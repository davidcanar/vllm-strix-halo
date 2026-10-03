#!/usr/bin/env python3
"""vllm-strix-halo: bf16-weight router gate GEMV with fp32 output (PATCHES.md 32).

On this pin GLM-5.3's MoE gate is a BF16 weight with an fp32 output, so
GateLinear takes tier 4 (torch.mm(out_dtype=fp32) -> hipBLASLt): ~150 us per
layer for a 288 x 4096 gate at decode, ~6 ms/step. For <= 8 rows route it to
libvsh_w8a16.so's vsh_w16_gemv (fp32 accumulate, same as the hipBLASLt epilogue).
VSH_ROUTER_GEMV=0 disables (shared with the fp32-weight hook of section 30).

Usage: python3 vsh-gate-bf16.py
"""
import ast
from pathlib import Path

P = Path("/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/fused_moe/router/gate_linear.py")
MARK = "# [vsh-gate-bf16]"
ANCHOR = '''        # Tier 4: cuBLAS bf16→fp32
        if self.allow_cublas_router_gemm and x.dtype == torch.bfloat16:
'''
INSERT = f'''        {MARK} ROCm bf16-weight/fp32-out router GEMV for decode-sized batches
        if self.allow_cublas_router_gemm and current_platform.is_rocm():
            from vllm.model_executor.layers import vsh_w8a16 as _w8

            if _w8.w16_ok(self.weight, x):
                return self._return(_w8.w16_gemv(x, self.weight))
'''
s = P.read_text()
if MARK not in s:
    assert s.count(ANCHOR) == 1, "anchor"
    s = s.replace(ANCHOR, INSERT + ANCHOR)
    ast.parse(s)
    P.write_text(s)
    print("vsh-gate-bf16: applied")
else:
    print("vsh-gate-bf16: already applied")
