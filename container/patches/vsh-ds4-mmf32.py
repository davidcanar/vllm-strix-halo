#!/usr/bin/env python3
"""vllm-strix-halo: DS4 compressor scores via the bf16 GEMV + gate-hook diagnostic (PATCHES.md 40).

DeepSeek-V4's attention computes the compressor (and indexer-compressor) KV
scores as torch.mm(hidden, fused_wkv_wgate.weight.T, out_dtype=float32): BF16
weights, fp32 out, hipBLASLt at ~55 GB/s for 6-row decode batches. Route them
through vsh_w8a16.mm_f32 (vsh_w16_gemv for <= 8 rows, torch.mm otherwise).
Also: GateLinear logs once when a decode-sized call misses the w16 GEMV
(the DS4 profile showed the router gate on torch.mm).

Usage: python3 vsh-ds4-mmf32.py
"""
import ast
from pathlib import Path

SP = Path("/opt/venv/lib/python3.12/site-packages/vllm")
ATT = SP / "models/deepseek_v4/attention.py"
GATE = SP / "model_executor/layers/fused_moe/router/gate_linear.py"
MARK = "# [vsh-ds4-mmf32]"

s = ATT.read_text()
if MARK not in s:
    for w in ("compressor.fused_wkv_wgate.weight", "indexer.compressor.fused_wkv_wgate.weight"):
        old = f'''                return torch.mm(
                    hidden_states,
                    {w}.T,
                    out_dtype=torch.float32,
                )
'''
        new = f'''                {MARK} decode rows -> bf16 GEMV (fp32 out)
                from vllm.model_executor.layers import vsh_w8a16 as _w8

                return _w8.mm_f32(hidden_states, {w})
'''
        assert s.count(old) == 1, w
        s = s.replace(old, new)
    ast.parse(s)
    ATT.write_text(s)
    print("vsh-ds4-mmf32: attention compressor scores patched")
else:
    print("vsh-ds4-mmf32: attention already patched")

g = GATE.read_text()
DIAG = "# [vsh-gate-bf16-diag]"
if DIAG not in g:
    anchor = '''        # Tier 4: cuBLAS bf16→fp32
        if self.allow_cublas_router_gemm and x.dtype == torch.bfloat16:
'''
    assert g.count(anchor) == 1, "tier4 anchor"
    diag = f'''        {DIAG} a decode-sized call reaching here missed the w16 GEMV: say why, once
        if (current_platform.is_rocm() and x.dim() == 2 and 0 < x.shape[0] <= 8
                and not GateLinear.__dict__.get("_vsh_w16_why", False)):
            GateLinear._vsh_w16_why = True
            import logging as _vlog

            _vlog.getLogger(__name__).warning(
                "[vsh-gate-bf16] decode gate not on w16: x %s %s stride %s | w %s %s stride %s"
                " | allow_cublas %s out_dtype %s",
                tuple(x.shape), x.dtype, tuple(x.stride()), tuple(self.weight.shape),
                self.weight.dtype, tuple(self.weight.stride()),
                self.allow_cublas_router_gemm, self.out_dtype)
'''
    g = g.replace(anchor, diag + anchor)
    ast.parse(g)
    GATE.write_text(g)
    print("vsh-ds4-mmf32: gate diagnostic added")
else:
    print("vsh-ds4-mmf32: gate diagnostic present")
