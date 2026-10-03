#!/usr/bin/env python3
"""vsh-gate-lmhead: fp32 router-gate GEMV + int8 lm_head on gfx1151 (PATCHES 30).

GateLinear on ROCm (fp32 weights, force_fp32_compute) falls to x.float() + an fp32
hipBLASLt GEMM: ~150 us x 42 layers per step for a 288 x 4096 fp32 matrix. Tier
"vsh" routes <= 8 rows to libvsh_w8a16.so's w32 GEMV (fp32 weights, fp32
accumulate: rel err ~2e-6). Gate VSH_ROUTER_GEMV=0.
ParallelLMHead: an int8 group-128 copy next to the BF16 weight (kept: the DFlash2
drafter shares the module), decode rows use the W8A16 GEMV. Gate VSH_W8A16_LMHEAD=0.
Usage: python3 vsh-gate-lmhead.py
"""
import ast, sys
from pathlib import Path
SP = Path("/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers")
MARK = "# [vsh-gate-lmhead]"

g = SP / "fused_moe/router/gate_linear.py"
s = g.read_text()
if MARK not in s:
    a = "        # Tier 5: F.linear (ReplicatedLinear)\n"
    assert s.count(a) == 1
    s = s.replace(a, f'''        {MARK} ROCm fp32-weight router GEMV for decode-sized batches
        if (
            current_platform.is_rocm()
            and self.is_unquantized
            and self.bias is None
            and self.weight.dtype == torch.float32
        ):
            from vllm.model_executor.layers import vsh_w8a16 as _w8

            if _w8.w32_ok(self.weight, x):
                return self._return(_w8.w32_gemv(x, self.weight))
''' + a)
    ast.parse(s); g.write_text(s); print("gate_linear: applied")
else:
    print("gate_linear: already")

v = SP / "vocab_parallel_embedding.py"
s = v.read_text()
if MARK not in s:
    a1 = "            dispatch_cpu_unquantized_gemm(layer, remove_weight=False)\n\n    def apply(\n"
    a2 = "        return dispatch_unquantized_gemm()(layer, x, layer.weight, bias)\n"
    assert s.count(a1) == 1 and s.count(a2) == 1
    s = s.replace(a1, f'''            dispatch_cpu_unquantized_gemm(layer, remove_weight=False)
        {MARK} int8 copy of the lm_head for decode-sized logits
        import os as _vg_os

        if (
            current_platform.is_rocm()
            and type(layer).__name__ == "ParallelLMHead"
            and _vg_os.environ.get("VSH_W8A16_LMHEAD", "1") not in ("", "0", "off")
            and getattr(layer, "weight", None) is not None
            and layer.weight.dtype == torch.bfloat16
            and layer.weight.ndim == 2
            and layer.weight.shape[1] in (1024, 1536, 2048, 3072, 4096, 6144, 8192)
        ):
            from vllm.model_executor.layers import vsh_w8a16 as _w8

            _w8.quantize_(layer, keep_bf16=True)

    def apply(
''')
    s = s.replace(a2, f'''        if getattr(layer, "vsh_w8_q", None) is not None:  {MARK}
            from vllm.model_executor.layers import vsh_w8a16 as _w8

            return _w8.apply(layer, x, bias)
''' + a2)
    ast.parse(s); v.write_text(s); print("vocab_parallel_embedding: applied")
else:
    print("vocab_parallel_embedding: already")
