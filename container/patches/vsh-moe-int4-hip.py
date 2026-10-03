#!/usr/bin/env python3
"""vsh-moe-int4-hip: route decode-sized int4 MoE GEMMs to a HIP GEMV kernel (gfx1151).

The stock Triton fused_moe_kernel_gptq_awq runs ~70-80 GB/s at decode sizes on
gfx1151 (it pads each expert to BLOCK_SIZE_M rows, loads every packed byte twice,
and slows with the core clock: latency/ALU-bound). vsh_moe_int4.hip is a
wvSplitK-style GEMV: 16 contiguous packed bytes per lane per step, activations
staged in LDS, fp32 accumulate, cross-lane shuffle reduce. Unit test vs stock
(scripts/test_moe_gemv.py): rel err 2.4e-3 (= the stock kernel's bf16 rounding),
2.1-3.0x faster, 160-208 GB/s.

Eligible: symmetric int4 (no zero points), group 128, bf16 activations,
K in {1024, 2048, 4096}, <= 64 (token, expert) pairs, contiguous fp32 routing
weights when they are multiplied in. Anything else falls through to Triton.
Gate: VSH_MOE_INT4_HIP=0 restores the stock kernel (default on).
Usage: python3 vsh-moe-int4-hip.py vsh_moe_int4.py libvsh_moe_int4.so
"""
import ast, shutil, sys
from pathlib import Path
SP = Path("/opt/venv/lib/python3.12/site-packages/vllm")
P = SP / "model_executor/layers/fused_moe/fused_moe.py"
MARK = "# [vsh-moe-int4-hip]"
ANCHOR = '''    assert B_scale is not None and B_scale.ndim == 3
    assert B_zp is None or B_zp.ndim == 3
    assert block_shape is not None and block_shape[0] == 0
'''
INSERT = f'''    {MARK} decode-sized int4 MoE -> HIP GEMV (gfx1151)
    import os as _vm_os
    if (
        use_int4_w4a16
        and _vm_os.environ.get("VSH_MOE_INT4_HIP", "1") not in ("", "0", "off")
        and (
            not mul_routed_weight
            or (topk_weights is not None and topk_weights.dtype == torch.float32
                and topk_weights.is_contiguous())
        )
    ):
        from vllm.model_executor.layers.fused_moe import vsh_moe_int4 as _vm

        if _vm.can_use(A, B, B_zp, block_shape, top_k, config, C):
            _vm.moe_int4_gemv(
                A, B, C, B_scale, topk_weights, sorted_token_ids, expert_ids,
                num_tokens_post_padded, mul_routed_weight, top_k, config, block_shape,
            )
            return
'''
s = P.read_text()
if MARK not in s:
    assert s.count(ANCHOR) == 1, "anchor"
    s = s.replace(ANCHOR, ANCHOR + INSERT); ast.parse(s); P.write_text(s); print("vsh-moe-int4-hip: routing applied")
else:
    print("vsh-moe-int4-hip: already applied")
shutil.copyfile(sys.argv[1], SP / "model_executor/layers/fused_moe/vsh_moe_int4.py")
shutil.copyfile(sys.argv[2], SP / "libvsh_moe_int4.so")
print("vsh-moe-int4-hip: module + library installed")
