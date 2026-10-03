#!/usr/bin/env python3
"""vllm-strix-halo: direct decode MoE path (PATCHES.md 31).

At decode (<= 64 token-expert pairs) CompressedTensorsWNA16MoEMethod.apply goes
through the modular kernel: prepare/finalize, workspace sizing, config lookup,
moe_align_block_size, two wna16 launches, clamp-SwiGLU and moe_sum -- ~0.4 ms of
host Python per layer. This hook calls vsh_moe_int4.direct_moe instead: three
HIP launches (w13 with in-kernel expert dedup, w2 with the clamp-SwiGLU fused
into its activation staging and the router weight applied, top-k sum).
Gated per layer by vsh_moe_int4.direct_static_ok (cached); VSH_MOE_DIRECT=0 disables.

Usage: python3 vsh-moe-direct.py
"""
import ast
from pathlib import Path

SP = Path("/opt/venv/lib/python3.12/site-packages/vllm")
P = SP / "model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_wna16.py"
MARK = "# [vsh-moe-direct]"
ANCHOR = '''        assert not self.is_monolithic
        assert self.moe_kernel is not None
        return self.moe_kernel.apply(
            x,
            layer.w13_weight,
            layer.w2_weight,
            topk_weights=topk_weights,
'''
INSERT = f'''        {MARK} decode-sized batches -> direct HIP int4 MoE (no modular kernel)
        _vd = getattr(layer, "_vsh_direct", None)
        if _vd is None:
            from vllm.model_executor.layers.fused_moe import vsh_moe_int4 as _vm
            _vd = layer._vsh_direct = _vm.direct_static_ok(self, layer)
        if _vd and x.dim() == 2 and x.dtype == torch.bfloat16 and 0 < x.size(0) * topk_ids.size(1) <= 64:
            from vllm.model_executor.layers.fused_moe import vsh_moe_int4 as _vm
            _qc = self.moe_kernel.fused_experts.quant_config
            return _vm.direct_moe(
                x, layer.w13_weight, layer.w2_weight, _qc.w1_scale, _qc.w2_scale,
                topk_weights, topk_ids,
                self.moe_kernel.fused_experts.activation_config.clamp_limit,
            )
'''
s = P.read_text()
if MARK not in s:
    assert s.count(ANCHOR) == 1, "anchor"
    head, _, tail = ANCHOR.partition("        return self.moe_kernel.apply(")
    s = s.replace(ANCHOR, head + INSERT + "        return self.moe_kernel.apply(" + tail)
    ast.parse(s)
    P.write_text(s)
    print("vsh-moe-direct: applied")
else:
    print("vsh-moe-direct: already applied")
