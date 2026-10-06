#!/usr/bin/env python3
"""vllm-strix-halo: direct HIP MXFP4 MoE for decode-sized batches (PATCHES.md 38).

DeepSeek-V4's routed experts are MXFP4; on gfx1151 they run
UnfusedOAITritonExperts (triton_kernels 3.8 matmul, upcast in-kernel, plus
routing-data build, activation and index_add combine) -- ~99 ms/step of the DS4
decode. For <= 64 token-expert pairs on gfx1151 this hook runs
vsh_moe_int4.mxfp4_direct_moe instead: w13 (expert dedup per slot), w2 with the
SiLU(+clamp) fused into its staging and the router weight applied, fp32 top-k
sum straight into `output`. The layout is checked once per weight pair
(vsh_moe_int4.mxfp4_plan); anything else falls through to the stock path.
VSH_MOE_MXFP4_DIRECT=0 disables.

Usage: python3 vsh-mxfp4-direct.py [path/to/gpt_oss_triton_kernels_moe.py]
"""
import ast
import sys
from pathlib import Path

P = Path(sys.argv[1] if len(sys.argv) > 1 else
         "/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py")
MARK = "# [vsh-mxfp4-direct]"
ANCHOR = '''        global_topk_ids = topk_ids
        if expert_map is not None:
'''
INSERT = f'''        {MARK} decode-sized MXFP4 MoE on gfx1151 -> HIP direct path
        if (expert_map is None and not apply_router_weight_on_input
                and getattr(self, "_lora_context", None) is None
                and activation == MoEActivation.SILU
                and hidden_states.dim() == 2 and hidden_states.dtype == torch.bfloat16
                and 0 < hidden_states.shape[0] * topk_ids.shape[1] <= 64):
            _vcache = UnfusedOAITritonExperts.__dict__.get("_vsh_mx_plans")
            if _vcache is None:
                _vcache = {{}}
                UnfusedOAITritonExperts._vsh_mx_plans = _vcache
            _vkey = (id(w1), id(w2))
            _vplan = _vcache.get(_vkey, 0)
            if _vplan == 0:
                _vplan = None
                from vllm.platforms import current_platform as _vcp

                if _vcp.is_rocm():
                    from vllm.platforms.rocm import on_gfx1151 as _vg

                    if _vg():
                        from vllm.model_executor.layers.fused_moe import vsh_moe_int4 as _vm

                        _vplan = _vm.mxfp4_plan(w1, w2, quant_config)
                _vcache[_vkey] = _vplan
                if len(_vcache) == 1:
                    logger.info("[vsh-mxfp4-direct] %s", "active" if _vplan is not None
                                else "not eligible (stock path)")
            if _vplan is not None:
                from vllm.model_executor.layers.fused_moe import vsh_moe_int4 as _vm

                _vm.mxfp4_direct_moe(output, hidden_states, _vplan, topk_weights, topk_ids)
                return

'''
s = P.read_text()
if MARK in s:
    print("vsh-mxfp4-direct: already applied")
else:
    assert s.count(ANCHOR) == 1, f"anchor x{s.count(ANCHOR)}"
    s = s.replace(ANCHOR, INSERT + ANCHOR)
    ast.parse(s)
    P.write_text(s)
    print("vsh-mxfp4-direct: applied")
