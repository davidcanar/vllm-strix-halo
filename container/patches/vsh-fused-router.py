#!/usr/bin/env python3
"""vsh-fused-router: one HIP kernel for GLM's sigmoid + correction-bias top-8 routing.

GroupedTopKRouter on ROCm ran the torch.compile'd grouped_topk: ~200 us of CPU in
the compiled region + ~50 us of Dynamo guard lookups per MoE layer (profiled),
42 layers per step. vsh_sigmoid_bias_topk (libvsh_w8a16.so) does sigmoid, bias,
the deterministic top-k (biased value desc, index asc), gather of the unbiased
scores, renormalisation and the routed scaling factor in one launch.
Unit test vs grouped_topk: identical ids 400/400 (incl. exact ties), weights
within 9e-8; 67 -> 10 us/call wall at M=4. Eligible: sigmoid scoring, bias present,
a single expert group, fp32 logits, <= 64 tokens. Gate VSH_FUSED_ROUTER=0.
"""
import ast, sys
from pathlib import Path
P = Path("/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/fused_moe/router/grouped_topk_router.py")
MARK = "# [vsh-fused-router]"
A = "        # Select grouped_topk implementation\n"
I = f"""        {MARK} decode-sized sigmoid+bias routing in one HIP kernel (gfx1151)
        if current_platform.is_rocm() and not rocm_aiter_ops.is_fused_moe_enabled():
            from vllm.model_executor.layers import vsh_w8a16 as _vr

            if _vr.route_ok(
                router_logits, self.e_score_correction_bias, self.scoring_func,
                self.num_expert_group, self.topk_group, self.top_k,
            ):
                return _vr.sigmoid_bias_topk(
                    router_logits, self.e_score_correction_bias, self.top_k,
                    self.renormalize, self.routed_scaling_factor,
                )

"""
s = P.read_text()
if MARK in s: print("vsh-fused-router: already"); sys.exit(0)
assert s.count(A) == 1, "anchor"
s = s.replace(A, I + A); ast.parse(s); P.write_text(s); print("vsh-fused-router: applied")
