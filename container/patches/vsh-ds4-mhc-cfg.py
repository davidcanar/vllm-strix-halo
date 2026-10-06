#!/usr/bin/env python3
"""vllm-strix-halo: aiter mHC tile configs + fused/unfused crossover for gfx1151 (PATCHES.md 42).

DeepSeek-V4 mixes 4 residual streams around every sublayer (manifold hyper-
connections). aiter has tuned mHC tile tables for gfx950/gfx942/gfx1250 only;
on gfx1151 its fallback config runs the fused post+pre GEMM at 13-16 GB/s
(2.6 ms per call at a 512-token prefill chunk, 38 us at decode), and its
fused->unfused crossover sits at 1024 tokens. Measured on gfx1151
(scripts/mhc_aiter_sweep.py), same aiter kernels, different tiles/routing:
  fused config  M<=32: (64,16,16,32)  37.9 -> 20.5 us at M=6
                M<=384: (32,16,16,32)  312 -> 140 us at M=128
  M >= 384: aiter's unfused mhc_post + mhc_pre (614 vs 2559 us at 512)
  mhc_pre split-k: 64 up to 512 rows, 16 up to 1024, 1 above (451 -> 364 us
                   at 512, 3132 -> 1845 us at 2048)
Env VSH_MHC_CFG=0 keeps aiter's defaults.
"""
import ast
import sys
from pathlib import Path

SP = Path("/opt/venv/lib/python3.12/site-packages/vllm")
F = SP / "_aiter_ops.py"
MARK = "[vsh-ds4-mhc-cfg]"
s = F.read_text()
if MARK in s:
    print("vsh-ds4-mhc-cfg: already applied")
    sys.exit(0)
anchor = "rocm_aiter_ops.register_ops_once()\n"
assert s.count(anchor) == 1, "register_ops_once anchor"
install = '''

def _vsh_mhc_cfg_install() -> None:  # [vsh-ds4-mhc-cfg] gfx1151 mHC tiles + fused/unfused crossover
    import os

    if os.environ.get("VSH_MHC_CFG", "1") in ("", "0", "off"):
        return
    try:
        from vllm.platforms.rocm import on_gfx1151

        if not on_gfx1151():
            return
        import aiter.ops.mhc as _m
        from aiter.jit.utils.chip_info import get_cu_num
    except Exception:
        return
    if getattr(_m, "_vsh_mhc_cfg", False):
        return

    def _fused_cfg(m, hidden_size, num_cu):
        if m <= 32:
            return 64, 16, 16, 32
        return 32, 16, 16, 32

    _m._MHC_FUSED_POST_PRE_CONFIG[("gfx1151", get_cu_num())] = _fused_cfg
    _m.get_mhc_fused_post_pre_config.cache_clear()

    def _pre_splitk(m, hc_hidden_size):
        sk = 64 if m <= 512 else 16 if m <= 1024 else 1
        while sk > 1 and (hc_hidden_size % (sk * 64) or hc_hidden_size // sk < 128):
            sk //= 2
        return sk, 64

    _m.get_mhc_pre_splitk = _pre_splitk
    _orig_fused = _m.mhc_fused_post_pre

    def _fused_post_pre(layer_input, residual_in, post_layer_mix, comb_res_mix, fn, hc_scale, hc_base,
                        rms_eps=1e-6, hc_pre_eps=1e-6, hc_sinkhorn_eps=1e-6, hc_post_mult_value=1.0,
                        sinkhorn_repeat=20, norm_weight=None, norm_eps=1e-6, force_fused=False,
                        is_fn_pack_bf16=0):
        if not force_fused and layer_input.size(0) >= 384:
            # aiter's own unfused branch (it switches at 1024 rows on unknown archs)
            next_residual = torch.empty_like(residual_in)
            _m.mhc_post(next_residual, layer_input, residual_in, post_layer_mix, comb_res_mix)
            post_mix, comb_mix, layer_input_out = _m.mhc_pre(
                next_residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps, hc_sinkhorn_eps,
                hc_post_mult_value, sinkhorn_repeat, norm_weight, norm_eps,
                is_fn_pack_bf16=is_fn_pack_bf16)
            return post_mix, comb_mix, layer_input_out, next_residual
        return _orig_fused(layer_input, residual_in, post_layer_mix, comb_res_mix, fn, hc_scale, hc_base,
                           rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat,
                           norm_weight, norm_eps, force_fused, is_fn_pack_bf16)

    _m.mhc_fused_post_pre = _fused_post_pre
    _m._vsh_mhc_cfg = True


_vsh_mhc_cfg_install()
'''
s = s.replace(anchor, anchor + install)
ast.parse(s)
F.write_text(s)
print("vsh-ds4-mhc-cfg: gfx1151 mHC configs installed")
