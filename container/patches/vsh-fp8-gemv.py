#!/usr/bin/env python3
"""vllm-strix-halo: HIP FP8 block-scaled GEMV for decode-sized linears (PATCHES.md 36).

DeepSeek-V4's dense linears are block-FP8 (128x128, ue8m0 scales). On gfx1151
they run TritonFp8BlockScaledMMKernel: fp8 activation quant + the Triton block
GEMM, which has no fp8 dot here and reaches 3-11 GB/s at decode sizes -- ~87 %
of the DS4 decode step (PATCHES 35). This replaces the kernel's apply_weights
and apply_block_scaled_mm on gfx1151 for M <= 8: bf16 activations go straight
to vsh_w8a16.fp8_gemv (fp8 weight streamed with in-register decode); already
quantized fp8 activations (DS4's fused q-norm -> wq_b path calls
apply_block_scaled_mm directly) are dequantized with their 1x128 scales first.
Everything else falls through to the stock path. GLM-5.3 has no FP8 linears.
VSH_FP8_GEMV=0 disables.

Usage: python3 vsh-fp8-gemv.py
"""
import ast
from pathlib import Path

P = Path("/opt/venv/lib/python3.12/site-packages/vllm/model_executor/kernels/linear/scaled_mm/triton.py")
MARK = "# [vsh-fp8-gemv]"
STOCK = '''    def apply_block_scaled_mm(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        As: torch.Tensor,
        Bs: torch.Tensor,
    ) -> torch.Tensor:
        return torch.ops.vllm.w8a8_triton_block_scaled_mm_func(
            A,
            B,
            As,
            Bs,
            list(self.weight_group_shape),
            self.config.out_dtype,
        )
'''
NEW = MARK + ''' gfx1151 decode: HIP block-scaled fp8 GEMV (bf16 activations)
    _vsh_fp8 = None
    _vsh_fp8_logged = [False, False, False]

    @classmethod
    def _vsh_fp8_on(cls) -> bool:
        if cls._vsh_fp8 is None:
            from vllm.platforms import current_platform as _cp

            on = False
            if _cp.is_rocm():
                from vllm.platforms.rocm import on_gfx1151

                on = on_gfx1151()
            cls._vsh_fp8 = on
        return cls._vsh_fp8

    def apply_weights(self, layer, x, bias=None, **kwargs):
        cls = TritonFp8BlockScaledMMKernel
        if cls._vsh_fp8_on() and x.dtype == torch.bfloat16:
            input_2d = x.view(-1, x.shape[-1])
            if 0 < input_2d.shape[0] <= 8:
                from vllm.model_executor.layers import vsh_w8a16 as _w8

                params = self._get_layer_params(layer)
                if _w8.fp8_gemv_ok(input_2d, params.weight, params.block_scale,
                                   list(self.weight_group_shape)):
                    out = _w8.fp8_gemv(input_2d, params.weight, params.block_scale)
                    if not cls._vsh_fp8_logged[0]:
                        cls._vsh_fp8_logged[0] = True
                        logger.info("[vsh-fp8-gemv] active (M=%d N=%d K=%d %s)", input_2d.shape[0],
                                    params.weight.shape[0], params.weight.shape[1],
                                    params.weight.dtype)
                    if bias is not None:
                        out = out + bias
                    return out.to(dtype=self.config.out_dtype).view(
                        *x.shape[:-1], params.weight.shape[0])
                if not cls._vsh_fp8_logged[1]:
                    cls._vsh_fp8_logged[1] = True
                    w, bs = params.weight, params.block_scale
                    logger.info("[vsh-fp8-gemv] fallback for M=%d: weight %s %s stride %s, scale %s %s",
                                input_2d.shape[0], tuple(w.shape), w.dtype, tuple(w.stride()),
                                None if bs is None else tuple(bs.shape),
                                None if bs is None else bs.dtype)
        return super().apply_weights(layer, x, bias, **kwargs)

    def apply_block_scaled_mm(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        As: torch.Tensor,
        Bs: torch.Tensor,
    ) -> torch.Tensor:
        # pre-quantized fp8 activations (DS4 fused q-norm -> wq_b): dequantize the
        # <= 8 rows with their 1x128 scales and use the GEMV as well
        cls = TritonFp8BlockScaledMMKernel
        if (cls._vsh_fp8_on() and A.dim() == 2 and 0 < A.shape[0] <= 8
                and A.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz)
                and As is not None and As.dim() == 2):
            from vllm.model_executor.layers import vsh_w8a16 as _w8

            if _w8.fp8_weight_ok(B, Bs, list(self.weight_group_shape), A.shape[1]):
                a_bf = _w8.dequant_act_fp8(A, As)
                if a_bf is not None:
                    if not cls._vsh_fp8_logged[2]:
                        cls._vsh_fp8_logged[2] = True
                        logger.info("[vsh-fp8-gemv] active on pre-quantized input (M=%d N=%d K=%d)",
                                    A.shape[0], B.shape[0], B.shape[1])
                    return _w8.fp8_gemv(a_bf, B, Bs).to(self.config.out_dtype)
        return torch.ops.vllm.w8a8_triton_block_scaled_mm_func(
            A,
            B,
            As,
            Bs,
            list(self.weight_group_shape),
            self.config.out_dtype,
        )
'''
s = P.read_text()
if MARK not in s:
    assert s.count(STOCK) == 1, "anchor"
    s = s.replace(STOCK, "    " + NEW)
    if "\nlogger = " not in s:
        s = s.replace("import torch\n", "import torch\n\nfrom vllm.logger import init_logger\n\nlogger = init_logger(__name__)\n", 1)
    ast.parse(s)
    P.write_text(s)
    print("vsh-fp8-gemv: applied")
else:
    print("vsh-fp8-gemv: already applied")
