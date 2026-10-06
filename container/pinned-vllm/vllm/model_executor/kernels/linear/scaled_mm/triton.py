# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm import (  # noqa: E501
    triton_scaled_mm,
)
from vllm.model_executor.layers.quantization.utils import replace_parameter
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    convert_to_channelwise,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

from .BlockScaledMMLinearKernel import (
    Fp8BlockScaledMMLinearKernel,
)
from .cutlass import CutlassInt8ScaledMMLinearKernel
from .ScaledMMLinearKernel import (
    Int8ScaledMMLinearLayerConfig,
)


class TritonInt8ScaledMMLinearKernel(CutlassInt8ScaledMMLinearKernel):
    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if current_platform.is_cuda_alike() or current_platform.is_xpu():
            return True, None
        return False, "requires ROCm, CUDA or XPU."

    @classmethod
    def can_implement(cls, c: Int8ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        w_q, _, i_s, _, _ = self._get_layer_params(layer)
        w_q_name, w_s_name, i_s_name, i_zp_name, azp_adj_name = self.layer_param_names

        replace_parameter(
            layer,
            w_q_name,
            torch.nn.Parameter(w_q.t().data, requires_grad=False),
        )

        # WEIGHT SCALE
        # Triton kernel supports only per-tensor and per-channel.
        # If we have a fused module (QKV, MLP) with per tensor scales (thus N
        # scales being passed to the kernel), convert to the per-channel case.
        is_fused_module = len(layer.logical_widths) > 1
        weight_scale = getattr(layer, w_s_name)
        if is_fused_module and not self.config.is_channelwise:
            weight_scale = convert_to_channelwise(weight_scale, layer.logical_widths)
        replace_parameter(
            layer,
            w_s_name,
            torch.nn.Parameter(weight_scale.data, requires_grad=False),
        )

        # INPUT SCALE
        if self.config.is_static_input_scheme:
            assert i_s is not None

            if self.config.input_symmetric:
                replace_parameter(
                    layer,
                    i_s_name,
                    torch.nn.Parameter(i_s.max(), requires_grad=False),
                )
                setattr(layer, i_zp_name, None)
            else:
                input_zero_point = getattr(layer, i_zp_name)

                # Reconstruct the ranges to find a single scale and azp
                int8_traits = torch.iinfo(torch.int8)
                azps = input_zero_point.to(dtype=torch.int32)
                range_max = (i_s * (int8_traits.max - azps)).max()
                range_min = (i_s * (int8_traits.min - azps)).min()

                scale = (range_max - range_min) / (int8_traits.max - int8_traits.min)
                replace_parameter(
                    layer,
                    i_s_name,
                    torch.nn.Parameter(scale, requires_grad=False),
                )

                # AZP loaded as int8 but used as int32
                azp = (int8_traits.min - range_min / scale).to(dtype=torch.int32)
                replace_parameter(
                    layer,
                    i_zp_name,
                    torch.nn.Parameter(azp, requires_grad=False),
                )
        else:
            setattr(layer, i_s_name, None)
            setattr(layer, i_zp_name, None)

        # azp_adj is the AZP adjustment term, used to account for weights.
        # It does not depend on scales or azp, so it is the same for
        # static and dynamic quantization.
        # See csrc/quantization/w8a8/cutlass/Epilogues.md for the math.
        if not self.config.input_symmetric:
            weight = getattr(layer, w_q_name)
            # weight is already transposed to [K, N], sum over K (dim=0)
            azp_adj = weight.sum(dim=0, keepdim=True, dtype=torch.int32)
            if self.config.is_static_input_scheme:
                # Fold azp into azp_adj for the per-tensor case
                azp_adj = getattr(layer, i_zp_name) * azp_adj
            setattr(
                layer,
                azp_adj_name,
                torch.nn.Parameter(azp_adj, requires_grad=False),
            )
        else:
            setattr(layer, azp_adj_name, None)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        w_q, w_s, i_s, i_zp, azp_adj = self._get_layer_params(layer)

        symmetric = azp_adj is None
        x_q, x_s, x_zp = ops.scaled_int8_quant(
            x.contiguous(), i_s, i_zp, symmetric=symmetric
        )

        out = triton_scaled_mm(
            x_q, w_q, scale_a=x_s, scale_b=w_s, out_dtype=x.dtype, bias=bias
        )

        if azp_adj is not None:
            # Asymmetric quantization: subtract the zero-point correction.
            # D = scale_a * scale_b * (A_q @ B_q - azp * azp_adj) + bias
            # triton_scaled_mm already computed scale_a * scale_b * (A_q @ B_q) + bias
            # so we subtract scale_a * scale_b * azp * azp_adj
            #
            # x_s: [M, 1] or scalar, w_s: [N, 1] or scalar, azp_adj: [1, N]
            # Reshape w_s from [N, 1] to [1, N] for proper broadcasting.
            w_s_row = w_s.view(1, -1) if w_s.dim() > 0 else w_s
            static = i_zp is not None
            if not static and x_zp is not None:
                # Dynamic per-token: azp is per-token, azp_adj is per-channel
                # x_zp: [M, 1], azp_adj: [1, N]
                out -= x_s * w_s_row * (x_zp * azp_adj).to(x.dtype)
            else:
                # Static per-tensor: azp already folded into azp_adj
                out -= (x_s * w_s_row * azp_adj).to(x.dtype)

        return out


class TritonFp8BlockScaledMMKernel(Fp8BlockScaledMMLinearKernel):
    @classmethod
    def is_supported(cls, compute_capability=None):
        if not (current_platform.is_cuda_alike() or current_platform.is_xpu()):
            return False, "only CUDA-alike and XPU devices are supported."
        return True, None

    # [vsh-fp8-gemv] gfx1151 decode: HIP block-scaled fp8 GEMV (bf16 activations)
    _vsh_fp8 = None
    _vsh_fp8_logged = [False, False, False, False, False]

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
            elif input_2d.shape[0] > 8:
                # [vsh-fp8-prefill] prefill rows: exact bf16 weight + hipBLASLt (5-8x the Triton block GEMM)
                from vllm.model_executor.layers import vsh_w8a16 as _w8

                params = self._get_layer_params(layer)
                if _w8.fp8_prefill_ok(input_2d, params.weight, params.block_scale,
                                      list(self.weight_group_shape)):
                    out = _w8.fp8_prefill_mm(input_2d, params.weight, params.block_scale)
                    if not cls._vsh_fp8_logged[3]:
                        cls._vsh_fp8_logged[3] = True
                        logger.info("[vsh-fp8-prefill] active (M=%d N=%d K=%d)", input_2d.shape[0],
                                    params.weight.shape[0], params.weight.shape[1])
                    if bias is not None:
                        out = out + bias
                    return out.to(dtype=self.config.out_dtype).view(
                        *x.shape[:-1], params.weight.shape[0])
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
        if (cls._vsh_fp8_on() and A.dim() == 2 and A.shape[0] > 8
                and A.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz)
                and As is not None and As.dim() == 2):
            # [vsh-fp8-prefill] pre-quantized prefill rows: back to bf16, then the bf16 weight + hipBLASLt
            from vllm.model_executor.layers import vsh_w8a16 as _w8

            a_bf = _w8.dequant_act_fp8(A, As)
            if a_bf is not None and _w8.fp8_prefill_ok(a_bf, B, Bs, list(self.weight_group_shape)):
                if not cls._vsh_fp8_logged[4]:
                    cls._vsh_fp8_logged[4] = True
                    logger.info("[vsh-fp8-prefill] active on pre-quantized input (M=%d N=%d K=%d)",
                                A.shape[0], B.shape[0], B.shape[1])
                return _w8.fp8_prefill_mm(a_bf, B, Bs).to(self.config.out_dtype)
        return torch.ops.vllm.w8a8_triton_block_scaled_mm_func(
            A,
            B,
            As,
            Bs,
            list(self.weight_group_shape),
            self.config.out_dtype,
        )


# TODO we should be able to change the type of block_size to GroupShape
# after we resolve GroupShape compilation issue
# https://github.com/vllm-project/vllm/issues/25270
def _w8a8_triton_block_scaled_mm_func(
    qx: torch.Tensor,
    weight: torch.Tensor,
    x_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    block_size: list[int],
    output_dtype: torch.dtype,
) -> torch.Tensor:
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        w8a8_triton_block_scaled_mm,
    )

    return w8a8_triton_block_scaled_mm(
        qx, weight, x_scale, weight_scale, block_size, output_dtype
    )


def _w8a8_triton_block_scaled_mm_fake(
    qx: torch.Tensor,
    weight: torch.Tensor,
    x_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    block_size: list[int],
    output_dtype: torch.dtype,
) -> torch.Tensor:
    return torch.empty(
        (qx.size(0), weight.size(0)), dtype=output_dtype, device=qx.device
    )


direct_register_custom_op(
    "w8a8_triton_block_scaled_mm_func",
    _w8a8_triton_block_scaled_mm_func,
    fake_impl=_w8a8_triton_block_scaled_mm_fake,
)
