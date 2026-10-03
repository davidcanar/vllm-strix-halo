# SPDX-License-Identifier: Apache-2.0
# ctypes wrapper for libvsh_moe_int4.so (vsh_moe_int4.hip): decode int4 MoE GEMV.
import ctypes, os
import torch

_LIB = None
_PATHS = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "libvsh_moe_int4.so"),
          "/opt/venv/lib/python3.12/site-packages/vllm/libvsh_moe_int4.so"]


def _lib():
    global _LIB
    if _LIB is None:
        for p in _PATHS:
            if os.path.exists(p):
                _LIB = ctypes.CDLL(p); break
        else:
            raise RuntimeError("libvsh_moe_int4.so not found")
        _LIB.vsh_moe_int4_gemv.restype = ctypes.c_int
        _LIB.vsh_moe_int4_gemv.argtypes = [ctypes.c_void_p] * 9 + [ctypes.c_int] * 7 + [ctypes.c_long] * 6
    return _LIB


def can_use(A, B, B_zp, block_shape, top_k, config, C=None):
    return (B_zp is None and B.dtype == torch.uint8 and A.dtype == torch.bfloat16
            and block_shape is not None and block_shape[1] == 128
            and A.size(1) in (1024, 2048, 4096) and B.size(2) * 2 == A.size(1)
            and B.stride(2) == 1 and A.stride(1) == 1
            and A.size(0) * top_k <= 64 and config["BLOCK_SIZE_M"] <= 64
            and (C is None or (C.dtype == torch.bfloat16 and C.stride(2) == 1)))


def moe_int4_gemv(A, B, C, B_scale, topk_weights, sorted_token_ids, expert_ids,
                  num_tokens_post_padded, mul_routed_weight, top_k, config, block_shape):
    M = A.size(0); BM = config["BLOCK_SIZE_M"]
    EM = min(sorted_token_ids.size(0), M * top_k * BM)
    nmb = (EM + BM - 1) // BM
    tw = topk_weights if (mul_routed_weight and topk_weights is not None) else A
    if mul_routed_weight:
        assert topk_weights.dtype == torch.float32 and topk_weights.is_contiguous()
    rc = _lib().vsh_moe_int4_gemv(
        torch.cuda.current_stream().cuda_stream, A.data_ptr(), B.data_ptr(), C.data_ptr(),
        B_scale.data_ptr(), tw.data_ptr(), sorted_token_ids.data_ptr(), expert_ids.data_ptr(),
        num_tokens_post_padded.data_ptr(), nmb, B.size(1), A.size(1), M * top_k, BM, top_k,
        int(bool(mul_routed_weight)), A.stride(0), B.stride(0), B.stride(1), C.stride(1),
        B_scale.stride(0), B_scale.stride(1))
    if rc != 0:
        raise RuntimeError(f"vsh_moe_int4_gemv failed: {rc}")
