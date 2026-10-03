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


# ---- direct decode path: topk_ids -> w13 -> clamp-SwiGLU -> w2 -> top-k sum --------
# Skips the modular kernel (prepare/finalize, workspaces, config lookup) and
# moe_align_block_size for small batches; the kernel dedups experts itself.
MAX_PAIRS = 64


def _direct_init(lib):
    lib.vsh_moe_int4_direct.restype = ctypes.c_int
    lib.vsh_moe_int4_direct.argtypes = ([ctypes.c_void_p, ctypes.c_int] + [ctypes.c_void_p] * 6
                                        + [ctypes.c_int] * 4 + [ctypes.c_float] + [ctypes.c_long] * 6)
    lib.vsh_moe_topk_sum.restype = ctypes.c_int
    lib.vsh_moe_topk_sum.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 3 + [ctypes.c_long] * 2
    lib._direct_init = True


def direct_static_ok(method, layer) -> bool:
    """Per-layer config check, cached on the layer by the caller."""
    if os.environ.get("VSH_MOE_DIRECT", "1") in ("", "0", "off"):
        return False
    try:
        from vllm.model_executor.layers.fused_moe.activation import MoEActivation
        mk = method.moe_kernel
        fe = mk.fused_experts
        qc = fe.quant_config
        if type(fe).__name__ != "TritonWNA16Experts" or mk.prepare_finalize.supports_async():
            return False
        if getattr(fe, "_lora_context", None) is not None:
            return False
        if not qc.use_int4_w4a16 or qc.w1_zp is not None or qc.w2_zp is not None:
            return False
        if fe.block_shape is None or fe.block_shape[1] != 128:
            return False
        if layer.expert_map is not None or layer.apply_router_weight_on_input:
            return False
        if layer.activation != MoEActivation.SILU or fe.activation_config.clamp_limit is None:
            return False
        w13, w2 = layer.w13_weight, layer.w2_weight
        if w13.dtype != torch.uint8 or w2.dtype != torch.uint8:
            return False
        K, I = w13.size(2) * 2, w2.size(2) * 2
        if K not in (1024, 2048, 4096) or I not in (1024, 2048) or w13.size(1) != 2 * I:
            return False
        if w13.stride(2) != 1 or w2.stride(2) != 1 or qc.w1_scale.stride(2) != 1 or qc.w2_scale.stride(2) != 1:
            return False
        if qc.w1_scale.dtype != torch.bfloat16 or qc.w2_scale.dtype != torch.bfloat16:
            return False
        return True
    except Exception:
        return False


def direct_moe(x, w13, w2, s13, s2, topk_weights, topk_ids, clamp):
    M, K = x.shape
    top_k = topk_ids.size(1)
    P = M * top_k
    N1, N2 = w13.size(1), w2.size(1)
    ids = topk_ids if topk_ids.dtype == torch.int32 else topk_ids.to(torch.int32)
    ids = ids.contiguous()
    tw = topk_weights.contiguous()
    if tw.dtype != torch.float32:
        tw = tw.float()
    x = x.contiguous()
    c1 = torch.empty((P, N1), dtype=torch.bfloat16, device=x.device)
    c3 = torch.empty((P, N2), dtype=torch.bfloat16, device=x.device)
    out = torch.empty((M, N2), dtype=torch.bfloat16, device=x.device)
    lib = _lib()
    if not getattr(lib, "_direct_init", False):
        _direct_init(lib)
    st = torch.cuda.current_stream().cuda_stream
    rc = lib.vsh_moe_int4_direct(st, 1, x.data_ptr(), w13.data_ptr(), c1.data_ptr(), s13.data_ptr(),
                                 tw.data_ptr(), ids.data_ptr(), P, N1, K, top_k, float(clamp),
                                 x.stride(0), w13.stride(0), w13.stride(1), c1.stride(0),
                                 s13.stride(0), s13.stride(1))
    if rc == 0:
        rc = lib.vsh_moe_int4_direct(st, 2, c1.data_ptr(), w2.data_ptr(), c3.data_ptr(), s2.data_ptr(),
                                     tw.data_ptr(), ids.data_ptr(), P, N2, N1 // 2, top_k, float(clamp),
                                     c1.stride(0), w2.stride(0), w2.stride(1), c3.stride(0),
                                     s2.stride(0), s2.stride(1))
    if rc == 0:
        rc = lib.vsh_moe_topk_sum(st, c3.data_ptr(), out.data_ptr(), M, N2, top_k, c3.stride(0), out.stride(0))
    if rc != 0:
        raise RuntimeError(f"vsh_moe_int4_direct failed: {rc}")
    return out
