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
                                        + [ctypes.c_int] * 4 + [ctypes.c_float] + [ctypes.c_long] * 6
                                        + [ctypes.c_int])
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
                                 s13.stride(0), s13.stride(1), w13.size(0))
    if rc == 0:
        rc = lib.vsh_moe_int4_direct(st, 2, c1.data_ptr(), w2.data_ptr(), c3.data_ptr(), s2.data_ptr(),
                                     tw.data_ptr(), ids.data_ptr(), P, N2, N1 // 2, top_k, float(clamp),
                                     c1.stride(0), w2.stride(0), w2.stride(1), c3.stride(0),
                                     s2.stride(0), s2.stride(1), w2.size(0))
    if rc == 0:
        rc = lib.vsh_moe_topk_sum(st, c3.data_ptr(), out.data_ptr(), M, N2, top_k, c3.stride(0), out.stride(0))
    if rc != 0:
        raise RuntimeError(f"vsh_moe_int4_direct failed: {rc}")
    return out


# ---- MXFP4 direct decode path (DeepSeek-V4 routed experts; PATCHES.md 38) ------
# UnfusedOAITritonExperts (triton_kernels 3.8 matmul, upcast-in-kernel) is ~99
# ms/step of the DS4 decode on gfx1151. vsh_moe_mxfp4_direct reads the weights
# exactly as vLLM 0.31's _swizzle_mxfp4 leaves them on gfx1151: values
# uint8 [E][N][K/2] behind a transposed [E, K/2, N] view, e8m0 scales in a
# separate [E, K/32, N] tensor (b_mx_scale).
def _mx_raw(t):
    """The torch tensor behind a triton_kernels wrapper (or a plain tensor)."""
    for attr in ("storage", "handle"):
        o = getattr(t, attr, None)
        if o is not None:
            d = getattr(o, "data", o)
            if isinstance(d, torch.Tensor):
                return d
    return t if isinstance(t, torch.Tensor) else None


def _mx_scale(prec):
    if prec is None:
        return None
    for name in ("b_mx_scale", "weight_scale"):
        v = getattr(prec, name, None)
        if v is not None:
            return _mx_raw(v)
    return None


def mxfp4_plan(w1, w2, quant_config):
    """Raw pointers/strides for the HIP path, or None when the layout is not the
    one the kernel reads (checked once per weight pair by the caller)."""
    if os.environ.get("VSH_MOE_MXFP4_DIRECT", "1") in ("", "0", "off"):
        return None
    try:
        if getattr(quant_config, "w1_bias", None) is not None or getattr(quant_config, "w2_bias", None) is not None:
            return None
        v1, v2 = _mx_raw(w1), _mx_raw(w2)
        s1, s2 = _mx_scale(quant_config.w1_precision), _mx_scale(quant_config.w2_precision)
        out = {}
        for tag, v, sc in (("1", v1, s1), ("2", v2, s2)):
            if v is None or sc is None or v.dtype != torch.uint8 or sc.dtype != torch.uint8:
                return None
            if v.dim() != 3 or sc.dim() != 3 or v.stride(1) != 1:
                return None
            E, KB, N = v.shape                  # [E, K/2, N] view, K/2 contiguous
            K = KB * 2
            if tuple(sc.shape) != (E, K // 32, N) or v.data_ptr() % 16 or v.stride(2) % 16:
                return None
            out[tag] = dict(v=v, s=sc, E=E, K=K, N=N, sbe=v.stride(0), sbn=v.stride(2),
                            sse=sc.stride(0), ssg=sc.stride(1), ssn=sc.stride(2))
        a, b = out["1"], out["2"]
        if a["K"] not in (1024, 2048, 4096) or b["K"] not in (1024, 2048) or a["N"] != 2 * b["K"]:
            return None
        lim = getattr(quant_config, "gemm1_clamp_limit", None)
        out["clamp"] = 3.0e38 if lim is None else float(lim)
        return out
    except Exception:
        return None


def mxfp4_direct_moe(output, x, plan, topk_weights, topk_ids):
    """output[M, N2] (bf16) = sum_k w_k * W2_e(act(W13_e x)) for M*top_k <= 64."""
    M = x.shape[0]
    top_k = topk_ids.size(1)
    P = M * top_k
    a, b = plan["1"], plan["2"]
    ids = topk_ids if topk_ids.dtype == torch.int32 else topk_ids.to(torch.int32)
    ids = ids.contiguous()
    tw = topk_weights.contiguous()
    if tw.dtype != torch.float32:
        tw = tw.float()
    if x.stride(1) != 1 or x.stride(0) % 8 or x.data_ptr() % 16:
        x = x.contiguous()
    c1 = torch.empty((P, a["N"]), dtype=torch.bfloat16, device=x.device)
    c3 = torch.empty((P, b["N"]), dtype=torch.bfloat16, device=x.device)
    out = output.view(M, b["N"])
    lib = _lib()
    if not getattr(lib, "_mx_init", False):
        for fn in (lib.vsh_moe_mxfp4_direct, lib.vsh_moe_mxfp4_v2, lib.vsh_moe_mxfp4_v3,
                   lib.vsh_moe_mxfp4_v4, lib.vsh_moe_swiglu):
            fn.restype = ctypes.c_int
        lib.vsh_moe_mxfp4_v4.argtypes = ([ctypes.c_void_p] + [ctypes.c_int] * 3 + [ctypes.c_void_p] * 6
                                         + [ctypes.c_int] * 4 + [ctypes.c_long] * 7 + [ctypes.c_int])
        lib.vsh_moe_mxfp4_v3.argtypes = ([ctypes.c_void_p, ctypes.c_int, ctypes.c_int] + [ctypes.c_void_p] * 6
                                         + [ctypes.c_int] * 4 + [ctypes.c_long] * 7 + [ctypes.c_int])
        lib.vsh_moe_mxfp4_direct.argtypes = ([ctypes.c_void_p, ctypes.c_int] + [ctypes.c_void_p] * 6
                                             + [ctypes.c_int] * 4 + [ctypes.c_float] + [ctypes.c_long] * 7
                                             + [ctypes.c_int])
        lib.vsh_moe_mxfp4_v2.argtypes = ([ctypes.c_void_p, ctypes.c_int] + [ctypes.c_void_p] * 6
                                         + [ctypes.c_int] * 4 + [ctypes.c_long] * 7 + [ctypes.c_int])
        lib.vsh_moe_swiglu.argtypes = ([ctypes.c_void_p] * 3 + [ctypes.c_int] * 2 + [ctypes.c_float]
                                       + [ctypes.c_long] * 2)
        if not getattr(lib, "_direct_init", False):
            _direct_init(lib)
        lib._mx_init = True
    st = torch.cuda.current_stream().cuda_stream
    ver = os.environ.get("VSH_MOE_MXFP4_V", "4")
    if ver == "1":      # v1: LDS-staged kernel, fused SwiGLU
        rc = lib.vsh_moe_mxfp4_direct(st, 1, x.data_ptr(), a["v"].data_ptr(), c1.data_ptr(), a["s"].data_ptr(),
                                      tw.data_ptr(), ids.data_ptr(), P, a["N"], a["K"], top_k, plan["clamp"],
                                      x.stride(0), a["sbe"], a["sbn"], c1.stride(0), a["sse"], a["ssn"], a["ssg"], a["E"])
        if rc == 0:
            rc = lib.vsh_moe_mxfp4_direct(st, 2, c1.data_ptr(), b["v"].data_ptr(), c3.data_ptr(), b["s"].data_ptr(),
                                          tw.data_ptr(), ids.data_ptr(), P, b["N"], b["K"], top_k, plan["clamp"],
                                          c1.stride(0), b["sbe"], b["sbn"], c3.stride(0), b["sse"], b["ssn"], b["ssg"], b["E"])
    elif ver == "4":
        act = torch.empty((P, b["K"]), dtype=torch.bfloat16, device=x.device)
        mt = 1 if M <= 1 else 2 if M <= 2 else 4
        mt = int(os.environ.get("VSH_MX_MT", mt))
        rp1 = int(os.environ.get("VSH_MX_RP1", "2"))
        rp2 = int(os.environ.get("VSH_MX_RP2", "4"))
        rc = lib.vsh_moe_mxfp4_v4(st, 1, mt, rp1, x.data_ptr(), a["v"].data_ptr(), c1.data_ptr(), a["s"].data_ptr(),
                                  tw.data_ptr(), ids.data_ptr(), P, a["N"], a["K"], top_k,
                                  x.stride(0), a["sbe"], a["sbn"], c1.stride(0), a["sse"], a["ssn"], a["ssg"], a["E"])
        if rc == 0:
            rc = lib.vsh_moe_swiglu(st, c1.data_ptr(), act.data_ptr(), P, b["K"], plan["clamp"],
                                    c1.stride(0), act.stride(0))
        if rc == 0:
            rc = lib.vsh_moe_mxfp4_v4(st, 2, mt, rp2, act.data_ptr(), b["v"].data_ptr(), c3.data_ptr(), b["s"].data_ptr(),
                                      tw.data_ptr(), ids.data_ptr(), P, b["N"], b["K"], top_k,
                                      act.stride(0), b["sbe"], b["sbn"], c3.stride(0), b["sse"], b["ssn"], b["ssg"], b["E"])
    elif ver == "3":
        act = torch.empty((P, b["K"]), dtype=torch.bfloat16, device=x.device)
        mt = 1 if M <= 1 else 2 if M <= 2 else 4      # sweep: 4 beats 6 at M=6 (occupancy)
        mt = int(os.environ.get("VSH_MX_MT", mt))
        rc = lib.vsh_moe_mxfp4_v3(st, 1, mt, x.data_ptr(), a["v"].data_ptr(), c1.data_ptr(), a["s"].data_ptr(),
                                  tw.data_ptr(), ids.data_ptr(), P, a["N"], a["K"], top_k,
                                  x.stride(0), a["sbe"], a["sbn"], c1.stride(0), a["sse"], a["ssn"], a["ssg"], a["E"])
        if rc == 0:
            rc = lib.vsh_moe_swiglu(st, c1.data_ptr(), act.data_ptr(), P, b["K"], plan["clamp"],
                                    c1.stride(0), act.stride(0))
        if rc == 0:
            rc = lib.vsh_moe_mxfp4_v3(st, 2, mt, act.data_ptr(), b["v"].data_ptr(), c3.data_ptr(), b["s"].data_ptr(),
                                      tw.data_ptr(), ids.data_ptr(), P, b["N"], b["K"], top_k,
                                      act.stride(0), b["sbe"], b["sbn"], c3.stride(0), b["sse"], b["ssn"], b["ssg"], b["E"])
    else:
        act = torch.empty((P, b["K"]), dtype=torch.bfloat16, device=x.device)
        rc = lib.vsh_moe_mxfp4_v2(st, 1, x.data_ptr(), a["v"].data_ptr(), c1.data_ptr(), a["s"].data_ptr(),
                                  tw.data_ptr(), ids.data_ptr(), P, a["N"], a["K"], top_k,
                                  x.stride(0), a["sbe"], a["sbn"], c1.stride(0), a["sse"], a["ssn"], a["ssg"], a["E"])
        if rc == 0:
            rc = lib.vsh_moe_swiglu(st, c1.data_ptr(), act.data_ptr(), P, b["K"], plan["clamp"],
                                    c1.stride(0), act.stride(0))
        if rc == 0:
            rc = lib.vsh_moe_mxfp4_v2(st, 2, act.data_ptr(), b["v"].data_ptr(), c3.data_ptr(), b["s"].data_ptr(),
                                      tw.data_ptr(), ids.data_ptr(), P, b["N"], b["K"], top_k,
                                      act.stride(0), b["sbe"], b["sbn"], c3.stride(0), b["sse"], b["ssn"], b["ssg"], b["E"])
    if rc == 0:
        rc = lib.vsh_moe_topk_sum(st, c3.data_ptr(), out.data_ptr(), M, b["N"], top_k, c3.stride(0), out.stride(0))
    if rc != 0:
        raise RuntimeError(f"vsh_moe_mxfp4_direct failed: {rc}")
    return output
