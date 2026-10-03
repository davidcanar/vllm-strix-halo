# SPDX-License-Identifier: Apache-2.0
# vllm-strix-halo: load-time int8 (W8A16, symmetric, group 128) for BF16 linears.
#
# The AWQ checkpoint keeps attention/KDA/MLA projections, shared experts and the
# dense MLP layers in BF16 (~49 ms/step of skinny GEMV at decode). After loading,
# eligible UnquantizedLinearMethod layers are quantized to int8 with one fp32 scale
# per 128 inputs (the ds4 Q8_0 idea, done online -- no checkpoint rewrite), the
# BF16 copy is freed, decode rows (<= 8) run libvsh_w8a16.so's GEMV and larger
# batches dequantize to a transient BF16 weight and use the stock GEMM.
import ctypes
import os
import re

import torch

_LIB = None
_PATHS = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "libvsh_w8a16.so"),
          "/opt/venv/lib/python3.12/site-packages/vllm/libvsh_w8a16.so"]
# never touch: weights read directly elsewhere (MLA absorbs kv_b_proj; the indexer
# slices wk_weights_proj), embeddings/heads (other path), vision encoder.
_EXCLUDE = re.compile(r"(kv_b_proj|wk_weights_proj|embed|lm_head|visual|vision|\.gate$)")
GROUP = 128
MAX_ROWS = 8


def _lib():
    global _LIB
    if _LIB is None:
        for p in _PATHS:
            if os.path.exists(p):
                _LIB = ctypes.CDLL(p)
                break
        else:
            raise RuntimeError("libvsh_w8a16.so not found")
        _LIB.vsh_w8a16_gemv.restype = ctypes.c_int
        _LIB.vsh_w8a16_gemv.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 3 + [ctypes.c_long] * 2
        _LIB.vsh_w8_dequant.restype = ctypes.c_int
        _LIB.vsh_w8_dequant.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 2
    return _LIB


def enabled() -> bool:
    return os.environ.get("VSH_W8A16", "0") not in ("", "0", "off")


def eligible(layer: torch.nn.Module) -> bool:
    w = getattr(layer, "weight", None)
    if w is None or not isinstance(w, torch.Tensor) or w.dtype != torch.bfloat16 or w.ndim != 2:
        return False
    if not w.is_cuda or not w.is_contiguous():
        return False
    n, k = w.shape
    if k not in (1024, 1536, 2048, 3072, 4096, 6144, 8192) or n < 32:
        return False
    if _EXCLUDE.search(getattr(layer, "prefix", "") or ""):
        return False
    # only layers whose forward goes through quant_method.apply (subclasses that
    # override forward -- e.g. the MoE router gate -- read .weight directly)
    from vllm.model_executor.layers import linear as _L
    std = {getattr(_L, c).forward for c in ("ColumnParallelLinear", "RowParallelLinear", "ReplicatedLinear")
           if hasattr(_L, c)}
    if type(layer).forward not in std:
        return False
    inc = os.environ.get("VSH_W8A16_INCLUDE", "")
    if inc and not re.search(inc, getattr(layer, "prefix", "") or ""):
        return False
    return True


@torch.no_grad()
def quantize_(layer: torch.nn.Module, keep_bf16: bool = False) -> None:
    w = layer.weight.data
    n, k = w.shape
    q = torch.empty((n, k), dtype=torch.int8, device=w.device)
    s = torch.empty((n, k // GROUP), dtype=torch.float32, device=w.device)
    step = max(1, (64 << 20) // (k * 4))               # bound the fp32 temporary to ~64 MB
    for i in range(0, n, step):
        wf = w[i:i + step].float().view(-1, k // GROUP, GROUP)
        sc = wf.abs().amax(dim=-1).clamp_min(1e-12) / 127.0
        q[i:i + step] = torch.round(wf / sc[..., None]).clamp_(-127, 127).to(torch.int8).view(-1, k)
        s[i:i + step] = sc
        del wf, sc
    layer.vsh_w8_q = q
    layer.vsh_w8_s = s
    layer.vsh_w8_shape = (n, k)
    # free the BF16 copy (keep a 0-element parameter so attribute access still works)
    if not keep_bf16:
        layer.weight.data = torch.empty(0, dtype=torch.bfloat16, device=w.device)
    del w


def apply(layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    n, k = layer.vsh_w8_shape
    x2 = x.reshape(-1, k)
    m = x2.size(0)
    if 0 < m <= MAX_ROWS and x.dtype == torch.bfloat16 and (bias is None or bias.dtype == torch.bfloat16):
        x2 = x2.contiguous()
        y = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
        rc = _lib().vsh_w8a16_gemv(
            torch.cuda.current_stream().cuda_stream, layer.vsh_w8_q.data_ptr(),
            layer.vsh_w8_s.data_ptr(), x2.data_ptr(), y.data_ptr(),
            bias.data_ptr() if bias is not None else None, n, k, m, x2.stride(0), y.stride(0))
        if rc != 0:
            raise RuntimeError(f"vsh_w8a16_gemv failed: {rc}")
        return y.reshape(*x.shape[:-1], n)
    wt = torch.empty((n, k), dtype=torch.bfloat16, device=x.device)
    rc = _lib().vsh_w8_dequant(torch.cuda.current_stream().cuda_stream, layer.vsh_w8_q.data_ptr(),
                               layer.vsh_w8_s.data_ptr(), wt.data_ptr(), n, k)
    if rc != 0:
        raise RuntimeError(f"vsh_w8_dequant failed: {rc}")
    return torch.nn.functional.linear(x, wt.to(x.dtype), bias)


# ---- fp32 router gate GEMV (GateLinear on ROCm, fp32 weights) -------------------
def w32_ok(weight: torch.Tensor, x: torch.Tensor) -> bool:
    return (os.environ.get("VSH_ROUTER_GEMV", "1") not in ("", "0", "off")
            and weight.dtype == torch.float32 and weight.ndim == 2 and weight.is_contiguous()
            and weight.shape[1] in (2048, 3072, 4096, 6144, 7168)
            and x.dtype == torch.bfloat16 and x.dim() == 2 and 0 < x.shape[0] <= MAX_ROWS)


def w32_gemv(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    n, k = weight.shape
    x = x.contiguous()
    y = torch.empty((x.shape[0], n), dtype=torch.float32, device=x.device)
    lib = _lib()
    if not getattr(lib, "_w32_init", False):
        lib.vsh_w32_gemv.restype = ctypes.c_int
        lib.vsh_w32_gemv.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 3 + [ctypes.c_long] * 2
        lib._w32_init = True
    rc = lib.vsh_w32_gemv(torch.cuda.current_stream().cuda_stream, weight.data_ptr(), x.data_ptr(),
                          y.data_ptr(), n, k, x.shape[0], x.stride(0), y.stride(0))
    if rc != 0:
        raise RuntimeError(f"vsh_w32_gemv failed: {rc}")
    return y
