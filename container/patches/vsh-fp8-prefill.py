#!/usr/bin/env python3
"""vllm-strix-halo: DS4 block-FP8 linears at prefill sizes -> bf16 weight + hipBLASLt (PATCHES.md 41).

The stock path for M > 8 rows quantises the activation to fp8 and runs Triton's
_w8a8_triton_block_scaled_mm, which has no fp8 dot on gfx1151: 2.5-9 ms per DS4
linear at a 512-token prefill chunk, 49 % of DS4 prefill GPU time. Dequantising
the fp8 weight to bf16 in one HIP pass (vsh_fp8_dequant, exact: e8m0 scales are
powers of two) and multiplying with torch.mm (hipBLASLt) is 5-8x faster and, with
no fp8 activation round trip, more exact (rel err 1.7e-3 vs 2.6e-2 vs fp32).
Decode rows (M <= 8) keep the FP8 GEMV (vsh-fp8-gemv). Env VSH_FP8_PREFILL=0 off.

Usage (inside the container, after vsh-fp8-gemv.py):
    python3 vsh-fp8-prefill.py [path/to/vsh_w8a16.hip]   # .hip edit is optional
"""
import ast
import os
import sys
from pathlib import Path

SP = Path("/opt/venv/lib/python3.12/site-packages/vllm")
TRI = Path(os.environ.get("VSH_PATCH_TRI", SP / "model_executor/kernels/linear/scaled_mm/triton.py"))
W8 = Path(os.environ.get("VSH_PATCH_W8", SP / "model_executor/layers/vsh_w8a16.py"))
MARK = "[vsh-fp8-prefill]"

HIP = r'''

// ---- FP8 block-scaled weight -> bf16 (DS4 prefill: dequant + hipBLASLt) [vsh-fp8-prefill] ----
// One thread turns 16 consecutive fp8 of a row into 16 bf16 with the row's 128x128
// block scale (fp32, [N/128][K/128]). e8m0 scales are powers of two, so the result
// is exact; the caller multiplies with torch.mm (hipBLASLt).
template <int FNUZ>
__global__ __launch_bounds__(256) void fp8_dequant_kernel(const uint8_t* __restrict__ W, const float* __restrict__ S,
                                                         uint16_t* __restrict__ Y, int N, int K, long ldw, long ss,
                                                         long ldy) {
  const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  const int kq = K / 16;
  if (idx >= (long)N * kq) return;
  const int n = (int)(idx / kq), k0 = (int)(idx % kq) * 16;
  const uint4 w = *(const uint4*)(W + (long)n * ldw + k0);
  const float s = S[(long)(n >> 7) * ss + (k0 >> 7)];
  const uint32_t wu[4] = {w.x, w.y, w.z, w.w};
  uint32_t o[8];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const float f0 = fp8_to_f32<FNUZ>(wu[j] & 0xFF) * s, f1 = fp8_to_f32<FNUZ>((wu[j] >> 8) & 0xFF) * s;
    const float f2 = fp8_to_f32<FNUZ>((wu[j] >> 16) & 0xFF) * s, f3 = fp8_to_f32<FNUZ>(wu[j] >> 24) * s;
    o[2 * j] = (uint32_t)f2bf(f0) | ((uint32_t)f2bf(f1) << 16);
    o[2 * j + 1] = (uint32_t)f2bf(f2) | ((uint32_t)f2bf(f3) << 16);
  }
  uint4* dst = (uint4*)(Y + (long)n * ldy + k0);
  dst[0] = make_uint4(o[0], o[1], o[2], o[3]);
  dst[1] = make_uint4(o[4], o[5], o[6], o[7]);
}

extern "C" int vsh_fp8_dequant(void* stream, const void* W, const void* S, void* Y, int N, int K,
                               long ldw, long ss, long ldy, int fnuz) {
  if (K % 16 || ldw < K || (ldw & 15) || (ldy & 7) || N < 1) return -1;
  const long total = (long)N * (K / 16);
  dim3 block(256), grid((unsigned)((total + 255) / 256));
  hipStream_t st = (hipStream_t)stream;
  if (fnuz)
    hipLaunchKernelGGL(fp8_dequant_kernel<1>, grid, block, 0, st, (const uint8_t*)W, (const float*)S, (uint16_t*)Y,
                       N, K, ldw, ss, ldy);
  else
    hipLaunchKernelGGL(fp8_dequant_kernel<0>, grid, block, 0, st, (const uint8_t*)W, (const float*)S, (uint16_t*)Y,
                       N, K, ldw, ss, ldy);
  return hipGetLastError() == hipSuccess ? 0 : -3;
}
'''

PY = '''

# ---- FP8 block-scaled linear at prefill sizes [vsh-fp8-prefill] -----------------------
# M > 8 rows: one HIP pass fp8 -> bf16 weight (exact with e8m0 block scales), then
# torch.mm on hipBLASLt. 5-8x the stock Triton block GEMM at 64-512 rows on gfx1151
# and more exact (the stock path quantises the activation to fp8 first).
def fp8_prefill_ok(x2d: torch.Tensor, weight: torch.Tensor, block_scale, group_shape) -> bool:
    return (os.environ.get("VSH_FP8_PREFILL", "1") not in ("", "0", "off")
            and x2d.dtype == torch.bfloat16 and x2d.dim() == 2 and x2d.shape[0] > MAX_ROWS
            and weight.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz) and weight.dim() == 2
            and weight.stride(1) == 1 and weight.stride(0) >= weight.shape[1]
            and weight.stride(0) % 16 == 0 and weight.data_ptr() % 16 == 0
            and weight.shape[1] % 16 == 0 and x2d.shape[1] == weight.shape[1]
            and list(group_shape) == [128, 128]
            and block_scale is not None and block_scale.dim() == 2
            and block_scale.shape[0] * 128 >= weight.shape[0]
            and block_scale.shape[1] * 128 >= weight.shape[1])


def fp8_dequant(weight: torch.Tensor, block_scale: torch.Tensor) -> torch.Tensor:
    n, k = weight.shape
    s = _fp8_scales(block_scale)
    y = torch.empty((n, k), dtype=torch.bfloat16, device=weight.device)
    lib = _lib()
    if not getattr(lib, "_fp8dq_init", False):
        lib.vsh_fp8_dequant.restype = ctypes.c_int
        lib.vsh_fp8_dequant.argtypes = ([ctypes.c_void_p] * 4 + [ctypes.c_int] * 2
                                        + [ctypes.c_long] * 3 + [ctypes.c_int])
        lib._fp8dq_init = True
    rc = lib.vsh_fp8_dequant(torch.cuda.current_stream().cuda_stream, weight.data_ptr(), s.data_ptr(),
                             y.data_ptr(), n, k, weight.stride(0), s.stride(0), y.stride(0),
                             int(weight.dtype == torch.float8_e4m3fnuz))
    if rc != 0:
        raise RuntimeError(f"vsh_fp8_dequant failed: {rc}")
    return y


def fp8_prefill_mm(x2d: torch.Tensor, weight: torch.Tensor, block_scale: torch.Tensor) -> torch.Tensor:
    return torch.mm(x2d, fp8_dequant(weight, block_scale).t())
'''

if len(sys.argv) > 1:   # source tree copy of the kernel (the .so is built from it)
    hp = Path(sys.argv[1]); h = hp.read_text()
    if "fp8_dequant_kernel" not in h:
        hp.write_text(h.rstrip("\n") + "\n" + HIP)
        print("vsh-fp8-prefill: kernel appended to", hp)

w = W8.read_text()
if MARK not in w:
    w = w.rstrip("\n") + "\n" + PY
    ast.parse(w)
    W8.write_text(w)
    print("vsh-fp8-prefill: vsh_w8a16.py helpers added")

t = TRI.read_text() if str(TRI) != "skip" else MARK
if MARK not in t:
    old_log = "    _vsh_fp8_logged = [False, False, False]\n"
    assert t.count(old_log) == 1
    t = t.replace(old_log, "    _vsh_fp8_logged = [False, False, False, False, False]\n")
    old = '''                if not cls._vsh_fp8_logged[1]:
                    cls._vsh_fp8_logged[1] = True
                    w, bs = params.weight, params.block_scale
                    logger.info("[vsh-fp8-gemv] fallback for M=%d: weight %s %s stride %s, scale %s %s",
                                input_2d.shape[0], tuple(w.shape), w.dtype, tuple(w.stride()),
                                None if bs is None else tuple(bs.shape),
                                None if bs is None else bs.dtype)
        return super().apply_weights(layer, x, bias, **kwargs)
'''
    new = '''                if not cls._vsh_fp8_logged[1]:
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
'''
    assert t.count(old) == 1, "apply_weights anchor"
    t = t.replace(old, new)
    old2 = '''                    return _w8.fp8_gemv(a_bf, B, Bs).to(self.config.out_dtype)
        return torch.ops.vllm.w8a8_triton_block_scaled_mm_func(
'''
    new2 = '''                    return _w8.fp8_gemv(a_bf, B, Bs).to(self.config.out_dtype)
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
'''
    assert t.count(old2) == 1, "apply_block_scaled_mm anchor"
    t = t.replace(old2, new2)
    ast.parse(t)
    TRI.write_text(t)
    print("vsh-fp8-prefill: triton.py hooks added")
else:
    print("vsh-fp8-prefill: already applied")
