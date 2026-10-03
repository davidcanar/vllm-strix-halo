// SPDX-License-Identifier: Apache-2.0
// vllm-strix-halo: W8A16 skinny GEMV + dequant for gfx1151 (RDNA 3.5, wave32).
//
// Weights int8 [N][K] (K contiguous), symmetric, one fp32 scale per 128 K
// (S [N][K/128]); activations bf16 [M][K] (M <= 8), output bf16 [M][N].
// One wave = one output row; each lane streams VEC x 16 contiguous int8 per step
// (tile sweep on DRAM-bound shapes, PATCHES 28.6: 1 row x 64 B/lane was best),
// fp32 accumulate, shuffle reduce; 8 waves per workgroup.
#include <hip/hip_runtime.h>
#include <stdint.h>

#define WAVES 8
#define WS 32
#define MMAX 8

__device__ __forceinline__ float bf2f(uint16_t v) { return __uint_as_float(((uint32_t)v) << 16); }
__device__ __forceinline__ uint16_t f2bf(float f) {
  uint32_t u = __float_as_uint(f);
  u += 0x7FFF + ((u >> 16) & 1);
  return (uint16_t)(u >> 16);
}

template <int K, int MT, int VEC>
__global__ __launch_bounds__(WAVES * WS) void w8a16_gemv_kernel(
    const int8_t* __restrict__ W, const float* __restrict__ S, const uint16_t* __restrict__ X,
    uint16_t* __restrict__ Y, const uint16_t* __restrict__ bias, int N, int M, long sx, long sy) {
  constexpr int STEP = WS * 16 * VEC;
  static_assert(K % STEP == 0 && 16 * VEC <= 128, "tile");
  const int lane = threadIdx.x % WS, wave = threadIdx.x / WS;
  const int n = blockIdx.x * WAVES + wave;
  if (n >= N) return;
  float acc[MT];
#pragma unroll
  for (int t = 0; t < MT; ++t) acc[t] = 0.f;
  const int8_t* wrow = W + (long)n * K;
  const float* srow = S + (long)n * (K >> 7);
  for (int it = 0; it < K / STEP; ++it) {
    const int k0 = it * STEP + lane * 16 * VEC;
    uint4 w[VEC];
    const uint4* p = (const uint4*)(wrow + k0);
#pragma unroll
    for (int v = 0; v < VEC; ++v) w[v] = p[v];
    const float sc = srow[k0 >> 7];
#pragma unroll
    for (int t = 0; t < MT; ++t) {
      if (t >= M) break;
      float s = 0.f;
#pragma unroll
      for (int v = 0; v < VEC; ++v) {
        const uint4* xp = (const uint4*)(X + (long)t * sx + k0 + v * 16);
        uint4 xa = xp[0], xb = xp[1];
        const uint32_t xu[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
        const uint32_t wu[4] = {w[v].x, w[v].y, w[v].z, w[v].w};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const uint32_t a = xu[2 * j], b = xu[2 * j + 1];
          s += (float)(int8_t)(wu[j] & 0xFF) * __uint_as_float(a << 16);
          s += (float)(int8_t)((wu[j] >> 8) & 0xFF) * __uint_as_float(a & 0xFFFF0000u);
          s += (float)(int8_t)((wu[j] >> 16) & 0xFF) * __uint_as_float(b << 16);
          s += (float)(int8_t)(wu[j] >> 24) * __uint_as_float(b & 0xFFFF0000u);
        }
      }
      acc[t] += s * sc;
    }
  }
#pragma unroll
  for (int t = 0; t < MT; ++t)
#pragma unroll
    for (int off = WS / 2; off > 0; off >>= 1) acc[t] += __shfl_xor(acc[t], off, WS);
  if (lane == 0) {
    const float b = bias ? bf2f(bias[n]) : 0.f;
    for (int t = 0; t < M; ++t) Y[(long)t * sy + n] = f2bf(acc[t] + b);
  }
}

// dequant int8 [N][K] * S -> bf16 [N][K] (prefill path)
__global__ void w8_dequant_kernel(const int8_t* __restrict__ W, const float* __restrict__ S,
                                  uint16_t* __restrict__ out, long total16, int K) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total16) return;
  long e0 = i * 16;
  long n = e0 / K;
  int k = (int)(e0 - n * K);
  float s = S[n * (K >> 7) + (k >> 7)];
  uint4 v = *(const uint4*)(W + e0);
  uint32_t u[4] = {v.x, v.y, v.z, v.w};
  uint32_t o[8];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    uint16_t a = f2bf((float)(int8_t)(u[j] & 0xFF) * s), b = f2bf((float)(int8_t)((u[j] >> 8) & 0xFF) * s);
    uint16_t c = f2bf((float)(int8_t)((u[j] >> 16) & 0xFF) * s), d = f2bf((float)(int8_t)(u[j] >> 24) * s);
    o[2 * j] = (uint32_t)a | ((uint32_t)b << 16);
    o[2 * j + 1] = (uint32_t)c | ((uint32_t)d << 16);
  }
  uint4* op = (uint4*)(out + e0);
  op[0] = make_uint4(o[0], o[1], o[2], o[3]);
  op[1] = make_uint4(o[4], o[5], o[6], o[7]);
}

extern "C" int vsh_w8a16_gemv(void* stream, const void* W, const void* S, const void* X, void* Y,
                              const void* bias, int N, int K, int M, long sx, long sy) {
  if (M < 1 || M > MMAX) return -1;
  dim3 grid((N + WAVES - 1) / WAVES), block(WAVES * WS);
  hipStream_t st = (hipStream_t)stream;
#define L(KK, MM, VV) hipLaunchKernelGGL((w8a16_gemv_kernel<KK, MM, VV>), grid, block, 0, st, \
      (const int8_t*)W, (const float*)S, (const uint16_t*)X, (uint16_t*)Y, (const uint16_t*)bias, N, M, sx, sy)
#define LK(KK, VV) do { if (M == 1) L(KK, 1, VV); else if (M <= 4) L(KK, 4, VV); else L(KK, 8, VV); } while (0)
  switch (K) {
    case 1024: LK(1024, 2); break;
    case 1536: LK(1536, 1); break;
    case 2048: LK(2048, 4); break;
    case 3072: LK(3072, 2); break;
    case 4096: LK(4096, 4); break;
    case 6144: LK(6144, 4); break;
    case 8192: LK(8192, 4); break;
    default: return -2;
  }
#undef LK
#undef L
  return hipGetLastError() == hipSuccess ? 0 : -3;
}

extern "C" int vsh_w8_dequant(void* stream, const void* W, const void* S, void* out, int N, int K) {
  if (K % 128) return -1;
  long total16 = (long)N * K / 16;
  dim3 block(256), grid((unsigned)((total16 + 255) / 256));
  hipLaunchKernelGGL(w8_dequant_kernel, grid, block, 0, (hipStream_t)stream, (const int8_t*)W,
                     (const float*)S, (uint16_t*)out, total16, K);
  return hipGetLastError() == hipSuccess ? 0 : -3;
}
