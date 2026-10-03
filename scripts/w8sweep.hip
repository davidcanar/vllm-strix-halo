// sweep variant of the W8A16 GEMV: ROWS rows per wave, VEC x 16 B contiguous per lane per step
#include <hip/hip_runtime.h>
#include <stdint.h>
#define WAVES 8
#define WS 32
__device__ __forceinline__ uint16_t f2bf(float f) { uint32_t u = __float_as_uint(f); u += 0x7FFF + ((u >> 16) & 1); return (uint16_t)(u >> 16); }

template <int K, int MT, int ROWS, int VEC>
__global__ __launch_bounds__(WAVES * WS) void k_w8(const int8_t* __restrict__ W, const float* __restrict__ S,
    const uint16_t* __restrict__ X, uint16_t* __restrict__ Y, int N, int M, long sx, long sy) {
  constexpr int STEP = WS * 16 * VEC;
  static_assert(K % STEP == 0, "K");
  const int lane = threadIdx.x % WS, wave = threadIdx.x / WS;
  const int n0 = (blockIdx.x * WAVES + wave) * ROWS;
  if (n0 >= N) return;
  float acc[ROWS][MT];
#pragma unroll
  for (int r = 0; r < ROWS; ++r)
#pragma unroll
    for (int t = 0; t < MT; ++t) acc[r][t] = 0.f;
  for (int it = 0; it < K / STEP; ++it) {
    const int k0 = it * STEP + lane * 16 * VEC;
    uint4 w[ROWS][VEC]; float sc[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      const int n = min(n0 + r, N - 1);
      const uint4* p = (const uint4*)(W + (long)n * K + k0);
#pragma unroll
      for (int v = 0; v < VEC; ++v) w[r][v] = p[v];
      sc[r] = S[(long)n * (K >> 7) + (k0 >> 7)];
    }
#pragma unroll
    for (int t = 0; t < MT; ++t) {
      if (t >= M) break;
#pragma unroll
      for (int v = 0; v < VEC; ++v) {
        const uint4* xp = (const uint4*)(X + (long)t * sx + k0 + v * 16);
        uint4 xa = xp[0], xb = xp[1];
        uint32_t xu[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
        float x[16];
#pragma unroll
        for (int j = 0; j < 8; ++j) { x[2 * j] = __uint_as_float(xu[j] << 16); x[2 * j + 1] = __uint_as_float(xu[j] & 0xFFFF0000u); }
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
          const uint32_t wu[4] = {w[r][v].x, w[r][v].y, w[r][v].z, w[r][v].w};
          float s = 0.f;
#pragma unroll
          for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int b = 0; b < 4; ++b) s += (float)(int8_t)((wu[j] >> (8 * b)) & 0xFF) * x[j * 4 + b];
          acc[r][t] += s * sc[r];
        }
      }
    }
  }
#pragma unroll
  for (int r = 0; r < ROWS; ++r)
#pragma unroll
    for (int t = 0; t < MT; ++t)
#pragma unroll
      for (int off = WS / 2; off > 0; off >>= 1) acc[r][t] += __shfl_xor(acc[r][t], off, WS);
  if (lane == 0)
    for (int t = 0; t < M; ++t)
      for (int r = 0; r < ROWS; ++r) if (n0 + r < N) Y[(long)t * sy + n0 + r] = f2bf(acc[r][t]);
}

#define VARIANTS(X) X(4,1) X(2,1) X(1,1) X(2,2) X(1,2) X(1,4) X(2,4) X(4,2)
extern "C" int sweep(int variant, void* st, const void* W, const void* S, const void* X, void* Y, int N, int K, int M, long sx, long sy) {
  int idx = 0;
#define TRY(R, V) if (variant == idx++) { \
    dim3 g((N + WAVES * R - 1) / (WAVES * R)), b(WAVES * WS); \
    if (K == 4096) { if (M == 1) hipLaunchKernelGGL((k_w8<4096,1,R,V>), g, b, 0, (hipStream_t)st, (const int8_t*)W, (const float*)S, (const uint16_t*)X, (uint16_t*)Y, N, M, sx, sy); \
                     else hipLaunchKernelGGL((k_w8<4096,4,R,V>), g, b, 0, (hipStream_t)st, (const int8_t*)W, (const float*)S, (const uint16_t*)X, (uint16_t*)Y, N, M, sx, sy); } \
    else if (K == 8192) { if (M == 1) hipLaunchKernelGGL((k_w8<8192,1,R,V>), g, b, 0, (hipStream_t)st, (const int8_t*)W, (const float*)S, (const uint16_t*)X, (uint16_t*)Y, N, M, sx, sy); \
                     else hipLaunchKernelGGL((k_w8<8192,4,R,V>), g, b, 0, (hipStream_t)st, (const int8_t*)W, (const float*)S, (const uint16_t*)X, (uint16_t*)Y, N, M, sx, sy); } \
    return 0; }
  VARIANTS(TRY)
  return -1;
}
