// sweep: FP8 block-scaled GEMV with ROWS output rows per wave (x reused across rows)
#include <hip/hip_runtime.h>
#include <stdint.h>
#define WAVES 8
#define WS 32
__device__ __forceinline__ uint16_t f2bf(float f) { uint32_t u = __float_as_uint(f); u += 0x7FFF + ((u >> 16) & 1); return (uint16_t)(u >> 16); }
__device__ __forceinline__ float fp8f(uint32_t b) {
  const uint32_t e = (b >> 3) & 0xF, m = b & 0x7;
  const float nrm = __uint_as_float(((b & 0x7Fu) << 20) + (120u << 23));
  const float v = e ? nrm : (float)m * 0x1p-9f;
  return (b & 0x80u) ? -v : v;
}
template <int K, int MT, int ROWS, int VEC>
__global__ __launch_bounds__(WAVES * WS) void k_fp8(const uint8_t* __restrict__ W, const float* __restrict__ S,
    const uint16_t* __restrict__ X, uint16_t* __restrict__ Y, int N, int M, long ldw, long ss, long sx, long sy) {
  constexpr int STEP = WS * 16 * VEC;
  static_assert(K % STEP == 0 && 16 * VEC <= 128, "tile");
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
      const uint4* p = (const uint4*)(W + (long)n * ldw + k0);
#pragma unroll
      for (int v = 0; v < VEC; ++v) w[r][v] = p[v];
      sc[r] = S[(long)(n >> 7) * ss + (k0 >> 7)];
    }
    float part[ROWS][MT];
#pragma unroll
    for (int r = 0; r < ROWS; ++r)
#pragma unroll
      for (int t = 0; t < MT; ++t) part[r][t] = 0.f;
#pragma unroll
    for (int v = 0; v < VEC; ++v) {
      float wf[ROWS][16];
#pragma unroll
      for (int r = 0; r < ROWS; ++r) {
        const uint32_t wu[4] = {w[r][v].x, w[r][v].y, w[r][v].z, w[r][v].w};
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
          for (int b = 0; b < 4; ++b) wf[r][4 * j + b] = fp8f((wu[j] >> (8 * b)) & 0xFF);
      }
#pragma unroll
      for (int t = 0; t < MT; ++t) {
        if (t >= M) break;
        const uint4* xp = (const uint4*)(X + (long)t * sx + k0 + v * 16);
        const uint4 xa = xp[0], xb = xp[1];
        const uint32_t xu[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
        float xf[16];
#pragma unroll
        for (int j = 0; j < 8; ++j) { xf[2 * j] = __uint_as_float(xu[j] << 16); xf[2 * j + 1] = __uint_as_float(xu[j] & 0xFFFF0000u); }
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
          float s = 0.f;
#pragma unroll
          for (int j = 0; j < 16; ++j) s += wf[r][j] * xf[j];
          part[r][t] += s;
        }
      }
    }
#pragma unroll
    for (int r = 0; r < ROWS; ++r)
#pragma unroll
      for (int t = 0; t < MT; ++t) acc[r][t] += part[r][t] * sc[r];
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
template <int K, int MT, int R, int V>
static int launch(void* st, const void* W, const void* S, const void* X, void* Y, int N, int M, long ldw, long ss, long sx, long sy) {
  if constexpr (K % (WS * 16 * V) == 0) {
    dim3 g((N + WAVES * R - 1) / (WAVES * R)), b(WAVES * WS);
    hipLaunchKernelGGL((k_fp8<K, MT, R, V>), g, b, 0, (hipStream_t)st, (const uint8_t*)W, (const float*)S, (const uint16_t*)X, (uint16_t*)Y, N, M, ldw, ss, sx, sy);
    return 0;
  } else {
    return -2;
  }
}
#define VARIANTS(X) X(1,4) X(1,2) X(2,2) X(2,1) X(4,1) X(2,4) X(4,2)
extern "C" int sweep(int variant, void* st, const void* W, const void* S, const void* X, void* Y, int N, int K, int M, long ldw, long ss, long sx, long sy) {
  int idx = 0;
#define TRY(R, V) if (variant == idx++) { \
    if (K == 4096) return M <= 4 ? launch<4096, 4, R, V>(st, W, S, X, Y, N, M, ldw, ss, sx, sy) : launch<4096, 8, R, V>(st, W, S, X, Y, N, M, ldw, ss, sx, sy); \
    if (K == 1024) return M <= 4 ? launch<1024, 4, R, V>(st, W, S, X, Y, N, M, ldw, ss, sx, sy) : launch<1024, 8, R, V>(st, W, S, X, Y, N, M, ldw, ss, sx, sy); \
    return -2; }
  VARIANTS(TRY)
  return -1;
}
