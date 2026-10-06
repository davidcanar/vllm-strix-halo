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

// ---- FP8 (e4m3) block-scaled weight GEMV, bf16 activations (DeepSeek-V4 linears) ----
// W fp8 [N][ldw] (rows may be padded: ldw >= K, 16 B aligned), block scales
// S fp32 [ceil(N/128)][ss] (one per 128x128 block), X bf16 [M][K], Y bf16 [M][N].
// Same skeleton as the W8A16 kernel (one wave per output row, 16*VEC contiguous
// bytes per lane per step); each weight byte is decoded once and reused for all
// M rows. The activation is used as bf16 directly: no fp8 activation round trip
// (the stock W8A8 path quantises x per 128-group first), so this is the more
// exact of the two. FNUZ=1 decodes e4m3fnuz (exponent bias 8) instead of e4m3fn.
template <int FNUZ>
__device__ __forceinline__ float fp8_to_f32(uint32_t b) {
  const uint32_t e = (b >> 3) & 0xF, m = b & 0x7;
  // normal: exponent rebias by bits; subnormal: m * 2^(1 - bias - 3)
  const float nrm = __uint_as_float(((b & 0x7Fu) << 20) + ((FNUZ ? 119u : 120u) << 23));
  const float sub = (float)m * (FNUZ ? 0x1p-10f : 0x1p-9f);
  const float v = e ? nrm : sub;
  return (b & 0x80u) ? -v : v;
}

template <int K, int MT, int ROWS, int VEC, int FNUZ>
__global__ __launch_bounds__(WAVES * WS) void fp8_gemv_kernel(
    const uint8_t* __restrict__ W, const float* __restrict__ S, const uint16_t* __restrict__ X,
    uint16_t* __restrict__ Y, const uint16_t* __restrict__ bias, int N, int M, long ldw, long ss,
    long sx, long sy) {
  // ROWS output rows per wave: each activation load is reused for ROWS rows
  // (the 4-6-row DSpark verify was L2-bound on re-reading x; PATCHES 38 sweep)
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
    uint4 w[ROWS][VEC];
    float sc[ROWS];
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
          for (int b = 0; b < 4; ++b) wf[r][4 * j + b] = fp8_to_f32<FNUZ>((wu[j] >> (8 * b)) & 0xFF);
      }
#pragma unroll
      for (int t = 0; t < MT; ++t) {
        if (t >= M) break;
        const uint4* xp = (const uint4*)(X + (long)t * sx + k0 + v * 16);
        const uint4 xa = xp[0], xb = xp[1];
        const uint32_t xu[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
        float xf[16];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          xf[2 * j] = __uint_as_float(xu[j] << 16);
          xf[2 * j + 1] = __uint_as_float(xu[j] & 0xFFFF0000u);
        }
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
  if (lane == 0) {
    for (int r = 0; r < ROWS; ++r) {
      if (n0 + r >= N) break;
      const float b = bias ? bf2f(bias[n0 + r]) : 0.f;
      for (int t = 0; t < M; ++t) Y[(long)t * sy + n0 + r] = f2bf(acc[r][t] + b);
    }
  }
}

template <int K, int MT, int ROWS, int VEC>
static void fp8_launch(hipStream_t st, int fnuz, const void* W, const void* S, const void* X, void* Y,
                       const void* bias, int N, int M, long ldw, long ss, long sx, long sy) {
  dim3 grid((N + WAVES * ROWS - 1) / (WAVES * ROWS)), block(WAVES * WS);
#define FL(FZ) hipLaunchKernelGGL((fp8_gemv_kernel<K, MT, ROWS, VEC, FZ>), grid, block, 0, st, \
      (const uint8_t*)W, (const float*)S, (const uint16_t*)X, (uint16_t*)Y, (const uint16_t*)bias, \
      N, M, ldw, ss, sx, sy)
  if (fnuz) FL(1); else FL(0);
#undef FL
}

extern "C" int vsh_fp8_gemv(void* stream, const void* W, const void* S, const void* X, void* Y,
                            const void* bias, int N, int K, int M, long ldw, long ss, long sx,
                            long sy, int fnuz) {
  if (M < 1 || M > MMAX) return -1;
  if (ldw < K || (ldw & 15)) return -4;
  hipStream_t st = (hipStream_t)stream;
#define A_ W, S, X, Y, bias, N, M, ldw, ss, sx, sy
  // tiles from the PATCHES 38 sweep (DS4 shapes: K = 1024 / 4096)
  switch (K) {
    case 1024:
      if (M == 1) fp8_launch<1024, 1, 2, 1>(st, fnuz, A_);
      else if (M <= 4) fp8_launch<1024, 4, 2, 1>(st, fnuz, A_);
      else fp8_launch<1024, 8, 4, 2>(st, fnuz, A_);
      break;
    case 4096:
      if (M == 1) fp8_launch<4096, 1, 2, 2>(st, fnuz, A_);
      else if (M <= 4) fp8_launch<4096, 4, 2, 2>(st, fnuz, A_);
      else fp8_launch<4096, 8, 4, 2>(st, fnuz, A_);
      break;
    case 12288:   // DSpark drafter: 3 target hidden states (3 x 4096) -> 4096
      if (M == 1) fp8_launch<12288, 1, 2, 2>(st, fnuz, A_);
      else if (M <= 4) fp8_launch<12288, 4, 2, 2>(st, fnuz, A_);
      else fp8_launch<12288, 8, 4, 2>(st, fnuz, A_);
      break;
#define GEN(KK, VV) case KK: \
      if (M == 1) fp8_launch<KK, 1, 1, VV>(st, fnuz, A_); \
      else if (M <= 4) fp8_launch<KK, 4, 1, VV>(st, fnuz, A_); \
      else fp8_launch<KK, 8, 1, VV>(st, fnuz, A_); \
      break;
    GEN(512, 1) GEN(1536, 1) GEN(2048, 4) GEN(3072, 2) GEN(6144, 4) GEN(7168, 1) GEN(8192, 4)
#undef GEN
    default: return -2;
  }
#undef A_
  return hipGetLastError() == hipSuccess ? 0 : -3;
}

// ---- fp32-weight GEMV for the MoE router gate (fp32 weights, bf16 x, fp32 out) ----
// One wave per output row; each lane reads 16 contiguous fp32 weights (4 x 16 B)
// and 16 bf16 activations per step.
template <int K, int MT>
__global__ __launch_bounds__(WAVES * WS) void w32_gemv_kernel(
    const float* __restrict__ W, const uint16_t* __restrict__ X, float* __restrict__ Y,
    int N, int M, long sx, long sy) {
  constexpr int STEP = WS * 16;
  static_assert(K % STEP == 0, "K");
  const int lane = threadIdx.x % WS, wave = threadIdx.x / WS;
  const int n = blockIdx.x * WAVES + wave;
  if (n >= N) return;
  float acc[MT];
#pragma unroll
  for (int t = 0; t < MT; ++t) acc[t] = 0.f;
  const float* wrow = W + (long)n * K;
#pragma unroll 2
  for (int it = 0; it < K / STEP; ++it) {
    const int k0 = it * STEP + lane * 16;
    const float4* p = (const float4*)(wrow + k0);
    float4 w4[4] = {p[0], p[1], p[2], p[3]};
    const float* w = (const float*)w4;
#pragma unroll
    for (int t = 0; t < MT; ++t) {
      if (t >= M) break;
      const uint4* xp = (const uint4*)(X + (long)t * sx + k0);
      uint4 xa = xp[0], xb = xp[1];
      const uint32_t xu[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
      float s = 0.f;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        s += w[2 * j] * __uint_as_float(xu[j] << 16);
        s += w[2 * j + 1] * __uint_as_float(xu[j] & 0xFFFF0000u);
      }
      acc[t] += s;
    }
  }
#pragma unroll
  for (int t = 0; t < MT; ++t)
#pragma unroll
    for (int off = WS / 2; off > 0; off >>= 1) acc[t] += __shfl_xor(acc[t], off, WS);
  if (lane == 0)
    for (int t = 0; t < M; ++t) Y[(long)t * sy + n] = acc[t];
}

extern "C" int vsh_w32_gemv(void* stream, const void* W, const void* X, void* Y, int N, int K, int M,
                            long sx, long sy) {
  if (M < 1 || M > MMAX) return -1;
  dim3 grid((N + WAVES - 1) / WAVES), block(WAVES * WS);
  hipStream_t st = (hipStream_t)stream;
#define L32(KK, MM) hipLaunchKernelGGL((w32_gemv_kernel<KK, MM>), grid, block, 0, st, (const float*)W, \
      (const uint16_t*)X, (float*)Y, N, M, sx, sy)
#define LK32(KK) do { if (M == 1) L32(KK, 1); else if (M <= 4) L32(KK, 4); else L32(KK, 8); } while (0)
  switch (K) {
    case 2048: LK32(2048); break;
    case 3072: LK32(3072); break;
    case 4096: LK32(4096); break;
    case 6144: LK32(6144); break;
    case 7168: LK32(7168); break;
    default: return -2;
  }
#undef LK32
#undef L32
  return hipGetLastError() == hipSuccess ? 0 : -3;
}

// ---- bf16-weight GEMV with fp32 output (MoE router gate: bf16 W, bf16 x) ----------
// GateLinear tier 4 (torch.mm out_dtype=fp32 -> hipBLASLt) costs ~150 us for a
// 288 x 4096 gate at decode. SPLIT waves per output row, each lane reads 16
// contiguous bf16 weights (2 x 16 B) per step; partials reduced through LDS.
#define W16_SPLIT 4
template <int K, int MT>
__global__ __launch_bounds__(WAVES * WS) void w16_gemv_kernel(
    const uint16_t* __restrict__ W, const uint16_t* __restrict__ X, float* __restrict__ Y,
    int N, int M, long sx, long sy) {
  constexpr int STEP = WS * 16 * W16_SPLIT;
  static_assert(K % STEP == 0, "K");
  constexpr int RPB = WAVES / W16_SPLIT;            // output rows per block
  __shared__ float part[WAVES][MT];
  const int lane = threadIdx.x % WS, wave = threadIdx.x / WS;
  const int r = wave / W16_SPLIT, sp = wave % W16_SPLIT;
  const int n = blockIdx.x * RPB + r;
  float acc[MT];
#pragma unroll
  for (int t = 0; t < MT; ++t) acc[t] = 0.f;
  if (n < N) {
    const uint16_t* wrow = W + (long)n * K;
#pragma unroll 2
    for (int it = 0; it < K / STEP; ++it) {
      const int k0 = it * STEP + (sp * WS + lane) * 16;
      const uint4* p = (const uint4*)(wrow + k0);
      const uint4 wa = p[0], wb = p[1];
      const uint32_t wu[8] = {wa.x, wa.y, wa.z, wa.w, wb.x, wb.y, wb.z, wb.w};
#pragma unroll
      for (int t = 0; t < MT; ++t) {
        if (t >= M) break;
        const uint4* xp = (const uint4*)(X + (long)t * sx + k0);
        const uint4 xa = xp[0], xb = xp[1];
        const uint32_t xu[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
        float s = 0.f;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          s += __uint_as_float(wu[j] << 16) * __uint_as_float(xu[j] << 16);
          s += __uint_as_float(wu[j] & 0xFFFF0000u) * __uint_as_float(xu[j] & 0xFFFF0000u);
        }
        acc[t] += s;
      }
    }
  }
#pragma unroll
  for (int t = 0; t < MT; ++t)
#pragma unroll
    for (int off = WS / 2; off > 0; off >>= 1) acc[t] += __shfl_xor(acc[t], off, WS);
  if (lane == 0)
#pragma unroll
    for (int t = 0; t < MT; ++t) part[wave][t] = acc[t];
  __syncthreads();
  if (sp == 0 && lane < M && n < N) {
    float s = 0.f;
#pragma unroll
    for (int q = 0; q < W16_SPLIT; ++q) s += part[r * W16_SPLIT + q][lane];
    Y[(long)lane * sy + n] = s;
  }
}

extern "C" int vsh_w16_gemv(void* stream, const void* W, const void* X, void* Y, int N, int K, int M,
                            long sx, long sy) {
  if (M < 1 || M > MMAX) return -1;
  constexpr int RPB = WAVES / W16_SPLIT;
  dim3 grid((N + RPB - 1) / RPB), block(WAVES * WS);
  hipStream_t st = (hipStream_t)stream;
#define L16(KK, MM) hipLaunchKernelGGL((w16_gemv_kernel<KK, MM>), grid, block, 0, st, (const uint16_t*)W, \
      (const uint16_t*)X, (float*)Y, N, M, sx, sy)
#define LK16(KK) do { if (M == 1) L16(KK, 1); else if (M <= 4) L16(KK, 4); else L16(KK, 8); } while (0)
  switch (K) {
    case 2048: LK16(2048); break;
    case 4096: LK16(4096); break;
    case 6144: LK16(6144); break;
    case 8192: LK16(8192); break;
    default: return -2;
  }
#undef LK16
#undef L16
  return hipGetLastError() == hipSuccess ? 0 : -3;
}

// ---- fused MoE routing for sigmoid + correction-bias top-k (one expert group) ----
// Matches vLLM grouped_topk(scoring="sigmoid", e_score_correction_bias, n_group=1)
// with the deterministic selection order (biased score desc, expert index asc):
// weights = unbiased sigmoid scores of the selected experts, renormalised, scaled.
#define RT 256
__global__ __launch_bounds__(RT) void sigmoid_bias_topk_kernel(
    const float* __restrict__ logits, const float* __restrict__ bias, float* __restrict__ w_out,
    int32_t* __restrict__ id_out, int E, int K, long ls, float scale, int renorm) {
  __shared__ float biased[512];
  __shared__ float orig[512];
  __shared__ float red_v[RT / WS];
  __shared__ int red_i[RT / WS];
  __shared__ int sel[16];
  const int row = blockIdx.x, t = threadIdx.x, lane = t % WS, wv = t / WS;
  for (int e = t; e < E; e += RT) {
    const float x = logits[(long)row * ls + e];
    const float s = 1.0f / (1.0f + expf(-x));
    orig[e] = s;
    // NaN ranks highest (as in torch.topk), so garbage inputs -- e.g. the dummy
    // batches of CUDA-graph warmup -- still yield valid, distinct expert ids.
    const float b = s + bias[e];
    biased[e] = b != b ? INFINITY : b;
  }
  __syncthreads();
  for (int k = 0; k < K; ++k) {
    float bv = -INFINITY; int bi = 0x7fffffff;
    for (int e = t; e < E; e += RT) {
      const float v = biased[e];
      if (v > bv || (v == bv && e < bi)) { bv = v; bi = e; }
    }
#pragma unroll
    for (int off = WS / 2; off > 0; off >>= 1) {
      const float ov = __shfl_xor(bv, off, WS); const int oi = __shfl_xor(bi, off, WS);
      if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    if (lane == 0) { red_v[wv] = bv; red_i[wv] = bi; }
    __syncthreads();
    if (t == 0) {
      float v = red_v[0]; int i = red_i[0];
      for (int w = 1; w < RT / WS; ++w)
        if (red_v[w] > v || (red_v[w] == v && red_i[w] < i)) { v = red_v[w]; i = red_i[w]; }
      sel[k] = i;
      biased[i] = -INFINITY;
    }
    __syncthreads();
  }
  if (t == 0) {
    float sum = 0.f;
    for (int k = 0; k < K; ++k) sum += orig[sel[k]];
    const float inv = renorm ? 1.0f / sum : 1.0f;
    for (int k = 0; k < K; ++k) {
      w_out[(long)row * K + k] = orig[sel[k]] * inv * scale;
      id_out[(long)row * K + k] = sel[k];
    }
  }
}

extern "C" int vsh_sigmoid_bias_topk(void* stream, const void* logits, const void* bias, void* w_out,
                                     void* id_out, int M, int E, int K, long ls, float scale, int renorm) {
  if (E > 512 || K > 16 || M < 1) return -1;
  hipLaunchKernelGGL(sigmoid_bias_topk_kernel, dim3(M), dim3(RT), 0, (hipStream_t)stream,
                     (const float*)logits, (const float*)bias, (float*)w_out, (int32_t*)id_out, E, K, ls,
                     scale, renorm);
  return hipGetLastError() == hipSuccess ? 0 : -3;
}


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
