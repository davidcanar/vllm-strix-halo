// SPDX-License-Identifier: Apache-2.0
// vllm-strix-halo: decode-specialised int4 (W4A16, symmetric uint4b8, group 128)
// MoE GEMV for gfx1151 (RDNA 3.5, wave32), wvSplitK-style.
//
// Same inputs/outputs as vLLM's fused_moe_kernel_gptq_awq (moe_align_block_size
// blocks): B uint8 [E][N][K/2] (K contiguous, low nibble = even k),
// S bf16 [E][N][K/128], A bf16 [rows][K], C bf16 rows indexed by the pair id
// from sorted_token_ids. One workgroup = (expert block, WAVES*ROWS output rows);
// each lane streams 16 contiguous packed bytes per row per step, activations of
// the block's valid tokens are staged once in LDS, cross-lane shuffle reduce.
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <stdint.h>

#define WAVES 8
#define ROWS 4          // output rows per wave
#define TMAX 4          // tokens per LDS pass
#define WS 32           // wave size (gfx11 wave32)

__device__ __forceinline__ float bf2f(uint16_t v) { return __uint_as_float(((uint32_t)v) << 16); }
__device__ __forceinline__ uint16_t f2bf(float f) {
  uint32_t u = __float_as_uint(f);
  u += 0x7FFF + ((u >> 16) & 1);           // round to nearest even
  return (uint16_t)(u >> 16);
}

template <int K>
__global__ __launch_bounds__(WAVES * WS) void moe_int4_gemv_kernel(
    const uint16_t* __restrict__ A, const uint8_t* __restrict__ B, uint16_t* __restrict__ C,
    const uint16_t* __restrict__ S, const float* __restrict__ topk_w,
    const int32_t* __restrict__ sorted_ids, const int32_t* __restrict__ expert_ids,
    const int32_t* __restrict__ ntpp, int N, int num_valid, int BM, int top_k, int mul_w,
    long stride_am, long stride_be, long stride_bn, long stride_cm,
    long stride_se, long stride_sn) {
  constexpr int KB = K / 2;                // packed bytes per row
  constexpr int ITERS = KB / (WS * 16);    // 16 bytes per lane per step
  static_assert(KB % (WS * 16) == 0, "K must be a multiple of 1024");
  __shared__ uint16_t xs[TMAX][K];
  __shared__ int toks[64];
  __shared__ int ntok;

  const int pid_m = blockIdx.x;
  if (pid_m * BM >= ntpp[0]) return;
  const int lane = threadIdx.x % WS, wave = threadIdx.x / WS;
  if (threadIdx.x == 0) ntok = 0;
  __syncthreads();
  if (threadIdx.x < BM) {
    int t = sorted_ids[pid_m * BM + threadIdx.x];
    if (t < num_valid) { int s = atomicAdd(&ntok, 1); toks[s] = t; }
  }
  __syncthreads();
  const int nt = ntok;
  if (nt == 0) return;
  const long e = expert_ids[pid_m];
  const int n0 = (blockIdx.y * WAVES + wave) * ROWS;

  for (int t0 = 0; t0 < nt; t0 += TMAX) {
    const int tn = min(TMAX, nt - t0);
    __syncthreads();
    for (int t = 0; t < tn; ++t) {          // stage activations (vectorised 16 B)
      const uint4* src = (const uint4*)(A + (long)(toks[t0 + t] / top_k) * stride_am);
      uint4* dst = (uint4*)xs[t];
      for (int i = threadIdx.x; i < K / 8; i += WAVES * WS) dst[i] = src[i];
    }
    __syncthreads();
    if (e < 0) {
      if (lane == 0)
        for (int r = 0; r < ROWS; ++r)
          for (int t = 0; t < tn; ++t)
            if (n0 + r < N) C[(long)toks[t0 + t] * stride_cm + n0 + r] = 0;
      continue;
    }
    float acc[ROWS][TMAX];
#pragma unroll
    for (int r = 0; r < ROWS; ++r)
#pragma unroll
      for (int t = 0; t < TMAX; ++t) acc[r][t] = 0.f;

#pragma unroll
    for (int it = 0; it < ITERS; ++it) {
      const int byte0 = it * WS * 16 + lane * 16;      // this lane's 16 bytes
      const int k0 = byte0 * 2;                          // 32 consecutive k
      uint4 w[ROWS];
      float sc[ROWS];
#pragma unroll
      for (int r = 0; r < ROWS; ++r) {                   // issue all loads first
        const int n = min(n0 + r, N - 1);
        w[r] = *(const uint4*)(B + e * stride_be + (long)n * stride_bn + byte0);
        sc[r] = bf2f(S[e * stride_se + (long)n * stride_sn + (k0 >> 7)]);
      }
#pragma unroll
      for (int t = 0; t < TMAX; ++t) {
        if (t >= tn) break;
        float x[32];
        const uint4* xp = (const uint4*)(&xs[t][k0]);
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          uint4 v = xp[q];
          uint32_t u[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            x[q * 8 + j * 2] = __uint_as_float(u[j] << 16);
            x[q * 8 + j * 2 + 1] = __uint_as_float(u[j] & 0xFFFF0000u);
          }
        }
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
          uint32_t wb[4] = {w[r].x, w[r].y, w[r].z, w[r].w};
          float s = 0.f;
#pragma unroll
          for (int j = 0; j < 4; ++j) {
#pragma unroll
            for (int b = 0; b < 4; ++b) {
              uint32_t byte = (wb[j] >> (8 * b)) & 0xFF;
              s += (float)((int)(byte & 0xF) - 8) * x[j * 8 + b * 2];
              s += (float)((int)(byte >> 4) - 8) * x[j * 8 + b * 2 + 1];
            }
          }
          acc[r][t] += s * sc[r];
        }
      }
    }
#pragma unroll
    for (int r = 0; r < ROWS; ++r)
#pragma unroll
      for (int t = 0; t < TMAX; ++t)
#pragma unroll
        for (int off = WS / 2; off > 0; off >>= 1) acc[r][t] += __shfl_xor(acc[r][t], off, WS);
    if (lane == 0) {
      for (int t = 0; t < tn; ++t) {
        const int tok = toks[t0 + t];
        const float wgt = mul_w ? topk_w[tok] : 1.f;
        for (int r = 0; r < ROWS; ++r)
          if (n0 + r < N) C[(long)tok * stride_cm + n0 + r] = f2bf(acc[r][t] * wgt);
      }
    }
  }
}

extern "C" int vsh_moe_int4_gemv(void* stream, const void* A, const void* B, void* C, const void* S,
                                 const void* topk_w, const void* sorted_ids, const void* expert_ids,
                                 const void* ntpp, int num_m_blocks, int N, int K, int num_valid,
                                 int BM, int top_k, int mul_w, long stride_am, long stride_be,
                                 long stride_bn, long stride_cm, long stride_se, long stride_sn) {
  if (BM > 64) return -1;
  dim3 grid(num_m_blocks, (N + WAVES * ROWS - 1) / (WAVES * ROWS));
  dim3 block(WAVES * WS);
  hipStream_t s = (hipStream_t)stream;
#define LAUNCH(KK)                                                                              \
  hipLaunchKernelGGL(moe_int4_gemv_kernel<KK>, grid, block, 0, s, (const uint16_t*)A,          \
                     (const uint8_t*)B, (uint16_t*)C, (const uint16_t*)S, (const float*)topk_w, \
                     (const int32_t*)sorted_ids, (const int32_t*)expert_ids,                    \
                     (const int32_t*)ntpp, N, num_valid, BM, top_k, mul_w, stride_am, stride_be, \
                     stride_bn, stride_cm, stride_se, stride_sn)
  switch (K) {
    case 4096: LAUNCH(4096); break;
    case 2048: LAUNCH(2048); break;
    case 1024: LAUNCH(1024); break;
    default: return -2;
  }
#undef LAUNCH
  return hipGetLastError() == hipSuccess ? 0 : -3;
}

// ---- direct decode path (no moe_align) ---------------------------------------
// grid.x = one slot per (token, k) pair; the first slot holding an expert leads,
// gathers every pair routed to that expert (<= tokens) and streams the expert's
// weights once. PHASE 1: A = hidden [M][K], C = w13 out [P][N].
// PHASE 2: A = w13 out [P][2K]; staging applies silu(min(g,L)) * clamp(u,-L,L)
// (rounded to bf16 like silu_and_mul_with_clamp), C = w2 out [P][N] * topk_w.
template <int K, int PHASE>
__global__ __launch_bounds__(WAVES * WS) void moe_int4_direct_kernel(
    const uint16_t* __restrict__ A, const uint8_t* __restrict__ B, uint16_t* __restrict__ C,
    const uint16_t* __restrict__ S, const float* __restrict__ topk_w,
    const int32_t* __restrict__ ids, int P, int N, int top_k, float L,
    long stride_am, long stride_be, long stride_bn, long stride_cm,
    long stride_se, long stride_sn) {
  constexpr int KB = K / 2;
  constexpr int ITERS = KB / (WS * 16);
  static_assert(KB % (WS * 16) == 0, "K must be a multiple of 1024");
  __shared__ uint16_t xs[TMAX][K];
  __shared__ int toks[64];
  __shared__ int ntok, dup;
  const int p = blockIdx.x;
  const int e = ids[p];
  const int lane = threadIdx.x % WS, wave = threadIdx.x / WS;
  if (threadIdx.x == 0) { ntok = 0; dup = 0; }
  __syncthreads();
  if (threadIdx.x < P && ids[threadIdx.x] == e) {
    if ((int)threadIdx.x < p) dup = 1;
    else { int s = atomicAdd(&ntok, 1); toks[s] = threadIdx.x; }
  }
  __syncthreads();
  if (dup) return;
  const int nt = ntok;
  const int n0 = (blockIdx.y * WAVES + wave) * ROWS;
  const uint8_t* Be = B + (long)e * stride_be;
  const uint16_t* Se = S + (long)e * stride_se;

  for (int t0 = 0; t0 < nt; t0 += TMAX) {
    const int tn = min(TMAX, nt - t0);
    __syncthreads();
    for (int t = 0; t < tn; ++t) {
      if (PHASE == 1) {
        const uint4* src = (const uint4*)(A + (long)(toks[t0 + t] / top_k) * stride_am);
        uint4* dst = (uint4*)xs[t];
        for (int i = threadIdx.x; i < K / 8; i += WAVES * WS) dst[i] = src[i];
      } else {
        const uint16_t* src = A + (long)toks[t0 + t] * stride_am;
        for (int i = threadIdx.x; i < K; i += WAVES * WS) {
          float g = fminf(bf2f(src[i]), L);
          float u = fminf(fmaxf(bf2f(src[K + i]), -L), L);
          xs[t][i] = f2bf(g / (1.f + expf(-g)) * u);
        }
      }
    }
    __syncthreads();
    float acc[ROWS][TMAX];
#pragma unroll
    for (int r = 0; r < ROWS; ++r)
#pragma unroll
      for (int t = 0; t < TMAX; ++t) acc[r][t] = 0.f;
#pragma unroll
    for (int it = 0; it < ITERS; ++it) {
      const int byte0 = it * WS * 16 + lane * 16;
      const int k0 = byte0 * 2;
      uint4 w[ROWS];
      float sc[ROWS];
#pragma unroll
      for (int r = 0; r < ROWS; ++r) {
        const int n = min(n0 + r, N - 1);
        w[r] = *(const uint4*)(Be + (long)n * stride_bn + byte0);
        sc[r] = bf2f(Se[(long)n * stride_sn + (k0 >> 7)]);
      }
#pragma unroll
      for (int t = 0; t < TMAX; ++t) {
        if (t >= tn) break;
        float x[32];
        const uint4* xp = (const uint4*)(&xs[t][k0]);
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          uint4 v = xp[q];
          uint32_t u[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            x[q * 8 + j * 2] = __uint_as_float(u[j] << 16);
            x[q * 8 + j * 2 + 1] = __uint_as_float(u[j] & 0xFFFF0000u);
          }
        }
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
          uint32_t wb[4] = {w[r].x, w[r].y, w[r].z, w[r].w};
          float s = 0.f;
#pragma unroll
          for (int j = 0; j < 4; ++j) {
#pragma unroll
            for (int b = 0; b < 4; ++b) {
              uint32_t byte = (wb[j] >> (8 * b)) & 0xFF;
              s += (float)((int)(byte & 0xF) - 8) * x[j * 8 + b * 2];
              s += (float)((int)(byte >> 4) - 8) * x[j * 8 + b * 2 + 1];
            }
          }
          acc[r][t] += s * sc[r];
        }
      }
    }
#pragma unroll
    for (int r = 0; r < ROWS; ++r)
#pragma unroll
      for (int t = 0; t < TMAX; ++t)
#pragma unroll
        for (int off = WS / 2; off > 0; off >>= 1) acc[r][t] += __shfl_xor(acc[r][t], off, WS);
    if (lane == 0) {
      for (int t = 0; t < tn; ++t) {
        const int tok = toks[t0 + t];
        const float wgt = PHASE == 2 ? topk_w[tok] : 1.f;
        for (int r = 0; r < ROWS; ++r)
          if (n0 + r < N) C[(long)tok * stride_cm + n0 + r] = f2bf(acc[r][t] * wgt);
      }
    }
  }
}

// out[m][n] = bf16(sum_j in[m*top_k + j][n]) (fp32 accumulate, j ascending)
__global__ void moe_topk_sum_kernel(const uint16_t* __restrict__ in, uint16_t* __restrict__ out,
                                    int M, int N, int top_k, long stride_in, long stride_out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M * N) return;
  const int m = i / N, n = i % N;
  float s = 0.f;
  for (int j = 0; j < top_k; ++j) s += bf2f(in[(long)(m * top_k + j) * stride_in + n]);
  out[(long)m * stride_out + n] = f2bf(s);
}

extern "C" int vsh_moe_int4_direct(void* stream, int phase, const void* A, const void* B, void* C,
                                   const void* S, const void* topk_w, const void* ids, int P, int N,
                                   int K, int top_k, float L, long stride_am, long stride_be,
                                   long stride_bn, long stride_cm, long stride_se, long stride_sn) {
  if (P > 64 || P < 1) return -1;
  dim3 grid(P, (N + WAVES * ROWS - 1) / (WAVES * ROWS));
  dim3 block(WAVES * WS);
  hipStream_t s = (hipStream_t)stream;
#define LAUNCHD(KK, PH)                                                                          \
  hipLaunchKernelGGL((moe_int4_direct_kernel<KK, PH>), grid, block, 0, s, (const uint16_t*)A,   \
                     (const uint8_t*)B, (uint16_t*)C, (const uint16_t*)S, (const float*)topk_w,  \
                     (const int32_t*)ids, P, N, top_k, L, stride_am, stride_be, stride_bn,       \
                     stride_cm, stride_se, stride_sn)
  if (phase == 1) {
    switch (K) {
      case 4096: LAUNCHD(4096, 1); break;
      case 2048: LAUNCHD(2048, 1); break;
      case 1024: LAUNCHD(1024, 1); break;
      default: return -2;
    }
  } else {
    switch (K) {
      case 2048: LAUNCHD(2048, 2); break;
      case 1024: LAUNCHD(1024, 2); break;
      default: return -2;
    }
  }
#undef LAUNCHD
  return hipGetLastError() == hipSuccess ? 0 : -3;
}

extern "C" int vsh_moe_topk_sum(void* stream, const void* in, void* out, int M, int N, int top_k,
                                long stride_in, long stride_out) {
  const int tot = M * N;
  hipLaunchKernelGGL(moe_topk_sum_kernel, dim3((tot + 255) / 256), dim3(256), 0, (hipStream_t)stream,
                     (const uint16_t*)in, (uint16_t*)out, M, N, top_k, stride_in, stride_out);
  return hipGetLastError() == hipSuccess ? 0 : -3;
}
