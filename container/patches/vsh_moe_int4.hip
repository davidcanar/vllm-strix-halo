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
    long stride_se, long stride_sn, int E) {
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
  if (e < 0 || e >= E) {   // padding sentinel (-1) / out of range: contributes nothing
    for (int i = threadIdx.x; i < nt * WAVES * ROWS; i += WAVES * WS) {
      const int n = blockIdx.y * WAVES * ROWS + i % (WAVES * ROWS);
      if (n < N) C[(long)toks[i / (WAVES * ROWS)] * stride_cm + n] = 0;
    }
    return;
  }
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
                                   long stride_bn, long stride_cm, long stride_se, long stride_sn, int E) {
  if (P > 64 || P < 1) return -1;
  dim3 grid(P, (N + WAVES * ROWS - 1) / (WAVES * ROWS));
  dim3 block(WAVES * WS);
  hipStream_t s = (hipStream_t)stream;
#define LAUNCHD(KK, PH)                                                                          \
  hipLaunchKernelGGL((moe_int4_direct_kernel<KK, PH>), grid, block, 0, s, (const uint16_t*)A,   \
                     (const uint8_t*)B, (uint16_t*)C, (const uint16_t*)S, (const float*)topk_w,  \
                     (const int32_t*)ids, P, N, top_k, L, stride_am, stride_be, stride_bn,       \
                     stride_cm, stride_se, stride_sn, E)
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


// ---- MXFP4 direct decode path (DeepSeek-V4 routed experts, gfx1151) ----------
// Same structure as moe_int4_direct_kernel (expert dedup per slot, clamp-SwiGLU
// fused into phase-2 staging, router weight in phase 2), for OCP MXFP4 weights
// as vLLM 0.31 / triton_kernels 3.8 lay them out on gfx1151:
//   values uint8 [E][N][K/2] (K contiguous, low nibble = even k, e2m1),
//   scales uint8 e8m0, one per 32 k, arbitrary strides (0.31: [E][K/32][N]).
// Each lane's 16 bytes are exactly one 32-value scale group. e2m1 magnitudes
// are decoded as 2x{0,.5,1,1.5,2,3,4,6} from a packed nibble table and the 1/2
// folded into the scale (2^(s-128)).
__device__ __forceinline__ float e2m1x2(uint32_t nib) {
  const float v = (float)((0xC8643210u >> ((nib & 7u) << 2)) & 0xFu);
  return (nib & 8u) ? -v : v;
}

template <int K, int PHASE>
__global__ __launch_bounds__(WAVES * WS) void moe_mxfp4_direct_kernel(
    const uint16_t* __restrict__ A, const uint8_t* __restrict__ B, uint16_t* __restrict__ C,
    const uint8_t* __restrict__ S, const float* __restrict__ topk_w,
    const int32_t* __restrict__ ids, int P, int N, int top_k, float L,
    long stride_am, long stride_be, long stride_bn, long stride_cm,
    long sse, long ssn, long ssg, int E) {
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
  if (e < 0 || e >= E) {   // padding sentinel (-1) / out of range: contributes nothing
    for (int i = threadIdx.x; i < nt * WAVES * ROWS; i += WAVES * WS) {
      const int n = blockIdx.y * WAVES * ROWS + i % (WAVES * ROWS);
      if (n < N) C[(long)toks[i / (WAVES * ROWS)] * stride_cm + n] = 0;
    }
    return;
  }
  const int n0 = (blockIdx.y * WAVES + wave) * ROWS;
  const uint8_t* Be = B + (long)e * stride_be;
  const uint8_t* Se = S + (long)e * sse;

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
        const uint32_t sb = Se[(long)n * ssn + (long)(k0 >> 5) * ssg];
        sc[r] = sb ? __uint_as_float((sb - 1u) << 23) : 0.f;   // 2^(sb-127) / 2
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
              s += e2m1x2(byte & 0xF) * x[j * 8 + b * 2];
              s += e2m1x2(byte >> 4) * x[j * 8 + b * 2 + 1];
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

extern "C" int vsh_moe_mxfp4_direct(void* stream, int phase, const void* A, const void* B, void* C,
                                    const void* S, const void* topk_w, const void* ids, int P, int N,
                                    int K, int top_k, float L, long stride_am, long stride_be,
                                    long stride_bn, long stride_cm, long sse, long ssn, long ssg, int E) {
  if (P > 64 || P < 1) return -1;
  dim3 grid(P, (N + WAVES * ROWS - 1) / (WAVES * ROWS));
  dim3 block(WAVES * WS);
  hipStream_t s = (hipStream_t)stream;
#define LAUNCHX(KK, PH)                                                                          \
  hipLaunchKernelGGL((moe_mxfp4_direct_kernel<KK, PH>), grid, block, 0, s, (const uint16_t*)A,  \
                     (const uint8_t*)B, (uint16_t*)C, (const uint8_t*)S, (const float*)topk_w,   \
                     (const int32_t*)ids, P, N, top_k, L, stride_am, stride_be, stride_bn,       \
                     stride_cm, sse, ssn, ssg, E)
  if (phase == 1) {
    switch (K) {
      case 4096: LAUNCHX(4096, 1); break;
      case 2048: LAUNCHX(2048, 1); break;
      case 1024: LAUNCHX(1024, 1); break;
      default: return -2;
    }
  } else {
    switch (K) {
      case 2048: LAUNCHX(2048, 2); break;
      case 1024: LAUNCHX(1024, 2); break;
      default: return -2;
    }
  }
#undef LAUNCHX
  return hipGetLastError() == hipSuccess ? 0 : -3;
}


// ---- MXFP4 direct path v2: no LDS staging, all of an expert's tokens per pass ----
// v1 staged activations in 33 KB of LDS (one workgroup per CU) and took at most
// TMAX=4 tokens per pass, so an expert shared by 5-8 draft tokens was streamed
// twice. v2 reads the activation slice straight from L2 (as the FP8 GEMV does),
// keeps up to MT tokens in registers, decodes each weight once per row and
// reuses it for every token. The SwiGLU runs once in its own small kernel.
template <int K, int PHASE, int MT, int R>
__global__ __launch_bounds__(WAVES * WS) void moe_mxfp4_v2_kernel(
    const uint16_t* __restrict__ A, const uint8_t* __restrict__ B, uint16_t* __restrict__ C,
    const uint8_t* __restrict__ S, const float* __restrict__ topk_w,
    const int32_t* __restrict__ ids, int P, int N, int top_k,
    long stride_am, long stride_be, long stride_bn, long stride_cm, long sse, long ssn, long ssg, int E) {
  constexpr int KB = K / 2;
  constexpr int ITERS = KB / (WS * 16);
  static_assert(KB % (WS * 16) == 0, "K must be a multiple of 1024");
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
  if (e < 0 || e >= E) {   // padding sentinel (-1) / out of range: contributes nothing
    for (int i = threadIdx.x; i < nt * WAVES * R; i += WAVES * WS) {
      const int n = blockIdx.y * WAVES * R + i % (WAVES * R);
      if (n < N) C[(long)toks[i / (WAVES * R)] * stride_cm + n] = 0;
    }
    return;
  }
  const int n0 = (blockIdx.y * WAVES + wave) * R;
  if (n0 >= N) return;
  const uint8_t* Be = B + (long)e * stride_be;
  const uint8_t* Se = S + (long)e * sse;
  for (int t0 = 0; t0 < nt; t0 += MT) {
    const int tn = min(MT, nt - t0);
    long arow[MT];
#pragma unroll
    for (int t = 0; t < MT; ++t) {
      const int tok = toks[t0 + min(t, tn - 1)];
      arow[t] = (long)(PHASE == 1 ? tok / top_k : tok) * stride_am;
    }
    float acc[R][MT];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
      for (int t = 0; t < MT; ++t) acc[r][t] = 0.f;
#pragma unroll
    for (int it = 0; it < ITERS; ++it) {
      const int byte0 = it * WS * 16 + lane * 16;
      const int k0 = byte0 * 2;
      uint4 w[R];
      float sc[R];
#pragma unroll
      for (int r = 0; r < R; ++r) {
        const int n = min(n0 + r, N - 1);
        w[r] = *(const uint4*)(Be + (long)n * stride_bn + byte0);
        const uint32_t sb = Se[(long)n * ssn + (long)(k0 >> 5) * ssg];
        sc[r] = sb ? __uint_as_float((sb - 1u) << 23) : 0.f;   // 2^(sb-127) / 2
      }
      float wf[R][32];
#pragma unroll
      for (int r = 0; r < R; ++r) {
        const uint32_t wb[4] = {w[r].x, w[r].y, w[r].z, w[r].w};
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
          for (int b = 0; b < 4; ++b) {
            const uint32_t byte = (wb[j] >> (8 * b)) & 0xFF;
            wf[r][j * 8 + b * 2] = e2m1x2(byte & 0xF) * sc[r];
            wf[r][j * 8 + b * 2 + 1] = e2m1x2(byte >> 4) * sc[r];
          }
      }
#pragma unroll
      for (int t = 0; t < MT; ++t) {
        if (t >= tn) break;
        const uint4* xp = (const uint4*)(A + arow[t] + k0);
        float s[R];
#pragma unroll
        for (int r = 0; r < R; ++r) s[r] = 0.f;
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const uint4 v = xp[q];
          const uint32_t u[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            const float x0 = __uint_as_float(u[j] << 16), x1 = __uint_as_float(u[j] & 0xFFFF0000u);
#pragma unroll
            for (int r = 0; r < R; ++r) s[r] += wf[r][q * 8 + j * 2] * x0 + wf[r][q * 8 + j * 2 + 1] * x1;
          }
        }
#pragma unroll
        for (int r = 0; r < R; ++r) acc[r][t] += s[r];
      }
    }
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
      for (int t = 0; t < MT; ++t)
#pragma unroll
        for (int off = WS / 2; off > 0; off >>= 1) acc[r][t] += __shfl_xor(acc[r][t], off, WS);
    if (lane == 0) {
      for (int t = 0; t < tn; ++t) {
        const int tok = toks[t0 + t];
        const float wgt = PHASE == 2 ? topk_w[tok] : 1.f;
        for (int r = 0; r < R; ++r)
          if (n0 + r < N) C[(long)tok * stride_cm + n0 + r] = f2bf(acc[r][t] * wgt);
      }
    }
  }
}

// act[p][i] = bf16(silu(min(g, L)) * clamp(u, -L, L)), g/u = gate/up halves of c1[p]
__global__ void moe_swiglu_kernel(const uint16_t* __restrict__ c1, uint16_t* __restrict__ act,
                                  int P, int I, float L, long s1, long sa) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= P * I) return;
  const int p = i / I, j = i % I;
  const float g = fminf(bf2f(c1[(long)p * s1 + j]), L);
  const float u = fminf(fmaxf(bf2f(c1[(long)p * s1 + I + j]), -L), L);
  act[(long)p * sa + j] = f2bf(g / (1.f + expf(-g)) * u);
}

extern "C" int vsh_moe_mxfp4_v2(void* stream, int phase, const void* A, const void* B, void* C,
                                const void* S, const void* topk_w, const void* ids, int P, int N,
                                int K, int top_k, long stride_am, long stride_be, long stride_bn,
                                long stride_cm, long sse, long ssn, long ssg, int E) {
  if (P > 64 || P < 1) return -1;
  constexpr int R = 2;
  dim3 grid(P, (N + WAVES * R - 1) / (WAVES * R));
  dim3 block(WAVES * WS);
  hipStream_t s = (hipStream_t)stream;
#define LV2(KK, PH) hipLaunchKernelGGL((moe_mxfp4_v2_kernel<KK, PH, 8, R>), grid, block, 0, s,    \
      (const uint16_t*)A, (const uint8_t*)B, (uint16_t*)C, (const uint8_t*)S, (const float*)topk_w, \
      (const int32_t*)ids, P, N, top_k, stride_am, stride_be, stride_bn, stride_cm, sse, ssn, ssg, E)
  if (phase == 1) {
    switch (K) { case 4096: LV2(4096, 1); break; case 2048: LV2(2048, 1); break;
                 case 1024: LV2(1024, 1); break; default: return -2; }
  } else {
    switch (K) { case 2048: LV2(2048, 2); break; case 1024: LV2(1024, 2); break; default: return -2; }
  }
#undef LV2
  return hipGetLastError() == hipSuccess ? 0 : -3;
}

extern "C" int vsh_moe_swiglu(void* stream, const void* c1, void* act, int P, int I, float L,
                              long s1, long sa) {
  const int tot = P * I;
  hipLaunchKernelGGL(moe_swiglu_kernel, dim3((tot + 255) / 256), dim3(256), 0, (hipStream_t)stream,
                     (const uint16_t*)c1, (uint16_t*)act, P, I, L, s1, sa);
  return hipGetLastError() == hipSuccess ? 0 : -3;
}


// ---- MXFP4 direct path v3: v1's LDS staging + decode-once + token tile sized by M ----
// v1 was ALU-bound on experts shared by several draft tokens (it re-decoded every
// e2m1 value per token) and re-streamed an expert's weights for tokens 5-8; v2
// (activations from L2) drowned in activation traffic. v3 stages the activation
// rows of the expert's tokens in LDS (MT rows, MT chosen per launch from M so a
// 1-token decode uses 8 KB, not 32), decodes each weight once per row and reuses
// it for all staged tokens.
template <int K, int MT, int R>
__global__ __launch_bounds__(WAVES * WS) void moe_mxfp4_v3_kernel(
    const uint16_t* __restrict__ A, const uint8_t* __restrict__ B, uint16_t* __restrict__ C,
    const uint8_t* __restrict__ S, const float* __restrict__ topk_w,
    const int32_t* __restrict__ ids, int P, int N, int top_k, int phase,
    long stride_am, long stride_be, long stride_bn, long stride_cm, long sse, long ssn, long ssg, int E) {
  constexpr int KB = K / 2;
  constexpr int ITERS = KB / (WS * 16);
  static_assert(KB % (WS * 16) == 0, "K must be a multiple of 1024");
  __shared__ uint16_t xs[MT][K];
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
  if (e < 0 || e >= E) {   // padding sentinel (-1) / out of range: contributes nothing
    for (int i = threadIdx.x; i < nt * WAVES * R; i += WAVES * WS) {
      const int n = blockIdx.y * WAVES * R + i % (WAVES * R);
      if (n < N) C[(long)toks[i / (WAVES * R)] * stride_cm + n] = 0;
    }
    return;
  }
  const int n0 = (blockIdx.y * WAVES + wave) * R;
  const uint8_t* Be = B + (long)e * stride_be;
  const uint8_t* Se = S + (long)e * sse;
  for (int t0 = 0; t0 < nt; t0 += MT) {
    const int tn = min(MT, nt - t0);
    __syncthreads();
    for (int t = 0; t < tn; ++t) {
      const int tok = toks[t0 + t];
      const uint4* src = (const uint4*)(A + (long)(phase == 1 ? tok / top_k : tok) * stride_am);
      uint4* dst = (uint4*)xs[t];
      for (int i = threadIdx.x; i < K / 8; i += WAVES * WS) dst[i] = src[i];
    }
    __syncthreads();
    if (n0 >= N) continue;
    float acc[R][MT];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
      for (int t = 0; t < MT; ++t) acc[r][t] = 0.f;
#pragma unroll
    for (int it = 0; it < ITERS; ++it) {
      const int byte0 = it * WS * 16 + lane * 16;
      const int k0 = byte0 * 2;
      uint4 w[R];
      float sc[R];
#pragma unroll
      for (int r = 0; r < R; ++r) {
        const int n = min(n0 + r, N - 1);
        w[r] = *(const uint4*)(Be + (long)n * stride_bn + byte0);
        const uint32_t sb = Se[(long)n * ssn + (long)(k0 >> 5) * ssg];
        sc[r] = sb ? __uint_as_float((sb - 1u) << 23) : 0.f;   // 2^(sb-127) / 2
      }
      float wf[R][32];
#pragma unroll
      for (int r = 0; r < R; ++r) {
        const uint32_t wb[4] = {w[r].x, w[r].y, w[r].z, w[r].w};
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
          for (int b = 0; b < 4; ++b) {
            const uint32_t byte = (wb[j] >> (8 * b)) & 0xFF;
            wf[r][j * 8 + b * 2] = e2m1x2(byte & 0xF);
            wf[r][j * 8 + b * 2 + 1] = e2m1x2(byte >> 4);
          }
      }
#pragma unroll
      for (int t = 0; t < MT; ++t) {
        if (t >= tn) break;
        const uint4* xp = (const uint4*)(&xs[t][k0]);
        float s[R];
#pragma unroll
        for (int r = 0; r < R; ++r) s[r] = 0.f;
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const uint4 v = xp[q];
          const uint32_t u[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            const float x0 = __uint_as_float(u[j] << 16), x1 = __uint_as_float(u[j] & 0xFFFF0000u);
#pragma unroll
            for (int r = 0; r < R; ++r) s[r] += wf[r][q * 8 + j * 2] * x0 + wf[r][q * 8 + j * 2 + 1] * x1;
          }
        }
#pragma unroll
        for (int r = 0; r < R; ++r) acc[r][t] += s[r] * sc[r];
      }
    }
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
      for (int t = 0; t < MT; ++t)
#pragma unroll
        for (int off = WS / 2; off > 0; off >>= 1) acc[r][t] += __shfl_xor(acc[r][t], off, WS);
    if (lane == 0) {
      for (int t = 0; t < tn; ++t) {
        const int tok = toks[t0 + t];
        const float wgt = phase == 2 ? topk_w[tok] : 1.f;
        for (int r = 0; r < R; ++r)
          if (n0 + r < N) C[(long)tok * stride_cm + n0 + r] = f2bf(acc[r][t] * wgt);
      }
    }
  }
}

template <int K, int MT, int R>
static int mx3_launch(hipStream_t s, int phase, const void* A, const void* B, void* C, const void* S,
                      const void* topk_w, const void* ids, int P, int N, int top_k, long stride_am,
                      long stride_be, long stride_bn, long stride_cm, long sse, long ssn, long ssg, int E) {
  dim3 grid(P, (N + WAVES * R - 1) / (WAVES * R)), block(WAVES * WS);
  hipLaunchKernelGGL((moe_mxfp4_v3_kernel<K, MT, R>), grid, block, 0, s, (const uint16_t*)A,
                     (const uint8_t*)B, (uint16_t*)C, (const uint8_t*)S, (const float*)topk_w,
                     (const int32_t*)ids, P, N, top_k, phase, stride_am, stride_be, stride_bn,
                     stride_cm, sse, ssn, ssg, E);
  return 0;
}

// mt = tokens staged per pass (1, 2, 4 or 6; host picks min(M, 6) rounded up)
extern "C" int vsh_moe_mxfp4_v3(void* stream, int phase, int mt, const void* A, const void* B,
                                void* C, const void* S, const void* topk_w, const void* ids, int P,
                                int N, int K, int top_k, long stride_am, long stride_be,
                                long stride_bn, long stride_cm, long sse, long ssn, long ssg, int E) {
  if (P > 64 || P < 1) return -1;
  hipStream_t s = (hipStream_t)stream;
  constexpr int R = 2;
#define ARGS s, phase, A, B, C, S, topk_w, ids, P, N, top_k, stride_am, stride_be, stride_bn, stride_cm, sse, ssn, ssg, E
#define BYMT(KK) do { if (mt <= 1) mx3_launch<KK, 1, R>(ARGS); else if (mt <= 2) mx3_launch<KK, 2, R>(ARGS); \
                      else if (mt <= 4) mx3_launch<KK, 4, R>(ARGS); else mx3_launch<KK, 6, R>(ARGS); } while (0)
  switch (K) {
    case 4096: BYMT(4096); break;
    case 2048: BYMT(2048); break;
    case 1024: BYMT(1024); break;
    default: return -2;
  }
#undef BYMT
#undef ARGS
  return hipGetLastError() == hipSuccess ? 0 : -3;
}


// ---- MXFP4 direct path v4: v3 with RP row-pair passes per wave ----------------
// In v3 a workgroup covers only WAVES*R = 16 rows of one expert: in w2 (K=1024)
// a wave streams 1 KB of weights and exits, so the dedup prologue, the LDS
// staging and the barriers dominate when the draft tokens hit many distinct
// experts. v4 keeps the staged activations and walks RP row pairs per wave.
template <int K, int MT, int R, int RP>
__global__ __launch_bounds__(WAVES * WS) void moe_mxfp4_v4_kernel(
    const uint16_t* __restrict__ A, const uint8_t* __restrict__ B, uint16_t* __restrict__ C,
    const uint8_t* __restrict__ S, const float* __restrict__ topk_w,
    const int32_t* __restrict__ ids, int P, int N, int top_k, int phase,
    long stride_am, long stride_be, long stride_bn, long stride_cm, long sse, long ssn, long ssg,
    int E) {
  constexpr int KB = K / 2;
  constexpr int ITERS = KB / (WS * 16);
  constexpr int TILE = WAVES * R * RP;
  static_assert(KB % (WS * 16) == 0, "K must be a multiple of 1024");
  __shared__ uint16_t xs[MT][K];
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
  if (e < 0 || e >= E) {   // padding sentinel (-1) / out of range: contributes nothing
    for (int i = threadIdx.x; i < nt * TILE; i += WAVES * WS) {
      const int n = blockIdx.y * TILE + i % TILE;
      if (n < N) C[(long)toks[i / TILE] * stride_cm + n] = 0;
    }
    return;
  }
  const uint8_t* Be = B + (long)e * stride_be;
  const uint8_t* Se = S + (long)e * sse;
  for (int t0 = 0; t0 < nt; t0 += MT) {
    const int tn = min(MT, nt - t0);
    __syncthreads();
    for (int t = 0; t < tn; ++t) {
      const int tok = toks[t0 + t];
      const uint4* src = (const uint4*)(A + (long)(phase == 1 ? tok / top_k : tok) * stride_am);
      uint4* dst = (uint4*)xs[t];
      for (int i = threadIdx.x; i < K / 8; i += WAVES * WS) dst[i] = src[i];
    }
    __syncthreads();
    for (int rp = 0; rp < RP; ++rp) {
      const int n0 = blockIdx.y * TILE + (rp * WAVES + wave) * R;
      if (n0 >= N) break;
      float acc[R][MT];
#pragma unroll
      for (int r = 0; r < R; ++r)
#pragma unroll
        for (int t = 0; t < MT; ++t) acc[r][t] = 0.f;
#pragma unroll
      for (int it = 0; it < ITERS; ++it) {
        const int byte0 = it * WS * 16 + lane * 16;
        const int k0 = byte0 * 2;
        uint4 w[R];
        float sc[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
          const int n = min(n0 + r, N - 1);
          w[r] = *(const uint4*)(Be + (long)n * stride_bn + byte0);
          const uint32_t sb = Se[(long)n * ssn + (long)(k0 >> 5) * ssg];
          sc[r] = sb ? __uint_as_float((sb - 1u) << 23) : 0.f;   // 2^(sb-127) / 2
        }
        float wf[R][32];
#pragma unroll
        for (int r = 0; r < R; ++r) {
          const uint32_t wb[4] = {w[r].x, w[r].y, w[r].z, w[r].w};
#pragma unroll
          for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int b = 0; b < 4; ++b) {
              const uint32_t byte = (wb[j] >> (8 * b)) & 0xFF;
              wf[r][j * 8 + b * 2] = e2m1x2(byte & 0xF);
              wf[r][j * 8 + b * 2 + 1] = e2m1x2(byte >> 4);
            }
        }
#pragma unroll
        for (int t = 0; t < MT; ++t) {
          if (t >= tn) break;
          const uint4* xp = (const uint4*)(&xs[t][k0]);
          float s[R];
#pragma unroll
          for (int r = 0; r < R; ++r) s[r] = 0.f;
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            const uint4 v = xp[q];
            const uint32_t u[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
            for (int j = 0; j < 4; ++j) {
              const float x0 = __uint_as_float(u[j] << 16), x1 = __uint_as_float(u[j] & 0xFFFF0000u);
#pragma unroll
              for (int r = 0; r < R; ++r) s[r] += wf[r][q * 8 + j * 2] * x0 + wf[r][q * 8 + j * 2 + 1] * x1;
            }
          }
#pragma unroll
          for (int r = 0; r < R; ++r) acc[r][t] += s[r] * sc[r];
        }
      }
#pragma unroll
      for (int r = 0; r < R; ++r)
#pragma unroll
        for (int t = 0; t < MT; ++t)
#pragma unroll
          for (int off = WS / 2; off > 0; off >>= 1) acc[r][t] += __shfl_xor(acc[r][t], off, WS);
      if (lane == 0) {
        for (int t = 0; t < tn; ++t) {
          const int tok = toks[t0 + t];
          const float wgt = phase == 2 ? topk_w[tok] : 1.f;
          for (int r = 0; r < R; ++r)
            if (n0 + r < N) C[(long)tok * stride_cm + n0 + r] = f2bf(acc[r][t] * wgt);
        }
      }
    }
  }
}

template <int K, int MT, int R, int RP>
static int mx4_launch(hipStream_t s, int phase, const void* A, const void* B, void* C, const void* S,
                      const void* topk_w, const void* ids, int P, int N, int top_k, long stride_am,
                      long stride_be, long stride_bn, long stride_cm, long sse, long ssn, long ssg, int E) {
  dim3 grid(P, (N + WAVES * R * RP - 1) / (WAVES * R * RP)), block(WAVES * WS);
  hipLaunchKernelGGL((moe_mxfp4_v4_kernel<K, MT, R, RP>), grid, block, 0, s, (const uint16_t*)A,
                     (const uint8_t*)B, (uint16_t*)C, (const uint8_t*)S, (const float*)topk_w,
                     (const int32_t*)ids, P, N, top_k, phase, stride_am, stride_be, stride_bn,
                     stride_cm, sse, ssn, ssg, E);
  return 0;
}

// rp = row-pair passes per wave (1, 2, 4); mt as for v3
extern "C" int vsh_moe_mxfp4_v4(void* stream, int phase, int mt, int rp, const void* A, const void* B,
                                void* C, const void* S, const void* topk_w, const void* ids, int P,
                                int N, int K, int top_k, long stride_am, long stride_be,
                                long stride_bn, long stride_cm, long sse, long ssn, long ssg, int E) {
  if (P > 64 || P < 1) return -1;
  hipStream_t s = (hipStream_t)stream;
#define ARGS4 s, phase, A, B, C, S, topk_w, ids, P, N, top_k, stride_am, stride_be, stride_bn, stride_cm, sse, ssn, ssg, E
#define BYRP(KK, MM) do { if (rp <= 1) mx4_launch<KK, MM, 2, 1>(ARGS4); else if (rp <= 2) mx4_launch<KK, MM, 2, 2>(ARGS4); \
                          else mx4_launch<KK, MM, 2, 4>(ARGS4); } while (0)
#define BYMT4(KK) do { if (mt <= 1) BYRP(KK, 1); else if (mt <= 2) BYRP(KK, 2); else BYRP(KK, 4); } while (0)
  switch (K) {
    case 4096: BYMT4(4096); break;
    case 2048: BYMT4(2048); break;
    case 1024: BYMT4(1024); break;
    default: return -2;
  }
#undef BYMT4
#undef BYRP
#undef ARGS4
  return hipGetLastError() == hipSuccess ? 0 : -3;
}


// ---- MXFP4 direct path v5 --------------------------------------------------------
// v4 at M=6 was LDS-bound (32 KB of staged activations per block -> 6 waves/SIMD)
// and spent ~7 ALU ops per e2m1 value on the decode, at the 2100 MHz SCLK cap. v5:
//  * activations staged per 1024-value K chunk, as fp16 (2 KB per token per chunk);
//  * each wave owns R consecutive rows: one 2- or 4-byte scale load per lane;
//  * e2m1 -> fp16 with one v_perm_b32 per value pair: the fp16 low byte of every e2m1
//    value is 0, so an 8-entry byte LUT {00 38 3C 3E 40 42 44 46} gives the high byte;
//    the sign bit is OR-ed in, and v_dot2_f32_f16 multiplies against fp16 activations.
// Pairing: one fp16x2 word holds values (j, j+4) of an 8-value nibble word, so the
// staged activations use the same (j, j+4) order. Values are exact (e2m1 and bf16
// both fit fp16; |activation| > 65504 saturates in cvt_pkrtz).
typedef _Float16 vsh_half2 __attribute__((ext_vector_type(2)));

__device__ __forceinline__ uint32_t mx5_pair(uint32_t sel, uint32_t sign) {
  return __builtin_amdgcn_perm(0x46444240u, 0x3E3C3800u, sel | 0x000C000Cu) | sign;
}

__device__ __forceinline__ void mx5_decode8(uint32_t w, uint32_t p[4]) {
  p[0] = mx5_pair((w & 0x00070007u) << 8, (w & 0x00080008u) << 12);   // (v0, v4)
  p[1] = mx5_pair((w & 0x00700070u) << 4, (w & 0x00800080u) << 8);    // (v1, v5)
  p[2] = mx5_pair(w & 0x07000700u, (w & 0x08000800u) << 4);           // (v2, v6)
  p[3] = mx5_pair((w >> 4) & 0x07000700u, w & 0x80008000u);           // (v3, v7)
}

__device__ __forceinline__ float mx5_dot2(uint32_t a, uint32_t b, float c) {
  return __builtin_amdgcn_fdot2(__builtin_bit_cast(vsh_half2, a), __builtin_bit_cast(vsh_half2, b), c, false);
}

__device__ __forceinline__ uint32_t mx5_h2(float a, float b) {
  return __builtin_bit_cast(uint32_t, __builtin_amdgcn_cvt_pkrtz(a, b));
}

// 8 bf16 (x0..x7) -> fp16 pairs (x0,x4) (x1,x5) (x2,x6) (x3,x7)
__device__ __forceinline__ uint4 mx5_stage8(uint4 v) {
  const float x0 = __uint_as_float(v.x << 16), x1 = __uint_as_float(v.x & 0xFFFF0000u);
  const float x2 = __uint_as_float(v.y << 16), x3 = __uint_as_float(v.y & 0xFFFF0000u);
  const float x4 = __uint_as_float(v.z << 16), x5 = __uint_as_float(v.z & 0xFFFF0000u);
  const float x6 = __uint_as_float(v.w << 16), x7 = __uint_as_float(v.w & 0xFFFF0000u);
  return make_uint4(mx5_h2(x0, x4), mx5_h2(x1, x5), mx5_h2(x2, x6), mx5_h2(x3, x7));
}

template <int K, int MT, int R, int RP>
__global__ __launch_bounds__(WAVES * WS) void moe_mxfp4_v5_kernel(
    const uint16_t* __restrict__ A, const uint8_t* __restrict__ B, uint16_t* __restrict__ C,
    const uint8_t* __restrict__ S, const float* __restrict__ topk_w,
    const int32_t* __restrict__ ids, int P, int N, int top_k, int phase,
    long stride_am, long stride_be, long stride_bn, long stride_cm, long sse, long ssn, long ssg,
    int E, int vec_scale, int tile_major) {
  constexpr int CV = WS * 32;               // values per K chunk: 32 per lane = one scale group
  constexpr int NCH = K / CV;
  constexpr int TILE = WAVES * R * RP;
  static_assert(K % CV == 0, "K must be a multiple of 1024");
  static_assert(R == 1 || R == 2 || R == 4, "R must be 1, 2 or 4");
  __shared__ uint4 xs[MT][CV / 8];          // fp16 pairs, 2 KB per token
  __shared__ int toks[64];
  __shared__ int ntok, dup;
  const int p = tile_major ? blockIdx.y : blockIdx.x;
  const int ty = tile_major ? blockIdx.x : blockIdx.y;
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
  if (e < 0 || e >= E) {   // padding sentinel (-1) / out of range: contributes nothing
    for (int i = threadIdx.x; i < nt * TILE; i += WAVES * WS) {
      const int n = ty * TILE + i % TILE;
      if (n < N) C[(long)toks[i / TILE] * stride_cm + n] = 0;
    }
    return;
  }
  const uint8_t* Be = B + (long)e * stride_be;
  const uint8_t* Se = S + (long)e * sse;
  for (int t0 = 0; t0 < nt; t0 += MT) {
    const int tn = min(MT, nt - t0);
    float acc[RP][R][MT];
#pragma unroll
    for (int rp = 0; rp < RP; ++rp)
#pragma unroll
      for (int r = 0; r < R; ++r)
#pragma unroll
        for (int t = 0; t < MT; ++t) acc[rp][r][t] = 0.f;
    for (int c = 0; c < NCH; ++c) {
      __syncthreads();
      for (int i = threadIdx.x; i < tn * (CV / 8); i += WAVES * WS) {
        const int t = i / (CV / 8), q = i % (CV / 8);
        const int tok = toks[t0 + t];
        const uint4* src = (const uint4*)(A + (long)(phase == 1 ? tok / top_k : tok) * stride_am + (long)c * CV);
        xs[t][q] = mx5_stage8(src[q]);
      }
      __syncthreads();
      const long g = (long)c * WS + lane;   // this lane's scale group
#pragma unroll
      for (int rp = 0; rp < RP; ++rp) {
        const int n0 = ty * TILE + (rp * WAVES + wave) * R;
        if (n0 >= N) break;
        uint4 w[R];
        uint32_t sb[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
          const int n = min(n0 + r, N - 1);
          w[r] = *(const uint4*)(Be + (long)n * stride_bn + (long)c * (CV / 2) + lane * 16);
        }
        if (R > 1 && vec_scale && n0 + R <= N) {
          const uint8_t* sp = Se + g * ssg + n0;
          const uint32_t sv = R == 4 ? *(const uint32_t*)sp : (uint32_t)*(const uint16_t*)sp;
#pragma unroll
          for (int r = 0; r < R; ++r) sb[r] = (sv >> (8 * r)) & 0xFFu;
        } else {
#pragma unroll
          for (int r = 0; r < R; ++r) sb[r] = Se[(long)min(n0 + r, N - 1) * ssn + g * ssg];
        }
        float s[R][MT];
#pragma unroll
        for (int r = 0; r < R; ++r)
#pragma unroll
          for (int t = 0; t < MT; ++t) s[r][t] = 0.f;
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          uint32_t pw[R][4];
#pragma unroll
          for (int r = 0; r < R; ++r) {
            const uint32_t word = q == 0 ? w[r].x : q == 1 ? w[r].y : q == 2 ? w[r].z : w[r].w;
            mx5_decode8(word, pw[r]);
          }
#pragma unroll
          for (int t = 0; t < MT; ++t) {
            if (t < tn) {
              const uint4 xa = xs[t][lane * 4 + q];
#pragma unroll
              for (int r = 0; r < R; ++r) {
                float v = s[r][t];
                v = mx5_dot2(pw[r][0], xa.x, v);
                v = mx5_dot2(pw[r][1], xa.y, v);
                v = mx5_dot2(pw[r][2], xa.z, v);
                v = mx5_dot2(pw[r][3], xa.w, v);
                s[r][t] = v;
              }
            }
          }
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
          const float sc = sb[r] ? __uint_as_float(sb[r] << 23) : 0.f;   // 2^(sb-127)
#pragma unroll
          for (int t = 0; t < MT; ++t) acc[rp][r][t] += s[r][t] * sc;
        }
      }
    }
#pragma unroll
    for (int rp = 0; rp < RP; ++rp) {
      const int n0 = ty * TILE + (rp * WAVES + wave) * R;
#pragma unroll
      for (int r = 0; r < R; ++r)
#pragma unroll
        for (int t = 0; t < MT; ++t) {
          if (t < tn) {
            float v = acc[rp][r][t];
#pragma unroll
            for (int off = WS / 2; off > 0; off >>= 1) v += __shfl_xor(v, off, WS);
            if (lane == 0 && n0 + r < N) {
              const int tok = toks[t0 + t];
              const float wgt = phase == 2 ? topk_w[tok] : 1.f;
              C[(long)tok * stride_cm + n0 + r] = f2bf(v * wgt);
            }
          }
        }
    }
  }
}

template <int K, int MT, int R, int RP>
static int mx5_launch(hipStream_t s, int phase, const void* A, const void* B, void* C, const void* S,
                      const void* topk_w, const void* ids, int P, int N, int top_k, long stride_am,
                      long stride_be, long stride_bn, long stride_cm, long sse, long ssn, long ssg, int E,
                      int vec, int tile_major) {
  const int tiles = (N + WAVES * R * RP - 1) / (WAVES * R * RP);
  dim3 grid = tile_major ? dim3(tiles, P) : dim3(P, tiles), block(WAVES * WS);
  hipLaunchKernelGGL((moe_mxfp4_v5_kernel<K, MT, R, RP>), grid, block, 0, s, (const uint16_t*)A,
                     (const uint8_t*)B, (uint16_t*)C, (const uint8_t*)S, (const float*)topk_w,
                     (const int32_t*)ids, P, N, top_k, phase, stride_am, stride_be, stride_bn,
                     stride_cm, sse, ssn, ssg, E, vec, tile_major);
  return 0;
}

// mt: token tile (1, 2, 4, 8); rc: rows per wave R x passes RP as 10*R + RP (11, 21, 22, 41, 42, 44),
// + 100 for tile-major grid order (the N tiles of one expert in consecutive blocks)
extern "C" int vsh_moe_mxfp4_v5(void* stream, int phase, int mt, int rc, const void* A, const void* B,
                                void* C, const void* S, const void* topk_w, const void* ids, int P,
                                int N, int K, int top_k, long stride_am, long stride_be,
                                long stride_bn, long stride_cm, long sse, long ssn, long ssg, int E) {
  if (P > 64 || P < 1) return -1;
  hipStream_t s = (hipStream_t)stream;
  const int vec = (ssn == 1 && ssg % 4 == 0 && sse % 4 == 0 && ((uintptr_t)S % 4) == 0) ? 1 : 0;
  const int tm = rc >= 100 ? 1 : 0;
  rc %= 100;
#define ARGS5 s, phase, A, B, C, S, topk_w, ids, P, N, top_k, stride_am, stride_be, stride_bn, stride_cm, sse, ssn, ssg, E, vec, tm
#define BYRC(KK, MM) do { switch (rc) { \
    case 11: mx5_launch<KK, MM, 1, 1>(ARGS5); break;     case 21: mx5_launch<KK, MM, 2, 1>(ARGS5); break;     case 22: mx5_launch<KK, MM, 2, 2>(ARGS5); break; \
    case 41: mx5_launch<KK, MM, 4, 1>(ARGS5); break; \
    case 42: mx5_launch<KK, MM, 4, 2>(ARGS5); break; \
    case 44: mx5_launch<KK, MM, 4, 4>(ARGS5); break; \
    default: return -4; } } while (0)
#define BYMT5(KK) do { if (mt <= 1) BYRC(KK, 1); else if (mt <= 2) BYRC(KK, 2); else if (mt <= 4) BYRC(KK, 4); \
                       else BYRC(KK, 8); } while (0)
  switch (K) {
    case 4096: BYMT5(4096); break;
    case 2048: BYMT5(2048); break;
    case 1024: BYMT5(1024); break;
    default: return -2;
  }
#undef BYMT5
#undef BYRC
#undef ARGS5
  return hipGetLastError() == hipSuccess ? 0 : -3;
}
