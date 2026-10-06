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
