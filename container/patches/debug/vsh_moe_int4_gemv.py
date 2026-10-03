# SPDX-License-Identifier: Apache-2.0
# vllm-strix-halo: decode-specialised int4 (W4A16, symmetric uint4b8, group-wise)
# MoE GEMV for gfx1151.
#
# The stock fused_moe_kernel_gptq_awq pads every expert to BLOCK_SIZE_M (32) rows
# and runs a tl.dot over a [32 x 64] x [64 x 32] tile per K step, loading each
# packed byte twice (once per nibble). At decode (<= 4 tokens, 1-2 real rows per
# expert) it is latency/ALU-bound: ~90 GB/s, and it slows with the GPU core clock.
# Here one program owns (expert block, BLOCK_N output rows), streams those rows'
# packed bytes once along the contiguous K axis, dequantises in registers and
# applies them to up to 4 token rows with FMAs.
#
# Layouts (as produced by convert_to_wna16_moe_kernel_format / used by the stock
# kernel): B uint8 [E, N, K//2] (low nibble = even k), B_scale [E, N, K//group],
# A [num_tokens, K], C viewed as [num_pairs, N] via stride_cm. Rows of a block are
# the first entries of sorted_token_ids[pid_m*BM : pid_m*BM + BM].
import torch

from vllm.triton_utils import tl, triton

MAX_ROWS = 16


@triton.jit
def _moe_int4_gemv_kernel(
    a_ptr, b_ptr, c_ptr, s_ptr, topk_w_ptr, sorted_ids_ptr, expert_ids_ptr, ntpp_ptr,
    N, K, num_valid,
    stride_am, stride_ak, stride_be, stride_bn, stride_bk,
    stride_cm, stride_cn, stride_se, stride_sn, stride_sk,
    GROUP: tl.constexpr, TOP_K: tl.constexpr, MUL_W: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_KB: tl.constexpr, ROWS: tl.constexpr, BM: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    if pid_m * BM >= tl.load(ntpp_ptr):
        return
    e = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    all_tok = tl.load(sorted_ids_ptr + pid_m * BM + tl.arange(0, BM))
    nrows = tl.sum((all_tok < num_valid).to(tl.int32), axis=0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    nmask = offs_n < N
    rows = tl.arange(0, ROWS)
    kb = tl.arange(0, BLOCK_KB)                 # byte index within the chunk
    b_base = b_ptr + e * stride_be + offs_n[:, None] * stride_bn
    s_base = s_ptr + e * stride_se + offs_n * stride_sn
    for r0 in range(0, nrows, ROWS):            # valid rows sit at the front of the block
        rvalid = (r0 + rows) < nrows
        tok = tl.load(sorted_ids_ptr + pid_m * BM + r0 + rows, mask=rvalid, other=0).to(tl.int64)
        if e == -1:
            z = tl.zeros((ROWS, BLOCK_N), dtype=c_ptr.dtype.element_ty)
            tl.store(c_ptr + tok[:, None] * stride_cm + offs_n[None, :] * stride_cn, z,
                     mask=rvalid[:, None] & nmask[None, :])
        else:
            a_row = tok // TOP_K
            acc = tl.zeros((ROWS, BLOCK_N), dtype=tl.float32)
            for k0 in range(0, K // 2, BLOCK_KB):   # k0 in bytes
                w = tl.load(b_base + (k0 + kb)[None, :] * stride_bk, mask=nmask[:, None], other=0x88)
                w = w.to(tl.int32)
                lo = ((w & 0xF) - 8).to(tl.float32)     # k = 2*(k0+kb)
                hi = ((w >> 4) - 8).to(tl.float32)      # k = 2*(k0+kb)+1
                sc = tl.load(s_base + ((2 * k0) // GROUP) * stride_sk, mask=nmask, other=0.0).to(tl.float32)
                xe = tl.load(a_ptr + a_row[:, None] * stride_am + (2 * (k0 + kb))[None, :] * stride_ak,
                             mask=rvalid[:, None], other=0.0)
                xo = tl.load(a_ptr + a_row[:, None] * stride_am + (2 * (k0 + kb) + 1)[None, :] * stride_ak,
                             mask=rvalid[:, None], other=0.0)
                lo_s = (lo * sc[:, None]).to(xe.dtype)      # [BLOCK_N, KB]
                hi_s = (hi * sc[:, None]).to(xe.dtype)
                acc = tl.dot(xe, tl.trans(lo_s), acc=acc)
                acc = tl.dot(xo, tl.trans(hi_s), acc=acc)
            if MUL_W:
                wgt = tl.load(topk_w_ptr + tok, mask=rvalid, other=0.0).to(tl.float32)
                acc = acc * wgt[:, None]
            tl.store(c_ptr + tok[:, None] * stride_cm + offs_n[None, :] * stride_cn,
                     acc.to(c_ptr.dtype.element_ty), mask=rvalid[:, None] & nmask[None, :])


def can_use(A, B, B_zp, block_shape, top_k, config):
    return (B_zp is None and B.dtype == torch.uint8 and block_shape is not None
            and block_shape[1] in (64, 128) and A.size(0) * top_k <= 64
            and config["BLOCK_SIZE_M"] in (16, 32, 64) and B.size(2) * 2 == A.size(1)
            and (A.size(1) // 2) % 64 == 0)


def moe_int4_gemv(A, B, C, B_scale, topk_weights, sorted_token_ids, expert_ids,
                  num_tokens_post_padded, mul_routed_weight, top_k, config, block_shape,
                  block_n=64, block_kb=64):
    M = A.size(0)
    BM = config["BLOCK_SIZE_M"]
    EM = min(sorted_token_ids.size(0), M * top_k * BM)
    N, K = B.size(1), A.size(1)
    grid = (triton.cdiv(EM, BM), triton.cdiv(N, block_n))
    _moe_int4_gemv_kernel[grid](
        A, B, C, B_scale, topk_weights if topk_weights is not None else A, sorted_token_ids,
        expert_ids, num_tokens_post_padded, N, K, M * top_k,
        A.stride(0), A.stride(1), B.stride(0), B.stride(1), B.stride(2),
        C.stride(1), C.stride(2), B_scale.stride(0), B_scale.stride(1), B_scale.stride(2),
        GROUP=block_shape[1], TOP_K=top_k, MUL_W=bool(mul_routed_weight),
        BLOCK_N=block_n, BLOCK_KB=block_kb, ROWS=MAX_ROWS, BM=BM, num_warps=4,
    )
