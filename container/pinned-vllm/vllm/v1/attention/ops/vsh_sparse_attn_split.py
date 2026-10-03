# SPDX-License-Identifier: Apache-2.0
# vllm-strix-halo: split-KV (flash-decoding) variant of _sparse_attn_prefill_ragged_kernel.
#
# GLM-5.3 decode runs its sparse MLA attention through the *prefill* ragged
# kernel: one program per (query token, 16-head block), each streaming all
# ~2048 selected keys of 512-dim latent serially -> 8 workgroups for a 4-token
# MTP verify on a 20-WGP GPU, ~0.55 ms per call x 14 calls/step. Here each
# program handles one slice of a query's key range and writes a partial
# (max, sum, unnormalised acc); a combine kernel merges the slices and the
# attention sink. Same math as the original (online softmax, fp32 accumulate).
import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _split_partial_kernel(
    q_ptr, kv_ptr, kv_indices_ptr, kv_indptr_ptr, part_ptr,
    q_stride_t, q_stride_h, q_stride_d, kv_stride_n, kv_stride_d,
    num_heads, head_dim, num_kv, scale,
    NSPLIT: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_K: tl.constexpr,
):
    qi = tl.program_id(0)
    pid_h = tl.program_id(1)
    sp = tl.program_id(2)
    heads = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    dims = tl.arange(0, BLOCK_D)
    hmask = heads < num_heads
    dmask = dims < head_dim
    q = tl.load(q_ptr + qi * q_stride_t + heads[:, None] * q_stride_h + dims[None, :] * q_stride_d,
                mask=hmask[:, None] & dmask[None, :], other=0.0)
    neg = -3.4028234663852886e38
    m_i = tl.full((BLOCK_H,), neg, dtype=tl.float32)
    l_i = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_H, BLOCK_D), dtype=tl.float32)
    kv_start = tl.load(kv_indptr_ptr + qi)
    kv_len = tl.load(kv_indptr_ptr + qi + 1) - kv_start
    chunk = tl.cdiv(tl.cdiv(kv_len, NSPLIT), BLOCK_K) * BLOCK_K
    lo = sp * chunk
    hi = tl.minimum(kv_len, lo + chunk)
    koff = tl.arange(0, BLOCK_K)
    for k0 in tl.range(lo, hi, BLOCK_K):
        kpos = k0 + koff
        inr = kpos < hi
        slot = tl.load(kv_indices_ptr + kv_start + kpos, mask=inr, other=-1)
        valid = inr & (slot >= 0) & (slot < num_kv)
        safe = tl.where(valid, slot, 0).to(tl.int64)
        kv = tl.load(kv_ptr + safe[:, None] * kv_stride_n + dims[None, :] * kv_stride_d,
                     mask=valid[:, None] & dmask[None, :], other=0.0)
        s = tl.dot(q, tl.trans(kv)) * scale
        s = tl.where(hmask[:, None] & valid[None, :], s, neg)
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        p = tl.where(hmask[:, None] & valid[None, :], p, 0.0)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(kv.dtype), kv)
        m_i = m_new
    # partial layout: [nq, H, NSPLIT, BLOCK_D + 2] fp32 (acc..., m, l)
    row = ((qi * num_heads + heads) * NSPLIT + sp) * (BLOCK_D + 2)
    tl.store(part_ptr + row[:, None] + dims[None, :], acc, mask=hmask[:, None])
    tl.store(part_ptr + row + BLOCK_D, m_i, mask=hmask)
    tl.store(part_ptr + row + BLOCK_D + 1, l_i, mask=hmask)


@triton.jit
def _split_combine_kernel(
    part_ptr, attn_sink_ptr, out_ptr, out_stride_t, out_stride_h, out_stride_d,
    num_heads, head_dim,
    HAS_ATTN_SINK: tl.constexpr, NSPLIT: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_D: tl.constexpr,
):
    qi = tl.program_id(0)
    pid_h = tl.program_id(1)
    heads = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    dims = tl.arange(0, BLOCK_D)
    hmask = heads < num_heads
    neg = -3.4028234663852886e38
    base = (qi * num_heads + heads) * NSPLIT * (BLOCK_D + 2)
    m_all = tl.full((BLOCK_H,), neg, dtype=tl.float32)
    for sp in tl.static_range(NSPLIT):
        m_all = tl.maximum(m_all, tl.load(part_ptr + base + sp * (BLOCK_D + 2) + BLOCK_D, mask=hmask, other=neg))
    if HAS_ATTN_SINK:
        sink = tl.load(attn_sink_ptr + heads, mask=hmask, other=neg).to(tl.float32)
        m_all = tl.maximum(m_all, sink)
    l_tot = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_H, BLOCK_D), dtype=tl.float32)
    for sp in tl.static_range(NSPLIT):
        r = base + sp * (BLOCK_D + 2)
        m_s = tl.load(part_ptr + r + BLOCK_D, mask=hmask, other=neg)
        l_s = tl.load(part_ptr + r + BLOCK_D + 1, mask=hmask, other=0.0)
        w = tl.where(l_s > 0.0, tl.exp(m_s - m_all), 0.0)
        a_s = tl.load(part_ptr + r[:, None] + dims[None, :], mask=hmask[:, None], other=0.0)
        acc += a_s * w[:, None]
        l_tot += l_s * w
    l_kv = l_tot
    if HAS_ATTN_SINK:
        l_tot = l_tot + tl.exp(sink - m_all)
    denom = tl.maximum(l_tot, 1.0e-30)
    out = tl.where(l_kv[:, None] > 0.0, acc / denom[:, None], 0.0)
    tl.store(out_ptr + qi * out_stride_t + heads[:, None] * out_stride_h + dims[None, :] * out_stride_d,
             out, mask=hmask[:, None] & (dims[None, :] < head_dim))


def sparse_attn_ragged_split(q, kv, indices, indptr, scale, attn_sink, target_programs=96):
    """Same contract as _rocm_sparse_attn_prefill_ragged_triton (q [sq,h,d], kv [skv,d],
    indices [nnz] int32, indptr [sq+1] int32) -> out like q."""
    nq, nh, hd = q.shape
    block_h = 16
    block_d = triton.next_power_of_2(hd)
    block_k = 16 if hd >= 256 else 32
    hb = triton.cdiv(nh, block_h)
    nsplit = max(1, min(16, triton.next_power_of_2(triton.cdiv(target_programs, max(1, nq * hb)))))
    part = torch.empty((nq, nh, nsplit, block_d + 2), device=q.device, dtype=torch.float32)
    has_sink = attn_sink is not None
    sink = attn_sink.contiguous() if has_sink else torch.empty(1, device=q.device, dtype=torch.float32)
    out = torch.empty_like(q)
    _split_partial_kernel[(nq, hb, nsplit)](
        q, kv, indices, indptr, part, q.stride(0), q.stride(1), q.stride(2), kv.stride(0), kv.stride(1),
        nh, hd, kv.shape[0], float(scale),
        NSPLIT=nsplit, BLOCK_H=block_h, BLOCK_D=block_d, BLOCK_K=block_k, num_warps=4)
    _split_combine_kernel[(nq, hb)](
        part, sink, out, out.stride(0), out.stride(1), out.stride(2), nh, hd,
        HAS_ATTN_SINK=has_sink, NSPLIT=nsplit, BLOCK_H=block_h, BLOCK_D=block_d, num_warps=4)
    return out
