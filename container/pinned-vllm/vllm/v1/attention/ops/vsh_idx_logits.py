# SPDX-License-Identifier: Apache-2.0
# vllm-strix-halo: DS4 prefill indexer logits on gfx1151 (PATCHES.md 42).
#
# logits[q, k] = kv_scale[k] * sum_h w[q, h] * relu(Q[q, h, :] . K[k, :])   for k in [ks[q], ke[q])
#
# aiter's Triton _fp8_mqa_logits_kernel runs one program per query row and feeds fp8
# operands to tl.dot, which gfx1151 (no fp8 WMMA) converts to fp16 inside the key loop
# for every query: ~3 TFLOPS, the quadratic term of DS4 prefill (93 ms per C4A layer per
# 512-token chunk at 128K). Here Q and K are converted to fp16 once per call, and each
# program scores BQ query rows x BK keys with one fp16 WMMA dot of [BQ*H, D] x [D, BK].
# kv_scale > 0, so relu(s * scale) = relu(s) * scale and the scale is applied after the
# head sum. Positions outside a row's [ks, ke) are left at -inf, as aiter does.
import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _vsh_idx_logits_kernel(Q, K, KS, W, KST, KEN, OUT, R, NK,
                           stride_qr, stride_qh, stride_kn, stride_wr, stride_or,
                           H: tl.constexpr, D: tl.constexpr, BQ: tl.constexpr, BK: tl.constexpr,
                           SPLIT: tl.constexpr):
    pid = tl.program_id(0)
    sid = tl.program_id(1)
    offs_m = tl.arange(0, BQ * H)
    rows_m = pid * BQ + offs_m // H                              # query row of each (q, h) line
    heads = offs_m % H
    offs_d = tl.arange(0, D)
    offs_q = tl.arange(0, BQ)
    rows_q = pid * BQ + offs_q
    rmask_q = rows_q < R
    ks_q = tl.load(KST + rows_q, mask=rmask_q, other=2147483647)
    ke_q = tl.load(KEN + rows_q, mask=rmask_q, other=0)
    ks_q = tl.maximum(ks_q, 0)
    ke_q = tl.minimum(ke_q, NK)
    lo = tl.min(ks_q, axis=0)
    hi = tl.max(ke_q, axis=0)
    if hi > lo:
        # this program's share of the block's key range (BK-aligned segments)
        nblk = tl.cdiv(hi - lo, BK)
        per = tl.cdiv(nblk, SPLIT)
        b0 = lo + sid * per * BK
        b1 = tl.minimum(lo + (sid + 1) * per * BK, hi)
        rmask_m = rows_m < R
        q = tl.load(Q + rows_m[:, None] * stride_qr + heads[:, None] * stride_qh + offs_d[None, :],
                    mask=rmask_m[:, None], other=0.0)                        # [BQ*H, D] fp16
        w = tl.load(W + rows_m * stride_wr + heads, mask=rmask_m, other=0.0)  # [BQ*H] fp32
        offs_k = tl.arange(0, BK)
        for kb in range(b0, b1, BK):
            kidx = kb + offs_k
            kmask = kidx < b1
            kt = tl.load(K + kidx[None, :] * stride_kn + offs_d[:, None], mask=kmask[None, :], other=0.0)  # [D, BK]
            sc = tl.load(KS + kidx, mask=kmask, other=0.0)
            s = tl.dot(q, kt)                                                 # [BQ*H, BK] fp32
            s = tl.maximum(s, 0.0) * w[:, None]
            o = tl.sum(tl.reshape(s, (BQ, H, BK)), axis=1) * sc[None, :]     # [BQ, BK]
            m = (kidx[None, :] >= ks_q[:, None]) & (kidx[None, :] < ke_q[:, None]) & rmask_q[:, None]
            tl.store(OUT + rows_q[:, None] * stride_or + kidx[None, :], o, mask=m)


def vsh_idx_logits(q: torch.Tensor, k_fp8: torch.Tensor, scale: torch.Tensor, weights: torch.Tensor,
                   cu_seqlen_ks: torch.Tensor, cu_seqlen_ke: torch.Tensor,
                   bq: int = 1, bk: int = 64, split: int = 0, num_warps: int = 4) -> torch.Tensor:
    """Same contract as aiter fp8_mqa_logits (clean_logits=True): fp32 [M, N], -inf outside [ks, ke)."""
    r, h, d = q.shape
    nk = k_fp8.shape[0]
    q16 = q.to(torch.float16) if q.dtype != torch.float16 else q
    if q16.stride(2) != 1:
        q16 = q16.contiguous()
    k16 = k_fp8.to(torch.float16) if k_fp8.dtype != torch.float16 else k_fp8
    if k16.stride(1) != 1:
        k16 = k16.contiguous()
    sc = scale.reshape(-1).to(torch.float32).contiguous()
    w = weights.to(torch.float32)
    if w.stride(1) != 1:
        w = w.contiguous()
    aligned = (nk + 255) // 256 * 256
    out = torch.full((r, aligned), float("-inf"), dtype=torch.float32, device=q.device)[:, :nk]
    if r == 0 or nk == 0:
        return out
    if split <= 0:   # tiles from the 512-row sweep (bq 1, bk 64, 4 warps); split the keys for short chunks
        split = 1 if r >= 256 else min(16, -(-256 // r))
    grid = (triton.cdiv(r, bq), split)
    _vsh_idx_logits_kernel[grid](q16, k16, sc, w, cu_seqlen_ks, cu_seqlen_ke, out, r, nk,
                                 q16.stride(0), q16.stride(1), k16.stride(0), w.stride(0), out.stride(0),
                                 H=h, D=d, BQ=bq, BK=bk, SPLIT=split, num_warps=num_warps)
    return out
