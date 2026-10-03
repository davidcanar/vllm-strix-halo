# SPDX-License-Identifier: Apache-2.0
# vllm-strix-halo: paged MQA indexer logits for the GLM-5.3 kpool cache on gfx1151.
#
# The GLM kpool writer (glm5next/amd/ops/kpool_compress.py) stores each indexer
# page as   [page_size x 128 fp8 e4m3fn values, 16x16 SHUFFLE order]
#           [page_size fp32 scales]
# and the decode path hands over a *page-level* block table. On gfx942/gfx950
# aiter's deepgemm kernel reads that layout (Preshuffle=True); gfx1151 fell
# through to aiter's `_stage1`, which is a block_size == 1 reader: it used
# block_table[b, pos] as a per-pool row index and read the first 128 bytes of a
# page, so every decode-time indexer score was garbage once the context passed
# index_topk. This kernel reads the real layout. fp8 is decoded from the raw
# e4m3fn bits in-kernel (exact in bf16), so no fnuz conversion is needed.
import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _e4m3fn_bits_to_f32(b):
    b = b.to(tl.int32)
    sign = tl.where((b & 0x80) != 0, -1.0, 1.0)
    e = (b >> 3) & 0xF
    m = (b & 0x7).to(tl.float32)
    mag = tl.where(e == 0, m * 0.001953125,  # subnormal: m/8 * 2^-6
                   (8.0 + m) * tl.exp2((e - 10).to(tl.float32)))  # (1+m/8)*2^(e-7)
    return sign * mag


@triton.jit
def _kpool_paged_logits_kernel(
    q_ptr,            # uint8 view of fp8 e4m3fn q  [B, NEXT_N, H, D]
    kv_ptr,           # uint8 raw cache             [num_pages, PAGE * (D + 4)]
    w_ptr,            # fp32 weights                [B * NEXT_N, H]
    ctx_ptr,          # int32 visible pools         [B] or [B, NEXT_N]
    bt_ptr,           # int32 page table            [B, max_pages]
    out_ptr,          # fp32 logits                 [B * NEXT_N, max_model_len]
    stride_qb, stride_qn, stride_qh,
    stride_w, stride_page, stride_bt, stride_out,
    max_pages, max_model_len,
    H: tl.constexpr, D: tl.constexpr, PAGE: tl.constexpr,
    NEXT_N: tl.constexpr, CTX_PER_ROW: tl.constexpr, PAGES_PER_PROG: tl.constexpr,
):
    row = tl.program_id(0)
    pid = tl.program_id(1)
    b = row // NEXT_N
    j = row % NEXT_N
    if CTX_PER_ROW:
        limit = tl.load(ctx_ptr + row)
    else:
        limit = tl.load(ctx_ptr + b) - NEXT_N + j + 1

    offs_h = tl.arange(0, H)
    offs_d = tl.arange(0, D)
    q_bits = tl.load(q_ptr + b * stride_qb + j * stride_qn
                     + offs_h[:, None] * stride_qh + offs_d[None, :])
    q = _e4m3fn_bits_to_f32(q_bits).to(tl.bfloat16)              # [H, D]
    w = tl.load(w_ptr + row * stride_w + offs_h)                    # [H]

    offs_p = tl.arange(0, PAGE)
    # SHUFFLE: 16-token tiles; inside a tile, 16-dim sub-blocks of 16x16 bytes.
    k_off = ((offs_p[:, None] // 16) * 16 * D + (offs_d[None, :] // 16) * 256
             + (offs_p[:, None] % 16) * 16 + offs_d[None, :] % 16)
    for i in range(PAGES_PER_PROG):
        page_i = pid * PAGES_PER_PROG + i
        pos = page_i * PAGE + offs_p
        if page_i < max_pages and page_i * PAGE < limit:
            page = tl.load(bt_ptr + b * stride_bt + page_i).to(tl.int64)
            base = kv_ptr + page * stride_page
            k = _e4m3fn_bits_to_f32(tl.load(base + k_off)).to(tl.bfloat16)  # [PAGE, D]
            scale = tl.load((base + PAGE * D).to(tl.pointer_type(tl.float32)) + offs_p)
            s = tl.dot(q, tl.trans(k))                             # [H, PAGE] fp32
            s = tl.maximum(s * scale[None, :], 0.0) * w[:, None]
            logit = tl.sum(s, axis=0)
            logit = tl.where(pos < limit, logit, float("-inf"))
            tl.store(out_ptr + row * stride_out + pos, logit, mask=pos < max_model_len)
        elif page_i < max_pages:
            tl.store(out_ptr + row * stride_out + pos, float("-inf"),
                     mask=pos < max_model_len)


def kpool_paged_mqa_logits(q_fp8, kv_cache, weights, context_lens, block_tables,
                           max_model_len, out=None):
    """q_fp8 [B, next_n, H, D] e4m3fn; kv_cache raw uint8 [num_pages, PAGE, (1,) D+4];
    returns fp32 logits [B*next_n, max_model_len] (-inf where not visible)."""
    B, next_n, H, D = q_fp8.shape
    page = kv_cache.shape[1]
    assert page % 16 == 0 and D % 16 == 0 and q_fp8.stride(-1) == 1
    kv_u8 = kv_cache.view(torch.uint8)
    rows = B * next_n
    if out is None:
        out = torch.empty((rows, max_model_len), dtype=torch.float32, device=q_fp8.device)
    out.fill_(float("-inf"))
    ctx = context_lens.to(torch.int32).contiguous()
    ctx_per_row = ctx.numel() == rows and not (next_n > 1 and ctx.numel() == B)
    max_pages = block_tables.shape[1]
    ppp = 4
    grid = (rows, triton.cdiv(max_pages, ppp))
    _kpool_paged_logits_kernel[grid](
        q_fp8.view(torch.uint8), kv_u8, weights, ctx, block_tables, out,
        q_fp8.stride(0), q_fp8.stride(1), q_fp8.stride(2),
        weights.stride(0), kv_u8.stride(0), block_tables.stride(0), out.stride(0),
        max_pages, max_model_len,
        H=H, D=D, PAGE=page, NEXT_N=next_n, CTX_PER_ROW=ctx_per_row,
        PAGES_PER_PROG=ppp, num_warps=4,
    )
    return out


def kpool_paged_mqa_logits_ref(q_fp8, kv_cache, weights, context_lens, block_tables,
                               max_model_len):
    """Plain-torch reference for the same layout (slow; for tests)."""
    B, next_n, H, D = q_fp8.shape
    page = kv_cache.shape[1]
    kv = kv_cache.view(torch.uint8).reshape(kv_cache.shape[0], -1)
    out = torch.full((B * next_n, max_model_len), float("-inf"), device=q_fp8.device)
    p = torch.arange(page, device=q_fp8.device)
    d = torch.arange(D, device=q_fp8.device)
    k_off = ((p[:, None] // 16) * 16 * D + (d[None, :] // 16) * 256
             + (p[:, None] % 16) * 16 + d[None, :] % 16)
    ctx = context_lens.to(torch.int64).reshape(-1).tolist()
    for b in range(B):
        for j in range(next_n):
            row = b * next_n + j
            lim = ctx[row] if len(ctx) == B * next_n and not (next_n > 1 and len(ctx) == B) \
                else ctx[b] - next_n + j + 1
            npg = (lim + page - 1) // page
            for pi in range(npg):
                pg = kv[int(block_tables[b, pi])]
                k = pg[k_off.reshape(-1)].view(torch.float8_e4m3fn).float().reshape(page, D)
                sc = pg[page * D: page * D + 4 * page].view(torch.float32)
                s = q_fp8[b, j].float() @ k.T * sc[None, :]
                lg = (s.clamp_min(0) * weights[row][:, None]).sum(0)
                pos = pi * page + p
                keep = pos < lim
                out[row, pos[keep]] = lg[keep]
    return out
