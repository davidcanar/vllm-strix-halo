#!/usr/bin/env python3
"""vllm-strix-halo: DeepSeek-V4 decode on gfx1151 never inverse-RoPE'd its output (PATCHES.md 37).

rocm_sparse_attn_decode folds the inverse RoPE of the attention output into the
decode reduce epilogue and reports every row as rotated whenever it is given
inv_rope_positions, so DeepseekV4ROCMAiterMLAAttention.forward_mqa skips the
standalone rocm_inverse_rope_rows_ pass for decode rows. But on anything that
is not gfx942/gfx950, _rocm_sparse_attn_decode_ragged_triton takes the
"fallback path for un-tuned architectures": one _sparse_attn_decode_ragged_kernel
with no inverse-RoPE stage, and returns. On gfx1151 every decode step therefore
fed wo_a an output whose RoPE half was still rotated by its position: prefill
(rotated by the standalone pass) was right, decode degraded with position --
coherent below ~100 positions, debris past ~150 (PATCHES 36). Fix: apply the
inverse RoPE on the fallback path too, so the function honours its contract.

Usage: python3 vsh-ds4-decode-inv-rope.py [path/to/rocm_aiter_mla_sparse.py]
"""
import ast
import sys
from pathlib import Path

P = Path(sys.argv[1] if len(sys.argv) > 1 else
         "/opt/venv/lib/python3.12/site-packages/vllm/v1/attention/ops/rocm_aiter_mla_sparse.py")
MARK = "# [vsh-ds4-decode-inv-rope]"
ANCHOR = '''            IS_FNUZ_EXTRA=False,
            BLOCK_H=block_h,
            BLOCK_K=block_k,
            num_warps=8,
        )
        return out

    block_k = 32  # KV tokens walked per split-K iteration. Tuned on gfx950.
'''
FIXED = f'''            IS_FNUZ_EXTRA=False,
            BLOCK_H=block_h,
            BLOCK_K=block_k,
            num_warps=8,
        )
        {MARK} this kernel has no fused inverse-RoPE epilogue, but callers
        # (rocm_sparse_attn_decode) report the rows as rotated whenever
        # inv_rope_positions is given -- rotate them here (gfx1151 and other
        # un-tuned archs; gfx942/950 rotate in _sparse_attn_decode_reduce_kernel).
        if inv_rope_positions is not None:
            assert inv_rope_cos_sin_cache is not None
            rocm_inverse_rope_rows_(out, inv_rope_positions, inv_rope_cos_sin_cache,
                                    rope_head_dim)
        return out

    block_k = 32  # KV tokens walked per split-K iteration. Tuned on gfx950.
'''
s = P.read_text()
if MARK in s:
    print("vsh-ds4-decode-inv-rope: already applied")
else:
    assert s.count(ANCHOR) == 1, f"anchor x{s.count(ANCHOR)}"
    s = s.replace(ANCHOR, FIXED)
    ast.parse(s)
    P.write_text(s)
    print("vsh-ds4-decode-inv-rope: applied")
