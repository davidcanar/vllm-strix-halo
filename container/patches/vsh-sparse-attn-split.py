#!/usr/bin/env python3
"""vsh-sparse-attn-split: split-KV decode path for the sparse MLA attention.

GLM decode (MTP verify rows and draft steps) runs _sparse_attn_prefill_ragged_kernel
with one program per (query, 16-head block): 8 workgroups at M=4, 2 at M=1.
Route small query counts to vllm/v1/attention/ops/vsh_sparse_attn_split.py
(flash-decoding split + combine). Unit test vs the original: max abs diff ~1e-3
(bf16 output rounding), 2.3x faster at M=4 / 2048 keys, 3.9x at M=1.
Gate: VSH_SPARSE_ATTN_SPLIT=0 restores the original (default on).
Usage: python3 vsh-sparse-attn-split.py path/to/vsh_sparse_attn_split.py
"""
import ast, shutil, sys
from pathlib import Path
SP = Path("/opt/venv/lib/python3.12/site-packages/vllm/v1/attention/ops")
P = SP / "rocm_aiter_mla_sparse.py"
MARK = "# [vsh-sparse-attn-split]"
ANCHOR = "    block_h = 16\n    block_d = triton.next_power_of_2(head_dim)\n    block_k = 16 if head_dim >= 256 else 32\n    num_warps = 4\n    out = torch.empty_like(q)\n    _sparse_attn_prefill_ragged_kernel["
INSERT = f"""    {MARK} few query rows (decode / MTP verify): split the key range
    import os as _vs_os
    if num_queries * triton.cdiv(num_heads, 16) <= 64 and _vs_os.environ.get(
        "VSH_SPARSE_ATTN_SPLIT", "1"
    ) not in ("", "0", "off"):
        from vllm.v1.attention.ops.vsh_sparse_attn_split import sparse_attn_ragged_split

        return sparse_attn_ragged_split(
            q, kv, indices, indptr, scale, attn_sink if has_attn_sink else None
        )
"""
s = P.read_text()
if MARK not in s:
    assert s.count(ANCHOR) == 1, "anchor"
    s = s.replace(ANCHOR, INSERT + ANCHOR); ast.parse(s); P.write_text(s); print("vsh-sparse-attn-split: routing applied")
else:
    print("vsh-sparse-attn-split: already applied")
shutil.copyfile(sys.argv[1], SP / "vsh_sparse_attn_split.py"); print("vsh-sparse-attn-split: module installed")
