#!/usr/bin/env python3
"""vllm-strix-halo: explicit cap for the gfx1151 paged-MQA decode logits workspace (PATCHES.md 42).

On non-gfx942/950 ROCm the DS4 indexer reserves a (heads, decode_rows,
max_model_len) fp32 workspace: 128 MiB per decode row at 512K context.
_max_decode_logits_rows bounds the rows by max_num_seqs * (1 + spec) only when
it can read the vLLM config; inside the worker's custom op it cannot, so the
bound falls back to the batch's token count (512 rows -> 64 GiB at a 2048-token
prefill chunk). VSH_DS4_DECODE_ROWS (set by the DS4 launch scripts to
max_num_seqs * (1 + num_speculative_tokens)) caps it explicitly.
"""
import ast
import sys
from pathlib import Path

F = Path("/opt/venv/lib/python3.12/site-packages/vllm/v1/attention/ops/rocm_aiter_mla_sparse.py")
MARK = "[vsh-ds4-decode-rows]"
s = F.read_text()
if MARK in s:
    print("vsh-ds4-decode-rows: already applied")
    sys.exit(0)
old = '''    estimated.
    """
    try:
        vllm_config = get_current_vllm_config()
    except Exception:
        return num_batched_tokens
'''
new = '''    estimated.
    """
    import os as _os

    _cap = _os.environ.get("VSH_DS4_DECODE_ROWS", "")  # [vsh-ds4-decode-rows]
    if _cap.isdigit() and int(_cap) > 0:
        return min(num_batched_tokens, int(_cap))
    try:
        vllm_config = get_current_vllm_config()
    except Exception:
        return num_batched_tokens
'''
assert s.count(old) == 1, "_max_decode_logits_rows anchor"
s = s.replace(old, new)
ast.parse(s)
F.write_text(s)
print("vsh-ds4-decode-rows: explicit decode-row cap patched")
