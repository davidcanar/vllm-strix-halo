#!/usr/bin/env python3
"""vllm-strix-halo: DS4 prefill indexer logits on gfx1151 -> vsh_idx_logits (PATCHES.md 42).

DeepSeek-V4's sparse attention reads 512 selected compressed positions per token,
but the indexer that selects them scores every earlier compressed position: the
quadratic term of DS4 prefill. On gfx1151 that scoring ran in aiter's Triton
_fp8_mqa_logits_kernel at ~3 TFLOPS (fp8 operands converted to fp16 inside the
key loop for every query): 93 ms per C4A layer per 512-token chunk at 128K.
vsh_idx_logits converts Q and K to fp16 once per call and runs one fp16 WMMA
dot per [64 heads x 128] x [128 x 64 keys] tile: ~25 TFLOPS, ~9x, identical
-inf masks and top-k selections (scripts/test_idx_logits.py).
Only the DS4 call site (compress_ratio > 1) switches; GLM-5.3 keeps aiter.
Env VSH_IDX_LOGITS=0 restores aiter.

Usage (inside the container): python3 vsh-ds4-idx-logits.py /path/to/vsh_idx_logits.py
"""
import ast
import shutil
import sys
from pathlib import Path

SP = Path("/opt/venv/lib/python3.12/site-packages/vllm")
OPS = SP / "v1/attention/ops"
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("vsh_idx_logits.py")
MARK = "[vsh-ds4-idx-logits]"

shutil.copyfile(SRC, OPS / "vsh_idx_logits.py")
F = OPS / "rocm_aiter_mla_sparse.py"
s = F.read_text()
if MARK in s:
    print("vsh-ds4-idx-logits: already applied (module refreshed)")
    sys.exit(0)
old = """            logits = rocm_fp8_mqa_logits(
                q_fp8[chunk.token_start : chunk.token_end],
                (k_fp8, k_scale.view(torch.float32)),
                weights[chunk.token_start : chunk.token_end],
                chunk.cu_seqlen_ks,
                chunk.cu_seqlen_ke,
            )
"""
new = """            if compress_ratio > 1 and _vsh_idx_logits_on():
                # [vsh-ds4-idx-logits] fp16 WMMA indexer logits (~9x aiter's on gfx1151)
                from vllm.v1.attention.ops.vsh_idx_logits import vsh_idx_logits

                logits = vsh_idx_logits(
                    q_fp8[chunk.token_start : chunk.token_end],
                    k_fp8,
                    k_scale.view(torch.float32),
                    weights[chunk.token_start : chunk.token_end],
                    chunk.cu_seqlen_ks,
                    chunk.cu_seqlen_ke,
                )
            else:
                logits = rocm_fp8_mqa_logits(
                    q_fp8[chunk.token_start : chunk.token_end],
                    (k_fp8, k_scale.view(torch.float32)),
                    weights[chunk.token_start : chunk.token_end],
                    chunk.cu_seqlen_ks,
                    chunk.cu_seqlen_ke,
                )
"""
assert s.count(old) == 1, "DS4 prefill logits call site"
s = s.replace(old, new)
anchor = "def rocm_fp8_mqa_logits(\n"
assert s.count(anchor) == 1
helper = '''_VSH_IDX_ON = None


def _vsh_idx_logits_on() -> bool:  # [vsh-ds4-idx-logits]
    global _VSH_IDX_ON
    if _VSH_IDX_ON is None:
        import os

        on = os.environ.get("VSH_IDX_LOGITS", "1") not in ("", "0", "off")
        if on:
            from vllm.platforms.rocm import on_gfx1151

            on = on_gfx1151()
        _VSH_IDX_ON = on
    return _VSH_IDX_ON


'''
s = s.replace(anchor, helper + anchor)
ast.parse(s)
F.write_text(s)
print("vsh-ds4-idx-logits: DS4 prefill call site patched")
