#!/usr/bin/env python3
"""vsh-kpool-paged-logits: a correct decode-time indexer-logits reader on gfx1151.

THE BUG. GLM-5.3's sparse attention picks, per decoded token, the top 512 kpool
pools (2048 tokens) by indexer score once the context exceeds index_topk. The
kpool writer stores each indexer page as [64 x 128 fp8 values in 16x16 SHUFFLE
order][64 fp32 scales] and decode passes a *page-level* block table. On
gfx942/gfx950 aiter's deepgemm kernel reads that layout (Preshuffle=True); on
gfx1151 rocm_fp8_paged_mqa_logits fell through to aiter's
deepgemm_fp8_paged_mqa_logits_stage1, a block_size == 1 reader: it indexes the
block table per *pool position* and reads the first 128 bytes of a page as a
pool row (plus, via our fnuz overlay, overflowed values to NaN). Every
decode-time indexer score was garbage.

Measured (unit test, cache filled by the production writer, ground truth from
the writer's own compressed K): production top-512 overlap 0.32-0.35 -- exactly
random for 1 500 pools; this reader 1.000, max rel err 9e-8. End to end, a 13K
prompt with three mid-context watchwords: 0/3 on every seed before.

THE FIX. Route gfx1151 paged decode (block_size > 1, fp8 q, non-C4A) to
vllm/v1/attention/ops/vsh_kpool_paged_logits.py, which reads the real layout
and decodes e4m3fn bits in-kernel (no whole-cache fnuz conversion: also removes
~12.6 ms/step of float8_copy). No host sync; static grid.

Gate: VSH_KPOOL_PAGED_LOGITS=0 restores the old path (default on).
Anchor: vllm/v1/attention/ops/rocm_aiter_mla_sparse.py (pin 73859fec).
Usage: python3 vsh-kpool-paged-logits.py [path/to/vsh_kpool_paged_logits.py]
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

SP = Path("/opt/venv/lib/python3.12/site-packages/vllm/v1/attention/ops")
P = SP / "rocm_aiter_mla_sparse.py"
MOD_SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("vsh_kpool_paged_logits.py")
MARK = "# [vsh-kpool-paged-logits]"
ANCHOR = "    aiter_paged_mqa_logits_module = None\n\n    if rocm_aiter_ops.is_enabled() or rocm_aiter_ops.is_rdna_aiter_enabled():\n"
INSERT = f"""    {MARK} gfx1151 has no aiter reader for the paged SHUFFLE indexer
    # cache; the stage1 fallback below is a block_size == 1 reader and scores
    # garbage (random top-k). Use the layout-correct Triton reader instead.
    import os as _vsh_os

    if (
        _on_gfx1151
        and block_size > 1
        and q_fp8.dtype == torch.float8_e4m3fn
        and not _indexer_k_is_c4a_block_flat(compress_ratio)
        and _vsh_os.environ.get("VSH_KPOOL_PAGED_LOGITS", "1") not in ("", "0", "off")
    ):
        from vllm.v1.attention.ops.vsh_kpool_paged_logits import (
            kpool_paged_mqa_logits,
        )

        (_vsh_out,) = current_workspace_manager().get_simultaneous(
            ((batch_size * next_n, max_model_len), torch.float32),
        )
        return kpool_paged_mqa_logits(
            q_fp8, kv_cache_fp8, weights, context_lens, block_tables,
            max_model_len, out=_vsh_out,
        )

"""

src = P.read_text()
if MARK in src:
    print("vsh-kpool-paged-logits: already applied")
else:
    if src.count(ANCHOR) != 1:
        sys.exit("vsh-kpool-paged-logits: anchor not found exactly once")
    P.write_text(src.replace(ANCHOR, INSERT + ANCHOR))
    print("vsh-kpool-paged-logits: routing applied")
shutil.copyfile(MOD_SRC, SP / "vsh_kpool_paged_logits.py")
print("vsh-kpool-paged-logits: kernel module installed")
