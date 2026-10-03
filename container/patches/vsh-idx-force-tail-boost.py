#!/usr/bin/env python3
"""vsh-idx-force-tail-boost: guarantee the recent pools *without* dropping selected ones.

vsh-idx-force-tail (2026-10-01) wrote the 128 most recent pools over the LAST 128
columns of the top-k output, assuming those were the lowest-ranked. The top-k
kernels do not return sorted output, so that overwrite discarded ~1/4 of the
pools the indexer had selected -- at random. Measured live (6.8K prompt, needle
pools all selected by top-k): only 6/10 needle tokens reached the sparse
attention on some steps, and the model copied the watchwords with wrong digits.
(The overwrite was masking the real bug, a garbage decode reader + page-aliased
cache; with those fixed it only does harm.)

Same intent, done safely: raise the logits of each row's most recent
_FORCE_TAIL_POOLS visible pools to a huge finite value *before* top-k, so they
are always selected and nothing else is evicted except the genuinely
lowest-scoring pools. Applies to both the prefill and the decode selection.
Anchor: vllm/models/glm5next/amd/sparse_indexer.py (pin 73859fec + vsh patches).
"""
import ast
import sys
from pathlib import Path

P = Path("/opt/venv/lib/python3.12/site-packages/vllm/models/glm5next/amd/sparse_indexer.py")
MARK = "# [vsh-idx-force-tail-boost]"
s = P.read_text()
if MARK in s:
    print("vsh-idx-force-tail-boost: already applied")
    sys.exit(0)

HELPER_ANCHOR = "# kpool write helper: form pools from the current token batch and compress them\n"
HELPER = f'''{MARK}
def _boost_recent_tail_logits(logits, row_start, row_end, k=_FORCE_TAIL_POOLS):
    """Make each row's last ``k`` visible pools win top-k (no post-hoc overwrite)."""
    if k <= 0 or logits.numel() == 0:
        return
    rows = logits.shape[0]
    ke = row_end.reshape(-1).to(torch.int64)
    if ke.shape[0] != rows:
        ke = ke.repeat_interleave(max(rows // max(ke.shape[0], 1), 1))[:rows]
    ks = row_start.reshape(-1).to(torch.int64) if row_start is not None else torch.zeros_like(ke)
    if ks.shape[0] != rows:
        ks = ks.repeat_interleave(max(rows // max(ks.shape[0], 1), 1))[:rows]
    lo = torch.maximum(ke - int(k), ks)
    cols = torch.arange(logits.shape[1], device=logits.device, dtype=torch.int64)
    mask = (cols[None, :] >= lo[:, None]) & (cols[None, :] < ke[:, None])
    logits.masked_fill_(mask, 3.0e38)


'''
edits = [
    (HELPER_ANCHOR, HELPER + HELPER_ANCHOR),
    # prefill: boost before top-k, drop the overwrite
    ("            torch.ops._C.top_k_per_row_prefill(\n",
     "            if index_kpool > 1:\n"
     "                _boost_recent_tail_logits(\n"
     "                    logits, chunk.cu_seqlen_ks, chunk.cu_seqlen_ke\n"
     "                )\n"
     "            torch.ops._C.top_k_per_row_prefill(\n"),
    ("                _force_recent_tail_pools(pool_ids, chunk.cu_seqlen_ke)\n", ""),
    # decode: boost before top-k, drop the overwrite
    ("        torch.ops._C.top_k_per_row_decode(\n",
     "        if index_kpool > 1:\n"
     "            _boost_recent_tail_logits(logits, None, seq_lens)\n"
     "        torch.ops._C.top_k_per_row_decode(\n"),
    ("            _force_recent_tail_pools(pool_ids, _vis_pools)\n", ""),
]
for old, new in edits:
    if s.count(old) != 1:
        sys.exit(f"vsh-idx-force-tail-boost: anchor count {s.count(old)} for {old[:70]!r}")
    s = s.replace(old, new)
ast.parse(s)
P.write_text(s)
print("vsh-idx-force-tail-boost: applied")
