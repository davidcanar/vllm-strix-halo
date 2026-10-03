#!/usr/bin/env python3
"""debug: live check of decode-time indexer selection (flag-file gated, inert by default).

touch /tmp/vsh_kpl_diag inside the container -> every 32nd decode indexer call with
>= 1024 visible pools appends one line to /tmp/vsh_kpl_diag.log: which reader ran,
visible pools of row 0, and the overlap of the kernel's selected pools with a
reference top-k recomputed in torch from the live cache. Costs a few ms when on."""
import sys
from pathlib import Path
P = Path("/opt/venv/lib/python3.12/site-packages/vllm/models/glm5next/amd/sparse_indexer.py")
MARK = "# [vsh-kpl-diag]"
ANCHOR = "        # Resolve to token-level indices in the output buffer.\n"
INSERT = f"""        {MARK}
        import os as _vd_os
        if index_kpool > 1 and _vd_os.path.exists("/tmp/vsh_kpl_diag"):
            try:
                import vllm.v1.attention.ops.vsh_kpool_paged_logits as _vkm
                _vd_n = getattr(_vkm, "_vd_n", 0) + 1
                _vkm._vd_n = _vd_n
                _sl = seq_lens.reshape(-1)
                _v0 = int(_sl[0])
                if _v0 >= 1024 and _vd_n % 32 == 0:
                    _ref = _vkm.kpool_paged_mqa_logits_ref(
                        padded_q_quant_cast[:1], kv_cache,
                        padded_weights[: next_n], seq_lens[:1],
                        decode_metadata.block_table[:1], max_pool_len)
                    _r0 = _ref[0, :_v0]
                    _kk = min(select_k, _v0)
                    _a = set(_r0.topk(_kk).indices.tolist())
                    _b = set(x for x in pool_topk[0].tolist() if 0 <= x < _v0)
                    _lg = logits[0, :_v0].float()
                    with open("/tmp/vsh_kpl_diag.log", "a") as _f:
                        _f.write("call=%d new_reader_calls=%d rows=%d next_n=%d vis0=%d sel_overlap=%.3f "
                                 "logit_maxdiff=%.3e nan=%d sel_min=%d sel_max=%d\n" % (
                                 _vd_n, getattr(_vkm, "_calls", -1), num_rows, next_n, _v0,
                                 len(_a & _b) / max(1, _kk),
                                 float((_lg - _r0).abs().nan_to_num(1e30).max()),
                                 int(_lg.isnan().sum()), min(_b) if _b else -1, max(_b) if _b else -1))
            except Exception as _e:
                with open("/tmp/vsh_kpl_diag.log", "a") as _f:
                    _f.write("diag error %r\n" % (_e,))
"""
src = P.read_text()
if MARK in src: print("vsh-kpl-diag: already applied"); sys.exit(0)
assert src.count(ANCHOR) == 1, "anchor"
P.write_text(src.replace(ANCHOR, INSERT + ANCHOR)); print("vsh-kpl-diag: applied")
