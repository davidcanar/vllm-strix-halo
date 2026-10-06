"""Routing dump for the MXFP4 direct path: patch a working copy of vsh_moe_int4.py so VSH_MX_DUMP_IDS=/path/prefix records the routed expert ids (eager runs; syncs per call). Used to tune MoE v5 on real DS4 decode routing (PATCHES 41); replay with moe_replay.py.
usage: python3 moe_dump_ids.py path/to/vsh_moe_int4.py"""
import sys
from pathlib import Path
p = Path(sys.argv[1]); t = p.read_text(encoding="utf-8")
if "VSH_MX_DUMP_IDS" not in t:
    old = '''def mxfp4_direct_moe(output, x, plan, topk_weights, topk_ids):
    """output[M, N2] (bf16) = sum_k w_k * W2_e(act(W13_e x)) for M*top_k <= 64."""
'''
    new = '''_MX_DUMP = os.environ.get("VSH_MX_DUMP_IDS", "")
_mx_dump_buf = []


def _mx_dump(ids):
    """Diagnostic: keep the routed expert ids of the first 30000 direct-path calls (syncs per call)."""
    if len(_mx_dump_buf) >= 30000:
        return
    _mx_dump_buf.append(ids.detach().to("cpu", torch.int16))
    if len(_mx_dump_buf) % 1500 == 0:
        torch.save(_mx_dump_buf, f"{_MX_DUMP}.{os.getpid()}.pt")


def mxfp4_direct_moe(output, x, plan, topk_weights, topk_ids):
    """output[M, N2] (bf16) = sum_k w_k * W2_e(act(W13_e x)) for M*top_k <= 64."""
    if _MX_DUMP:
        _mx_dump(topk_ids)
'''
    assert t.count(old) == 1
    t = t.replace(old, new)
    p.write_text(t, encoding="utf-8", newline="\n")
print("dump hook present")
