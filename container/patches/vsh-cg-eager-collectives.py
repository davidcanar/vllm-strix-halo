#!/usr/bin/env python3
"""vllm-strix-halo: keep TP collectives out of CUDA graphs (PATCHES.md 33).

FULL CUDA graphs capture the TP all-reduces (RCCL over the OdinLink net
plugin) into the graph, and the first replay hangs with both ranks spinning
in ncclDevKernel (section 29). With VLLM_USE_BREAKABLE_CUDAGRAPH=1 and a
PIECEWISE cudagraph mode, this hook turns every GroupCoordinator
all_reduce / all_gather / reduce_scatter issued during a breakable capture
into an eager break: the current graph segment ends, the collective runs
eagerly (odl_ar2 for decode sizes) into a persistent output buffer, and
replay re-runs it eagerly and copies into that buffer. Outside a breakable
capture (eager mode, FULL capture, replay) nothing changes.
VSH_CG_EAGER_COLLECTIVES=0 disables.

Usage: python3 vsh-cg-eager-collectives.py
"""
import ast
from pathlib import Path

P = Path("/opt/venv/lib/python3.12/site-packages/vllm/distributed/parallel_state.py")
MARK = "# [vsh-cg-eager-collectives]"
SITES = [
    ("        if self.use_custom_op_call:\n            return torch.ops.vllm.all_reduce(",
     "self.all_reduce", "(input_,)"),
    ("        if self.use_custom_op_call:\n            return torch.ops.vllm.all_gather(",
     "self.all_gather", "(input_, dim)"),
    ("        if self.use_custom_op_call:\n            return torch.ops.vllm.reduce_scatter(",
     "self.reduce_scatter", "(input_, dim)"),
]
HELPER = f'''

{MARK}
_VSH_CG_EAGER = None


def _vsh_cg_break(fn, args):
    """Eager break for a collective issued inside a breakable cudagraph capture.

    Returns None when not applicable (caller continues normally); otherwise the
    collective's result, which stays at a fixed address: replay recomputes the
    collective eagerly and copies into it.
    """
    global _VSH_CG_EAGER
    if _VSH_CG_EAGER is None:
        import os as _os
        _VSH_CG_EAGER = _os.environ.get("VSH_CG_EAGER_COLLECTIVES", "1") not in ("", "0", "off")
    if not _VSH_CG_EAGER:
        return None
    from vllm.compilation.breakable_cudagraph import (
        BreakableCUDAGraphCapture,
        is_breakable_cudagraph_enabled,
    )
    if not is_breakable_cudagraph_enabled():
        return None
    cap = BreakableCUDAGraphCapture.current()
    if cap is None or not cap._capturing:
        return None
    from vllm.config import CUDAGraphMode
    from vllm.forward_context import get_forward_context, is_forward_context_available
    if (is_forward_context_available()
            and get_forward_context().cudagraph_runtime_mode == CUDAGraphMode.FULL):
        return None
    from vllm.utils.torch_utils import weak_ref_tensor
    wargs = tuple(weak_ref_tensor(a) if isinstance(a, torch.Tensor) else a for a in args)
    cap._end_segment()
    res = fn(*args)          # not capturing now -> the normal (eager) path

    def _replay():
        res.copy_(fn(*wargs))

    cap.segments.append(_replay)
    cap._num_eager_breaks += 1
    cap._begin_segment()
    return res
'''
s = P.read_text()
if MARK not in s:
    for anchor, fn, args in SITES:
        assert s.count(anchor) == 1, anchor
        ins = (f"        {MARK}\n        _vb = _vsh_cg_break({fn}, {args})\n"
               f"        if _vb is not None:\n            return _vb\n")
        s = s.replace(anchor, ins + anchor)
    s = s + HELPER
    ast.parse(s)
    P.write_text(s)
    print("vsh-cg-eager-collectives: applied")
else:
    print("vsh-cg-eager-collectives: already applied")
