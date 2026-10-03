#!/usr/bin/env python3
"""vsh-w8a16-linear: load-time int8 (group 128) for BF16 linears + HIP W8A16 GEMV (gfx1151).

UnquantizedLinearMethod.process_weights_after_loading quantizes eligible layers
(vllm/model_executor/layers/vsh_w8a16.py: bf16, 2-D, K in {1024..8192 set},
N >= 32, not kv_b_proj / wk_weights_proj / embed / lm_head / vision) and frees
the BF16 copy; apply() runs libvsh_w8a16.so's GEMV for <= 8 rows and a
dequant + stock GEMM above that (prefill). Gate VSH_W8A16=1 (default off in
code; the cluster env sets it). Optional VSH_W8A16_INCLUDE=<regex on prefix>.
Usage: python3 vsh-w8a16-linear.py vsh_w8a16.py libvsh_w8a16.so
"""
import ast, shutil, sys
from pathlib import Path
SP = Path("/opt/venv/lib/python3.12/site-packages/vllm")
P = SP / "model_executor/layers/linear.py"
MARK = "# [vsh-w8a16-linear]"
A1 = "                layer.weight.data = weight.t().contiguous().t()\n"
I1 = f"""        {MARK} load-time int8 for eligible BF16 linears (gfx1151 GEMV)
        if current_platform.is_rocm():
            from vllm.model_executor.layers import vsh_w8a16 as _w8

            if _w8.enabled() and _w8.eligible(layer):
                _w8.quantize_(layer)
"""
A2 = "        return self._gemm_impl(layer, x, layer.weight, bias)\n"
I2 = f"""        if getattr(layer, "vsh_w8_q", None) is not None:  {MARK}
            from vllm.model_executor.layers import vsh_w8a16 as _w8

            return _w8.apply(layer, x, bias)
"""
s = P.read_text()
if MARK not in s:
    assert s.count(A1) == 1 and s.count(A2) == 1, "anchors"
    s = s.replace(A1, A1 + I1).replace(A2, I2 + A2)
    ast.parse(s); P.write_text(s); print("vsh-w8a16-linear: applied")
else:
    print("vsh-w8a16-linear: already applied")
shutil.copyfile(sys.argv[1], SP / "model_executor/layers/vsh_w8a16.py")
shutil.copyfile(sys.argv[2], SP / "libvsh_w8a16.so")
print("vsh-w8a16-linear: module + library installed")
