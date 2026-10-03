#!/usr/bin/env python3
"""vsh-idx-fp8max-fnuz: stop the decode-time indexer from scoring NaN.

On gfx1151 current_platform.fp8_dtype() is float8_e4m3fn (max 448; is_fp8_fnuz()
is gfx94x-only), so the AMD kpool compressor quantized every pooled indexer K
vector with FP8_MAX = 448 -- the absmax element of every vector lands at
~+-448. The decode-side paged-MQA-logits path (our aiter pa_mqa_logits.py
overlay) then converts the whole cache to float8_e4m3fnuz, whose max is 240.
torch's fp8 cast is non-saturating: every value with |x| > 240 becomes NaN
(measured: 256/288/.../448 -> nan). So practically every pool carried a NaN,
every decode-time indexer score was garbage, and once the context exceeds
index_topk (2048 tokens) the sparse attention selected pools arbitrarily
(only the force-tail pools were reliable). Prefill was unaffected (aiter's
prefill wrapper converts only on gfx942, and does it with the *0.5 rescale).

Fix: quantize to the fnuz-safe range when the fp8 dtype is e4m3fn on ROCm.
224.0 is the value vLLM itself uses for fnuz (quant_utils.get_fp8_min_max).
Scales are stored per vector, so every consumer stays consistent; the
fn -> fnuz conversion becomes exact (and -0 maps to +0, verified).
Cached KV written before the fix is wrong -- restart the engine.

Gate: VSH_IDX_FP8MAX_FNUZ=0 restores the stock 448 (default on).
Anchor: vllm/models/glm5next/amd/ops/kpool_compress.py (pin 73859fec).
"""
from __future__ import annotations

import sys
from pathlib import Path

P = Path("/opt/venv/lib/python3.12/site-packages/vllm/models/glm5next/amd/ops/kpool_compress.py")
MARK = "# [vsh-idx-fp8max-fnuz]"
OLD = "FP8_MAX = torch.finfo(FP8_DTYPE).max\n"
NEW = (
    "FP8_MAX = torch.finfo(FP8_DTYPE).max\n"
    f"{MARK} the decode indexer reads this cache as e4m3fnuz (max 240); values\n"
    "# above 240 turn into NaN in that cast, so quantize into the fnuz-safe range.\n"
    "import os as _vsh_os\n"
    "if (current_platform.is_rocm() and FP8_DTYPE == torch.float8_e4m3fn\n"
    "        and _vsh_os.environ.get(\"VSH_IDX_FP8MAX_FNUZ\", \"1\") not in (\"\", \"0\", \"off\")):\n"
    "    FP8_MAX = 224.0\n"
)

src = P.read_text()
if MARK in src:
    print("vsh-idx-fp8max-fnuz: already applied")
    sys.exit(0)
if src.count(OLD) != 1:
    sys.exit("vsh-idx-fp8max-fnuz: anchor not found exactly once")
P.write_text(src.replace(OLD, NEW))
print("vsh-idx-fp8max-fnuz: applied")
