#!/usr/bin/env python3
"""vsh-router-packed-topk: same deterministic expert order, without the slow fused sort.

vsh-moe-router-deterministic-topk selects experts with a stable descending sort.
Inside grouped_topk's torch.compile region, inductor lowers that sort into one
persistent Triton kernel launched with 2 workgroups x 32 threads at decode
sizes: 242 us per call, 45 calls per step = ~10.9 ms/step of GPU (profile,
2026-10-02). Standalone the same code compiles differently (aten radix sort,
~19 us), so the cost is specific to the in-model compile.

Replacement: pack an order-preserving int64 key -- (monotonic int of the fp32
score) << 16 | (65535 - expert index) -- and take torch.topk(sorted=True).
All keys are distinct, so the result is unique: value descending, then lower
index first, i.e. exactly the stable sort's order (verified bit-identical on
600 cases incl. exact ties and -inf masks; full grouped_topk ids and weights
identical at M = 1/4/8/740). topk cannot be fused into a Triton sort.
Anchor: vllm/model_executor/layers/fused_moe/router/grouped_topk_router.py
"""
import ast, sys
from pathlib import Path
P = Path("/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/fused_moe/router/grouped_topk_router.py")
MARK = "# [vsh-router-packed-topk]"
OLD = "    return x.sort(dim=-1, descending=True, stable=True).indices[..., :k]\n"
NEW = f"""    {MARK} packed unique keys -> topk; same order as the stable sort
    assert x.shape[-1] <= 65536
    xi = x.float().contiguous().view(torch.int32).to(torch.int64)
    ordk = torch.where(xi >= 0, xi, xi ^ 0x7FFFFFFF)
    idx = torch.arange(x.shape[-1], device=x.device, dtype=torch.int64)
    return torch.topk(ordk * 65536 + (65535 - idx), k, dim=-1, sorted=True).indices
"""
s = P.read_text()
if MARK in s: print("vsh-router-packed-topk: already applied"); sys.exit(0)
assert s.count(OLD) == 1, "anchor"
s = s.replace(OLD, NEW); ast.parse(s); P.write_text(s); print("vsh-router-packed-topk: applied")
