#!/usr/bin/env python3
"""Kernel-class time share, normalised per step / per token.

Split the GPU time the way the requant decision needs it:
  * bf16 dense GEMMs  = what a requant of self_attn.*/lm_head would shrink
  * int4 MoE experts  = already quantized
  * attention / indexer / elementwise / collectives = untouched by the requant

usage: trace_share.py <trace.json.gz> [n_steps]
"""
from __future__ import annotations

import collections
import gzip
import json
import sys

with gzip.open(sys.argv[1], "rt") as fh:
    data = json.load(fh)
ev = data["traceEvents"] if isinstance(data, dict) else data

kern = [
    (e.get("name", ""), e.get("dur") or 0)
    for e in ev
    if e.get("ph") == "X" and e.get("cat") in ("kernel", "Kernel")
]
total_us = sum(d for _, d in kern)
print(f"kernels={len(kern)}  total GPU time = {total_us / 1e6:.2f} s")

CLASSES = (
    ("bf16 dense GEMM (wvSplitK/hipBLASLt)", ("wvSplitK", "Cijk_")),
    ("int4 MoE experts (fused_moe_gptq_awq)", ("fused_moe_kernel_gptq_awq", "fused_moe_kernel")),
    ("MoE plumbing (align/sort/sum)", ("moe_align", "count_and_sort", "moe_sum")),
    ("attention (MLA/indexer/kpool)", ("mla", "attention", "indexer", "kpool", "sparse_attn")),
    ("KDA (recurrent/chunk/conv)", ("kda", "delta_rule", "gated_delta", "conv1d")),
    ("mhc", ("mhc",)),
    ("collectives (odl/nccl)", ("odl2", "nccl", "all_reduce", "all_gather")),
    ("elementwise/reduce/copy", ("elementwise", "reduce_kernel", "copy", "vectorized", "act_and_mul", "silu")),
    ("triton fused (index/mask/layernorm)", ("triton_per", "triton_poi", "triton_red", "layer_norm", "norm")),
    ("memcpy/memset", ("memcpy", "memset", "Memcpy", "Memset")),
)

seen = collections.Counter()
by_class = collections.Counter()
for name, dur in kern:
    for label, pats in CLASSES:
        if any(p in name for p in pats):
            by_class[label] += dur
            seen[label] += 1
            break
    else:
        by_class["other"] += dur
        seen["other"] += 1

print(f"\n{'ms':>10} {'share':>7} {'n':>8}  class")
for label, us in by_class.most_common():
    print(f"{us / 1000:10.1f} {us / total_us * 100:6.1f}% {seen[label]:8d}  {label}")

if len(sys.argv) > 2:
    n = float(sys.argv[2])
    print(f"\nper step (n={n:g}):")
    for label, us in by_class.most_common():
        print(f"  {us / 1000 / n:8.2f} ms/step  {label}")
