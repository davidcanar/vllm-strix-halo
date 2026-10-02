#!/usr/bin/env python3
"""vsh-sync-instr: measure what the KDA chunk-index host sync actually costs.

`prepare_chunk_indices()` (`third_party/flash_linear_attention/ops/index.py`,
decorated `@tensor_cache`) materialises `cdiv(lens, chunk_size)` on the host
with `.tolist()` -- a device->host round trip. It is reached from the *chunked*
KDA path only (`kda.py`: `attn_metadata_narrowed.num_prefills > 0`), i.e. mixed
prefill+decode steps, not pure decode.

This patch adds an env-gated timer around that copy and logs the stalls, so the
cost can be measured instead of inferred. `VSH_SYNC_INSTR=1` enables; unset/0 is
byte-identical to stock.
"""
from __future__ import annotations

import pathlib
import shutil
import time

P = pathlib.Path(
    "/opt/venv/lib/python3.12/site-packages/vllm/third_party/flash_linear_attention/"
    "ops/index.py"
)
MARK = "# [vsh-sync-instr]"

OLD = """@tensor_cache
def prepare_chunk_indices(cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    # This will be fixed by https://github.com/vllm-project/vllm/pull/51540.
    with gpu_sync_allowed():
        chunk_counts = triton.cdiv(prepare_lens(cu_seqlens), chunk_size).tolist()
"""

NEW = f'''_VSH_SYNC = {{"n": 0, "stall_ms": 0.0, "worst_ms": 0.0}}  {MARK}


@tensor_cache
def prepare_chunk_indices(cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    # This will be fixed by https://github.com/vllm-project/vllm/pull/51540.
    import os  # {MARK}
    import time  # {MARK}

    _vsh_instr = os.environ.get("VSH_SYNC_INSTR", "").strip() in ("1", "on", "true")  # {MARK}
    if _vsh_instr:  # {MARK}
        _t0 = time.monotonic()
    with gpu_sync_allowed():
        chunk_counts = triton.cdiv(prepare_lens(cu_seqlens), chunk_size).tolist()
    if _vsh_instr:  # {MARK}
        _dt = (time.monotonic() - _t0) * 1000.0
        _VSH_SYNC["n"] += 1
        _VSH_SYNC["stall_ms"] += _dt
        _VSH_SYNC["worst_ms"] = max(_VSH_SYNC["worst_ms"], _dt)
        if _VSH_SYNC["n"] % 20 == 0:
            print(
                "[vsh-sync-instr] prepare_chunk_indices misses=%d stall_total=%.1fms "
                "worst=%.1fms seqs=%d chunk_size=%d"
                % (
                    _VSH_SYNC["n"],
                    _VSH_SYNC["stall_ms"],
                    _VSH_SYNC["worst_ms"],
                    max(cu_seqlens.numel() - 1, 0),
                    chunk_size,
                ),
                flush=True,
            )
'''


def main() -> int:
    text = P.read_text()
    if MARK in text:
        print("already patched")
        return 0
    n = text.count(OLD)
    if n != 1:
        raise SystemExit(f"anchor count {n}")
    backup = P.with_suffix(f".py.bak-{time.strftime('%Y%m%d-%H%M%S')}")
    shutil.copy2(P, backup)
    text = text.replace(OLD, NEW, 1)
    compile(text, str(P), "exec")
    P.write_text(text)
    print(f"sync instrumentation installed; backup={backup.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
