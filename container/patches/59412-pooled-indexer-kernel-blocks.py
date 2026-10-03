#!/usr/bin/env python3
"""59412-pooled-indexer-kernel-blocks: port of upstream vllm PR #59412 (fixes #58858).

On ROCm the DSA indexer backend advertised kernel block sizes [1, MultipleOf(16)],
so select_common_block_size kept the hybrid KV-manager block as the kernel block
for the MLA/indexer group. GLM-5.3-Flash's kpool index cache stores pages of
64 pools = 256 tokens and both the writer (slot mapping) and every reader address
the block table at that page granularity. With a manager-granular table the
indexer write AND read for every pool past the table's width landed on page 0
(measured live on this rig, 13K prompt: 17 valid table columns for 51 needed
pages; slots for pools 3261..3264 = 61, 62, 63, 0) -- i.e. everything beyond
~4K tokens of context was aliased onto one page and unreadable.

The fix selects page-aligned kernel blocks (page * index_kpool tokens) for pooled
indexers, so vLLM's BlockTable expands manager blocks into storage pages.
indexer.py: upstream hunk verbatim (git apply). worker/utils.py: upstream change,
re-anchored (the pin's error-message line differs).
Usage (inside the container): python3 59412-pooled-indexer-kernel-blocks.py 59412-indexer.diff
"""
import subprocess
import sys
from pathlib import Path

SP = Path("/opt/venv/lib/python3.12/site-packages")
MARK = "# [59412]"

idx = SP / "vllm/v1/attention/backends/mla/indexer.py"
if "_kv_pool_tokens_per_state" in idx.read_text():
    print("59412 indexer.py: already applied")
else:
    subprocess.run(["git", "apply", "-p1", sys.argv[1]], cwd=SP, check=True)
    print("59412 indexer.py: applied")

ut = SP / "vllm/v1/worker/utils.py"
s = ut.read_text()
if MARK in s:
    print("59412 utils.py: already applied")
    sys.exit(0)
edits = [
    ("    kv_manager_block_size: int,\n    backends: list[type[AttentionBackend]],\n) -> int:\n",
     "    kv_manager_block_size: int,\n    backends: list[type[AttentionBackend]],\n"
     "    kv_cache_spec: KVCacheSpec | None = None,\n) -> int:\n"),
    ("    def block_size_is_supported(\n",
     f"    {MARK} forward the group spec so pooled indexers can pick page-aligned blocks\n"
     "    def supported_sizes(backend):\n"
     "        if kv_cache_spec is None:\n"
     "            return backend.get_supported_kernel_block_sizes()\n"
     "        return backend.get_supported_kernel_block_sizes(kv_cache_spec)\n\n"
     "    def block_size_is_supported(\n"),
    ("            for supported_size in backend.get_supported_kernel_block_sizes():\n",
     "            for supported_size in supported_sizes(backend):\n"),
    ("        for size in backend.get_supported_kernel_block_sizes()\n"
     "        if isinstance(size, int) and kv_manager_block_size % size == 0\n",
     "        for size in supported_sizes(backend)\n"
     "        if isinstance(size, int) and kv_manager_block_size % size == 0\n"),
    ("            selected_kernel_size = select_common_block_size(\n"
     "                kv_manager_block_size, group_backends\n            )\n",
     "            selected_kernel_size = select_common_block_size(\n"
     "                kv_manager_block_size, group_backends, kv_cache_spec\n            )\n"),
]
for old, new in edits:
    if s.count(old) != 1:
        sys.exit(f"59412 utils.py: anchor not unique/missing: {old[:60]!r}")
    s = s.replace(old, new)
ut.write_text(s)
print("59412 utils.py: applied")
