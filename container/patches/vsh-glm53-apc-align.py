#!/usr/bin/env python3
"""vsh-glm53-apc-align: stop the EAGLE last-block drop from eating prefix hits.

Root cause on this pin (verified live, vLLM 0.31.0.dev0+git.73859fec):

  * `v1/core/kv_cache_utils.py:2364-2365` returns EARLY for GLM-5-Next
    (`elif glm5_groups := _get_kv_cache_groups_glm5_next(...): return glm5_groups`),
    so `_annotate_eagle_groups()` (called further down at 2419) is dead code and
    **no group ever gets `is_eagle_group = True`**.
  * `v1/core/kv_cache_coordinator.py:108-110` then "conservatively" falls back to
    flagging **every** group as EAGLE. Our spec method normalizes to `mtp`
    (`config/speculative.py:1135-1139`), so `use_eagle()` is True and the
    fallback fires every boot.
  * Consequence: the target MLA group, the padded KDA/Mamba groups and the
    kpool-tail group all take an EAGLE last-block pop (one scheduler page =
    2304 tokens on this rig), and the hybrid `min()` walks the hit down.

Fix: resolve the set to groups that hold **only** drafter layers. The MTP
drafter layer shares group 0 with the target MLA layers, so that set is empty
and the drop stays off -- which is what upstream intends ("the group that holds
the drafter"), not "every group". The write-side protection
(`num_reprefillable_tokens = num_prefill_lookahead - 1` tokens never cached,
`kv_cache_coordinator.py:347`) is untouched, and the lookahead guard is
re-asserted explicitly because an empty set silently disables it.

Env-gated: VSH_GLM53_APC_ALIGN=1 enables. Default off = byte-identical behaviour
(the edited region is inside `if _vsh_apc_align_enabled():`).

Idempotent, fail-closed (every anchor must match exactly once), backed up, and
AST-checked before writing.
"""
from __future__ import annotations

import pathlib
import py_compile
import shutil
import time

P = pathlib.Path(
    "/opt/venv/lib/python3.12/site-packages/vllm/v1/core/kv_cache_coordinator.py"
)
MARK = "# [vsh-glm53-apc-align]"

HELPER = f'''def _vsh_apc_align_enabled() -> bool:  {MARK}
    import os

    return os.environ.get("VSH_GLM53_APC_ALIGN", "0").strip() in ("1", "on", "true")


def _vsh_apc_drafter_group_ids(kv_cache_config) -> set[int]:  {MARK}
    """KV cache groups that hold *only* spec-decode drafter layers.

    Two conservative signals:
      1. every layer name in the group carries a drafter marker
         (mtp / nextn / draft / eagle);
      2. the group's spec is an exact ``SlidingWindowSpec`` -- a DFlash2-style
         drafter's own group. ``KpoolTailSpec`` subclasses it and is per-request
         scratch, so it is excluded by type identity.
    A group that mixes target and drafter layers matches neither: flagging it
    would cost the target a whole scheduler page of hit on every lookup, so it
    is deliberately left alone.
    """
    ids: set[int] = set()
    for i, group in enumerate(kv_cache_config.kv_cache_groups):
        names = list(getattr(group, "layer_names", ()) or ())
        if names and all(
            any(m in str(n).lower() for m in ("mtp", "nextn", "draft", "eagle"))
            for n in names
        ):
            ids.add(i)
            continue
        spec = group.kv_cache_spec
        inner = getattr(spec, "kv_cache_specs", None)
        if isinstance(inner, dict) and inner:
            spec = next(iter(inner.values()))
        if type(spec).__name__ == "SlidingWindowSpec":
            ids.add(i)
    return ids


'''

E1_OLD = """        # Conservatively fall back to flag all groups when no group is flagged.
        if use_eagle and not self.eagle_group_ids:
            self.eagle_group_ids = set(range(len(kv_cache_config.kv_cache_groups)))
"""

E1_NEW = f"""        # Conservatively fall back to flag all groups when no group is flagged.
        if use_eagle and not self.eagle_group_ids:
            if _vsh_apc_align_enabled():  {MARK}
                # GLM-5-Next never reaches the annotator, so this fallback used
                # to flag EVERY group -- target MLA, padded KDA/Mamba and kpool
                # tail -- and each one dropped a scheduler page from the hit.
                self.eagle_group_ids = _vsh_apc_drafter_group_ids(kv_cache_config)
                logger.info(
                    "[vsh-glm53-apc-align] %d groups, no annotation: "
                    "eagle_group_ids=%s (upstream fallback would be all %d)",
                    len(kv_cache_config.kv_cache_groups),
                    sorted(self.eagle_group_ids),
                    len(kv_cache_config.kv_cache_groups),
                )
                if not self.eagle_group_ids:
                    logger.warning(
                        "[vsh-glm53-apc-align] no pure drafter KV cache group; "
                        "EAGLE last-block drop disabled"
                    )
            else:
                self.eagle_group_ids = set(
                    range(len(kv_cache_config.kv_cache_groups))
                )
"""

# the lookahead guard: re-assert it when the drop set can be empty
E2_OLD = """        if (
            enable_caching
            and self.eagle_group_ids
            and scheduler_block_size < num_prefill_lookahead
        ):
"""

E2_NEW = f"""        if (
            enable_caching
            and (self.eagle_group_ids or _vsh_apc_align_enabled())  {MARK}
            and scheduler_block_size < num_prefill_lookahead
        ):
"""

E3_OLD = """        # Propagate the eagle bit to each manager (default to ``use_eagle=False``).
        for group in self.attention_groups:
            if group.use_eagle:
                for gid in group.group_ids:
                    self.single_type_managers[gid].use_eagle = True
"""

E3_NEW = (
    E3_OLD
    + f"""
        if _vsh_apc_align_enabled():  {MARK}
            logger.info(
                "[vsh-glm53-apc-align] eagle=%s scheduler_block=%d align=%d "
                "partial_hash_hits=%s retention=%s groups=%s",
                sorted(self.eagle_group_ids),
                self.scheduler_block_size,
                self._cache_hit_alignment_tokens,
                self.enable_partial_hash_hits,
                self.retention_interval,
                [
                    (
                        type(g.spec).__name__,
                        tuple(g.group_ids),
                        g.manager_cls.__name__,
                        g.use_eagle,
                    )
                    for g in self.attention_groups
                ],
            )
"""
)


def main() -> int:
    text = P.read_text()
    if MARK in text:
        print("already patched")
        return 0
    if text.count(E1_OLD) != 1:
        raise SystemExit(f"E1 anchor count {text.count(E1_OLD)}")
    if text.count(E2_OLD) != 1:
        raise SystemExit(f"E2 anchor count {text.count(E2_OLD)}")
    if text.count(E3_OLD) != 1:
        raise SystemExit(f"E3 anchor count {text.count(E3_OLD)}")

    backup = P.with_suffix(f".py.bak-{time.strftime('%Y%m%d-%H%M%S')}")
    shutil.copy2(P, backup)

    # helper must be defined before the class that calls it
    anchor = "class KVCacheCoordinator(ABC):"
    if text.count(anchor) != 1:
        raise SystemExit(f"class anchor count {text.count(anchor)}")
    text = text.replace(anchor, HELPER + anchor, 1)

    text = text.replace(E1_OLD, E1_NEW, 1)
    text = text.replace(E2_OLD, E2_NEW, 1)
    text = text.replace(E3_OLD, E3_NEW, 1)

    compile(text, str(P), "exec")
    P.write_text(text)
    py_compile.compile(str(P), doraise=True, cfile="/tmp/_apc.pyc")
    print(f"vsh-glm53-apc-align installed; backup={backup.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
