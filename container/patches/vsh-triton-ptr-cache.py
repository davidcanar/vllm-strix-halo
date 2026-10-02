#!/usr/bin/env python3
"""vsh-triton-ptr-cache: memoise Triton/AMD's ptr -> device_ptr lookup.

Where the time goes (root-caused 2026-10-02 with gdb on the live worker):

    hipPointerGetAttribute
      <- extractPointer()   triton/backends/amd/driver.c  (compiled into
      <- launchKernel()      ~/.triton/cache/<hash>/hip_utils...so)

`extractPointer` runs once per **pointer argument** of every Triton kernel
launch (~5.6 pointers/launch on this model, 253 launches/step => ~1430 calls
per decode step), and each call costs ~6 us: 8.7 ms/step, measured as
`hipPointerGetAttribute` self-time in the torch profiler. The driver resolves
the symbol through `hipGetProcAddress`, so an LD_PRELOAD interposer cannot see
it -- only the source can be patched.

The mapping is stable for the life of an allocation (device pointers map to
themselves; the runtime's own table is address-keyed), so the result is cached
here. Semantics are preserved: an address that was never valid is never cached,
so the "cpu tensor?" ValueError path is unchanged.

Gated on `VSH_TRITON_PTR_CACHE=1`, read once per process; unset/0 leaves the
generated code path byte-identical to stock.

Idempotent, fail-closed, backed up. Triton recompiles the module on next import
(the cache key includes the source hash), so no .so surgery is needed.
"""
from __future__ import annotations

import pathlib
import shutil
import time

P = pathlib.Path(
    "/opt/venv/lib/python3.12/site-packages/triton/backends/amd/driver.c"
)
MARK = "[vsh-triton-ptr-cache]"

HELPER = f"""// {MARK} ---------------------------------------------------------------
// Memoised hipPointerGetAttribute(HIP_POINTER_ATTRIBUTE_DEVICE_POINTER) for
// kernel-launch pointer arguments. See container/patches/ for the analysis.
#include <stdint.h>
#include <string.h>

#define VSH_PTR_CACHE_SLOTS 16384
static struct {{
  const void *key;
  void *val;
}} vsh_ptr_cache[VSH_PTR_CACHE_SLOTS];
static int vsh_ptr_cache_state = -1;

static inline int vsh_ptr_cache_enabled(void) {{
  if (vsh_ptr_cache_state < 0) {{
    const char *e = getenv("VSH_TRITON_PTR_CACHE");
    vsh_ptr_cache_state = (e && e[0] == '1') ? 1 : 0;
  }}
  return vsh_ptr_cache_state;
}}

static inline size_t vsh_ptr_cache_hash(const void *key) {{
  return (size_t)((((uintptr_t)key) >> 4) * 2654435761u) &
         (VSH_PTR_CACHE_SLOTS - 1);
}}

static inline void *vsh_ptr_cache_get(const void *key) {{
  size_t i = vsh_ptr_cache_hash(key);
  for (int n = 0; n < 16; n++) {{
    size_t j = (i + (size_t)n) & (VSH_PTR_CACHE_SLOTS - 1);
    const void *k = vsh_ptr_cache[j].key;
    if (k == key)
      return vsh_ptr_cache[j].val;
    if (k == NULL)
      return NULL;
  }}
  return NULL;
}}

static inline void vsh_ptr_cache_put(const void *key, void *val) {{
  size_t i = vsh_ptr_cache_hash(key);
  for (int n = 0; n < 16; n++) {{
    size_t j = (i + (size_t)n) & (VSH_PTR_CACHE_SLOTS - 1);
    const void *k = vsh_ptr_cache[j].key;
    if (k == key || k == NULL) {{
      vsh_ptr_cache[j].key = key;
      vsh_ptr_cache[j].val = val;
      return;
    }}
  }}
  // Table full: fall through, the caller recomputes next time.
}}
// {MARK} ---------------------------------------------------------------

static PyObject *data_ptr_str = NULL;
"""

OLD_LOOKUP = """  if (*dev_ptr == 0) {
    return true; // valid nullptr
  }
  hipError_t status = hipSymbolTable.hipPointerGetAttribute(
      dev_ptr, HIP_POINTER_ATTRIBUTE_DEVICE_POINTER, *dev_ptr);
  if (status == hipErrorInvalidValue) {
"""

NEW_LOOKUP = f"""  if (*dev_ptr == 0) {{
    return true; // valid nullptr
  }}
  hipDeviceptr_t vsh_in = *dev_ptr;  // {MARK}
  if (vsh_ptr_cache_enabled()) {{  // {MARK}
    void *vsh_hit = vsh_ptr_cache_get((const void *)vsh_in);
    if (vsh_hit) {{
      *dev_ptr = (hipDeviceptr_t)vsh_hit;
      return true;
    }}
  }}
  hipError_t status = hipSymbolTable.hipPointerGetAttribute(
      dev_ptr, HIP_POINTER_ATTRIBUTE_DEVICE_POINTER, *dev_ptr);
  if (status == hipErrorInvalidValue) {{
"""

OLD_TAIL = """    // Clear and ignore HIP error
    (void)hipSymbolTable.hipGetLastError();
    return false;
  }
  return true;
}
"""

NEW_TAIL = f"""    // Clear and ignore HIP error
    (void)hipSymbolTable.hipGetLastError();
    return false;
  }}
  if (vsh_ptr_cache_enabled() && *dev_ptr) {{  // {MARK}
    vsh_ptr_cache_put((const void *)vsh_in, (void *)*dev_ptr);
  }}
  return true;
}}
"""


def main() -> int:
    text = P.read_text()
    if MARK in text:
        print("already patched")
        return 0
    for label, old in (("lookup", OLD_LOOKUP), ("tail", OLD_TAIL), ("anchor", "static PyObject *data_ptr_str = NULL;")):
        n = text.count(old)
        if n != 1:
            raise SystemExit(f"{label}: anchor count {n}")

    backup = P.with_suffix(f".bak-{time.strftime('%Y%m%d-%H%M%S')}")
    shutil.copy2(P, backup)

    text = text.replace("static PyObject *data_ptr_str = NULL;", HELPER.rstrip("\n"), 1)
    text = text.replace(OLD_LOOKUP, NEW_LOOKUP, 1)
    text = text.replace(OLD_TAIL, NEW_TAIL, 1)
    P.write_text(text)
    print(f"triton driver.c patched; backup={backup.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
