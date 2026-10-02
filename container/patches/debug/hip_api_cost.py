import ctypes
import os
import time

import torch

# HIP_POINTER_ATTRIBUTE_DEVICE_POINTER from hip_runtime_api.h
HIP_POINTER_ATTRIBUTE_DEVICE_POINTER = 3

hip = ctypes.CDLL("libamdhip64.so")
hip.hipPointerGetAttribute.restype = ctypes.c_int
hip.hipPointerGetAttribute.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
]

t = torch.randn(4096, device="cuda")
p = ctypes.c_void_p(t.data_ptr())
out = ctypes.c_void_p(0)

# correctness of the call first
rc = hip.hipPointerGetAttribute(ctypes.byref(out), HIP_POINTER_ATTRIBUTE_DEVICE_POINTER, p)
print("rc=%d in=0x%x out=0x%x" % (rc, p.value or 0, out.value or 0))

N = 20000
t0 = time.perf_counter()
for _ in range(N):
    hip.hipPointerGetAttribute(ctypes.byref(out), HIP_POINTER_ATTRIBUTE_DEVICE_POINTER, p)
dt = time.perf_counter() - t0
print("hipPointerGetAttribute: %.2f us/call  (%d calls in %.3fs)" % (dt / N * 1e6, N, dt))

# control: an empty ctypes call round-trip
t0 = time.perf_counter()
for _ in range(N):
    pass
dt2 = time.perf_counter() - t0
print("python loop control:   %.2f us/iter" % (dt2 / N * 1e6))
