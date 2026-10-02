import time

import torch

torch.manual_seed(0)
dev = "cuda"


def bench(m, k, n, iters=200, dtype=torch.bfloat16):
    x = torch.randn(m, k, device=dev, dtype=dtype)
    w = torch.randn(n, k, device=dev, dtype=dtype)
    for _ in range(20):
        torch.nn.functional.linear(x, w)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        torch.nn.functional.linear(x, w)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    bytes_moved = (m * k + k * n) * 2  # bf16: weights + activations
    return dt, bytes_moved / dt / 1e9


shapes = [
    ("KDA in_proj_qkvbfg_a", 12576, 4096),
    ("KDA o_proj", 4096, 4096),
    ("MLA fused_qkv_a", 2048, 4096),
    ("lm_head", 154880, 4096),
]
print(f"{'shape':22s} {'M':>3s} {'ms':>8s} {'GB/s':>8s}")
for name, n, k in shapes:
    for m in (1, 2, 4, 8):
        dt, gbs = bench(m, k, n, iters=200 if m < 8 else 100)
        print(f"{name:22s} {m:3d} {dt * 1000:8.3f} {gbs:8.1f}")

print("\nrow-sum of bf16 weight bytes read per decode step (M=4, per rank):")
total = 0
for name, n, k in shapes:
    b = n * k * 2
    total += b
    print(f"  {name:22s} {n:6d} x {k:5d} x 2B = {b / 1e9:5.2f} GB")
print(f"  (single read of each; the model has 34 KDA + 11 MLA layers)")
kda = 34 * (12576 * 4096 + 4096 * 4096) * 2
mla = 11 * (2048 * 4096 + 4096 * 4096) * 2
lm = 154880 * 4096 * 2
print(f"  KDA layers  x34: {kda / 1e9:5.2f} GB")
print(f"  MLA layers  x11: {mla / 1e9:5.2f} GB")
print(f"  lm_head     x1 : {lm / 1e9:5.2f} GB")
print(f"  TOTAL          : {(kda + mla + lm) / 1e9:5.2f} GB per rank per step")
