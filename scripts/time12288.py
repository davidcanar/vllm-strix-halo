# FP8 GEMV at K=12288 (DSpark drafter fc: 4096 x 12288, row stride 12544), best-of-5
import os, sys, time, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import vsh_w8a16 as w8
torch.manual_seed(0)
full = torch.empty(4096, 12288 + 256, dtype=torch.float8_e4m3fn, device="cuda"); w = full[:, :12288]
w.copy_((torch.randn(4096, 12288, device="cuda") * 0.5).to(torch.float8_e4m3fn))
s = (torch.rand(32, 96, device="cuda") * 0.02 + 0.002)
for M in (1, 2, 3, 4, 6):
    x = torch.randn(M, 12288, device="cuda").bfloat16()
    best = 1e9
    for rep in range(5):
        for _ in range(3): w8.fp8_gemv(x, w, s)
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(20): w8.fp8_gemv(x, w, s)
        torch.cuda.synchronize(); best = min(best, (time.perf_counter() - t) / 20 * 1e6)
    print(f"K=12288 M={M}: best {best:.0f} us ({4096*12288/best/1e3:.0f} GB/s)")
