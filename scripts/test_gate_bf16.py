import os, sys, time, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import vsh_w8a16 as w
torch.manual_seed(0)
for N, K in ((288, 4096), (256, 6144), (160, 2048)):
    W = (torch.randn(N, K, device="cuda") * 0.02).bfloat16()
    for M in (1, 2, 3, 4, 5, 8):
        xb = torch.randn(M + 2, K, device="cuda").bfloat16()
        x = xb[1:M + 1]
        ref = torch.mm(x, W.T, out_dtype=torch.float32)
        r64 = (x.double() @ W.double().T)
        got = w.w16_gemv(x, W)
        print(N, K, M, f"vs mm {(got-ref).abs().max().item():.2e}  vs fp64 {(got-r64).abs().max().item():.2e} (mm {(ref-r64).abs().max().item():.2e})")
W = (torch.randn(288, 4096, device="cuda") * 0.02).bfloat16(); x = torch.randn(4, 4096, device="cuda").bfloat16()
for name, fn in (("mm", lambda: torch.mm(x, W.T, out_dtype=torch.float32)), ("w16", lambda: w.w16_gemv(x, W))):
    for _ in range(20): fn()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(500): fn()
    torch.cuda.synchronize(); print(name, f"{(time.perf_counter()-t)/500*1e6:.1f} us/call (wall)")
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as pr:
        for _ in range(50): fn()
        torch.cuda.synchronize()
    ev = [e for e in pr.key_averages() if e.device_time_total > 0]
    print("   gpu:", ", ".join(f"{e.key[:40]} {e.device_time_total/e.count:.1f}us" for e in ev[:3]))
