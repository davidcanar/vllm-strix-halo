import ctypes, os, sys, time, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import vsh_w8a16 as w8
lib = ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), "libfp8sweep.so"))
lib.sweep.restype = ctypes.c_int
lib.sweep.argtypes = [ctypes.c_int] + [ctypes.c_void_p] * 5 + [ctypes.c_int] * 3 + [ctypes.c_long] * 4
names = ["R1V4", "R1V2", "R2V2", "R2V1", "R4V1", "R2V4", "R4V2"]
torch.manual_seed(0)
for (N, K) in ((16384, 1024), (4096, 4096), (2048, 4096), (1536, 4096), (4096, 1024)):
    full = torch.empty(N, K + 256, dtype=torch.float8_e4m3fn, device="cuda"); w = full[:, :K]
    w.copy_((torch.randn(N, K, device="cuda") * 0.5).to(torch.float8_e4m3fn))
    s = torch.rand(N // 128, K // 128, device="cuda") * 0.02
    for M in (1, 4, 6):
        x = torch.randn(M, K, device="cuda").bfloat16()
        ref = w8.fp8_gemv(x, w, s).float()
        res = []
        for vi, nm in enumerate(names):
            y = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
            st = torch.cuda.current_stream().cuda_stream
            call = lambda: lib.sweep(vi, st, w.data_ptr(), s.data_ptr(), x.data_ptr(), y.data_ptr(), N, K, M, w.stride(0), s.stride(0), x.stride(0), y.stride(0))
            if call() != 0: continue
            torch.cuda.synchronize()
            err = ((y.float() - ref).norm() / ref.norm()).item()
            for _ in range(3): call()
            torch.cuda.synchronize(); t = time.perf_counter()
            for _ in range(50): call()
            torch.cuda.synchronize(); us = (time.perf_counter() - t) / 50 * 1e6
            res.append((us, nm, err))
        t0 = (lambda: w8.fp8_gemv(x, w, s))
        for _ in range(3): t0()
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(50): t0()
        torch.cuda.synchronize(); cur = (time.perf_counter() - t) / 50 * 1e6
        best = min(res)
        print(f"N={N:5d} K={K} M={M}: current {cur:6.1f} us | best {best[1]} {best[0]:6.1f} us ({N*K/best[0]/1e3:4.0f} GB/s, err {best[2]:.1e}) | " + " ".join(f"{nm}={us:.0f}" for us, nm, _ in sorted(res, key=lambda r: names.index(r[1]))))
