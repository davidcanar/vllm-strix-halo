import ctypes, time, torch
L = ctypes.CDLL("/tmp/libw8sweep.so")
L.sweep.argtypes = [ctypes.c_int, ctypes.c_void_p] + [ctypes.c_void_p] * 4 + [ctypes.c_int] * 3 + [ctypes.c_long] * 2
names = ["R4V1", "R2V1", "R1V1", "R2V2", "R1V2", "R1V4", "R2V4", "R4V2"]
dev = "cuda"; torch.manual_seed(0)
# several copies so the working set is well beyond the 32 MB MALL
for N, K in ((12576, 4096), (4096, 8192)):
    Ws = [torch.randint(-127, 128, (N, K), device=dev, dtype=torch.int8) for _ in range(4)]
    Ss = [torch.rand(N, K // 128, device=dev) * 0.001 for _ in range(4)]
    for M in (1, 4):
        X = torch.randn(M, K, device=dev, dtype=torch.bfloat16); Y = torch.empty(M, N, device=dev, dtype=torch.bfloat16)
        ref = None; res = []
        for v, nm in enumerate(names):
            st = torch.cuda.current_stream().cuda_stream
            f = lambda i: L.sweep(v, st, Ws[i % 4].data_ptr(), Ss[i % 4].data_ptr(), X.data_ptr(), Y.data_ptr(), N, K, M, X.stride(0), Y.stride(0))
            for i in range(4): f(i)
            torch.cuda.synchronize()
            if v == 0: ref = Y.clone()
            ok = (Y.float() - ref.float()).abs().max().item() < 1e-2 * ref.float().abs().max().item() + 1e-3
            t = time.time()
            for i in range(40): f(i)
            torch.cuda.synchronize(); dt = (time.time() - t) / 40
            res.append(f"{nm} {N*K/dt/1e9:.0f}{'' if ok else '!'}")
        print(f"N={N} K={K} M={M} GB/s: " + "  ".join(res))
