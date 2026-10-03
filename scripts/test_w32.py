import sys, time, torch
sys.path.insert(0, "/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers")
import vsh_w8a16 as W
dev = "cuda"; torch.manual_seed(0)
Ws = [torch.randn(288, 4096, device=dev) * 0.02 for _ in range(42)]   # 42 layers -> DRAM-resident like in model
for m in (1, 3, 4, 8):
    x = torch.randn(m, 4096, device=dev, dtype=torch.bfloat16)
    ref = torch.nn.functional.linear(x.float(), Ws[0]); y = W.w32_gemv(x, Ws[0])
    err = ((y - ref).abs().max() / ref.abs().max()).item()
    def bench(f):
        for i in range(3): f(i)
        torch.cuda.synchronize(); t = time.time()
        for i in range(84): f(i)
        torch.cuda.synchronize(); return (time.time() - t) / 84 * 1e6
    t0 = bench(lambda i: torch.nn.functional.linear(x.float(), Ws[i % 42]))
    t1 = bench(lambda i: W.w32_gemv(x, Ws[i % 42]))
    print(f"M={m}: max rel err {err:.1e}  stock {t0:.0f} us  gemv {t1:.0f} us  ({t0/t1:.1f}x)")
