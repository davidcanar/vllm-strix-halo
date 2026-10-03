import sys, time, types, torch
sys.path.insert(0, "/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers")
import vsh_w8a16 as W
from vllm.model_executor.layers.utils import rocm_unquantized_gemm_impl as stock
torch.manual_seed(0); dev = "cuda"
def bench(f, n=50):
    for _ in range(3): f()
    torch.cuda.synchronize(); t = time.time()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.time() - t) / n * 1e3
for name, n, k in (("kda in_proj", 12576, 4096), ("kda o_proj", 4096, 4096), ("mla o_proj", 4096, 8192),
                   ("mla q_b", 8192, 1536), ("shared gate_up", 2048, 4096), ("dense down", 4096, 6144), ("qkv_a", 2048, 4096)):
    w = (torch.randn(n, k, device=dev) * 0.02).to(torch.bfloat16)
    layer = types.SimpleNamespace(weight=torch.nn.Parameter(w.clone(), requires_grad=False), prefix="x")
    W.quantize_(layer)
    wdq = torch.empty((n, k), dtype=torch.bfloat16, device=dev)
    W._lib().vsh_w8_dequant(torch.cuda.current_stream().cuda_stream, layer.vsh_w8_q.data_ptr(), layer.vsh_w8_s.data_ptr(), wdq.data_ptr(), n, k)
    for m in (1, 4, 8):
        x = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
        y = W.apply(layer, x, None)
        ref_dq = torch.nn.functional.linear(x.float(), wdq.float())
        ref_bf = torch.nn.functional.linear(x.float(), w.float())
        e_k = ((y.float() - ref_dq).norm() / ref_dq.norm()).item()
        e_q = ((y.float() - ref_bf).norm() / ref_bf.norm()).item()
        t_s = bench(lambda: stock(x, w, None)); t_w = bench(lambda: W.apply(layer, x, None))
        print(f"{name:15s} N={n:5d} K={k:4d} M={m}: kernel err {e_k:.1e} quant err {e_q:.1e} | bf16 stock {t_s:.3f} ms ({n*k*2/t_s/1e6:.0f} GB/s)  w8 {t_w:.3f} ms ({n*k/t_w/1e6:.0f} GB/s)  {t_s/t_w:.2f}x")
    xp = torch.randn(300, k, device=dev, dtype=torch.bfloat16)
    yp = W.apply(layer, xp, None); rp = torch.nn.functional.linear(xp.float(), wdq.float())
    print(f"{'':15s} prefill M=300 path err {((yp.float()-rp).norm()/rp.norm()).item():.1e}")
