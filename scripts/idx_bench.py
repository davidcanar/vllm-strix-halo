# DS4 prefill indexer ops at chunk=512 rows: aiter fp8_mqa_logits + vLLM top_k_per_row_prefill, by context
import time, torch
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import rocm_fp8_mqa_logits
import vllm._custom_ops  # noqa: registers torch.ops._C
torch.manual_seed(0); dev = "cuda"; R, H, D, TOPK = 512, 64, 128, 512
def tm(fn, it=5):
    fn(); torch.cuda.synchronize(); best = 1e9
    for _ in range(3):
        t = time.perf_counter()
        for _ in range(it): fn()
        torch.cuda.synchronize(); best = min(best, (time.perf_counter() - t) / it)
    return best * 1e3
for ctx in (16384, 32768, 65536, 131072):
    nk = ctx // 4
    q = (torch.randn(R, H, D, device=dev) * 0.5).to(torch.float8_e4m3fn)
    k = (torch.randn(nk, D, device=dev) * 0.5).to(torch.float8_e4m3fn)
    ks = torch.rand(nk, 1, device=dev) * 0.01 + 0.001
    w = torch.randn(R, H, device=dev) * 0.1
    ke = (torch.arange(R, device=dev, dtype=torch.int32) + ctx - R) // 4 + 1   # causal: compressed entries visible
    kst = torch.zeros(R, dtype=torch.int32, device=dev)
    idx = torch.empty(R, TOPK, dtype=torch.int32, device=dev)
    t_l = tm(lambda: rocm_fp8_mqa_logits(q, (k, ks), w, kst, ke))
    lg = rocm_fp8_mqa_logits(q, (k, ks), w, kst, ke)
    t_k = tm(lambda: torch.ops._C.top_k_per_row_prefill(lg, kst, ke, idx, R, lg.stride(0), lg.stride(1), TOPK))
    fl = 2.0 * R * H * D * ke.float().mean().item()
    print(f"ctx {ctx//1024:4d}K (Nk {nk:6d}): logits {t_l:7.2f} ms ({fl/t_l/1e9:5.1f} TFLOPS)  top-k {t_k:6.2f} ms  "
          f"-> x21 layers per 512-token chunk: {21*(t_l+t_k)/1e3:5.2f} s", flush=True)
