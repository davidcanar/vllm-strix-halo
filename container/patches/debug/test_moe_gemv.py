import sys, time, torch
sys.path.insert(0, "/home/davidcanar/claude_kp")
import vsh_moe_int4_gemv as G
import vsh_moe_int4 as H
from vllm.model_executor.layers.fused_moe.fused_moe import invoke_fused_moe_wna16_triton_kernel as orig
from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
from vllm.triton_utils import tl
torch.manual_seed(0); dev = "cuda"; E = 48; GS = 128
cfg = {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 1, "SPLIT_K": 1, "num_warps": 2, "num_stages": 2}
def mk(N, K):
    B = torch.randint(0, 256, (E, N, K // 2), device=dev, dtype=torch.uint8)
    S = (torch.rand(E, N, K // GS, device=dev) * 0.02 + 0.001).to(torch.bfloat16)
    return B, S
def bench(f, n=30):
    for _ in range(3): f()
    torch.cuda.synchronize(); t = time.time()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.time() - t) / n * 1e3
for M in (1, 4, 8):
    topk_ids = torch.stack([torch.randperm(E, device=dev)[:8] for _ in range(M)]).to(torch.int32)
    topk_w = torch.rand(M, 8, device=dev)
    sid, eid, ntpp = moe_align_block_size(topk_ids, cfg["BLOCK_SIZE_M"], E)[:3]
    for name, N, K, rows, tk, mul in (("w13", 2048, 4096, M, 8, False), ("w2 ", 4096, 1024, M * 8, 1, True)):
        B, S = mk(N, K)
        A = torch.randn(rows, K, device=dev, dtype=torch.bfloat16)
        C1 = torch.zeros(M, 8, N, device=dev, dtype=torch.bfloat16); C2 = torch.zeros_like(C1)
        f1 = lambda: orig(A, B, C1, S, None, topk_w, sid, eid, ntpp, mul, tk, cfg, tl.bfloat16, False, True, [0, GS])
        f2 = lambda: G.moe_int4_gemv(A, B, C2, S, topk_w, sid, eid, ntpp, mul, tk, cfg, [0, GS])
        C3 = torch.zeros_like(C1)
        f3 = lambda: H.moe_int4_gemv(A, B, C3, S, topk_w.reshape(-1).contiguous(), sid, eid, ntpp, mul, tk, cfg, [0, GS])
        f1(); f3(); torch.cuda.synchronize()
        rel = ((C1.float() - C3.float()).norm() / C1.float().norm()).item()
        t1, t2 = bench(f1), bench(f3)
        uniq = len(set(topk_ids.flatten().tolist())); mb = uniq * N * (K // 2 + K // GS * 2) / 1e6
        print(f"M={M} {name} rel_err {rel:.2e}  stock {t1:.3f} ms ({mb/t1:.0f} GB/s)  HIP {t2:.3f} ms ({mb/t2:.0f} GB/s)  {t1/t2:.2f}x  can_use={H.can_use(A, B, None, [0, GS], tk, cfg, C3)}")
