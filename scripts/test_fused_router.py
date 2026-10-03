import sys, time, torch
sys.path.insert(0, "/home/davidcanar/claude_kp")
import vsh_w8a16 as W
from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import grouped_topk
dev = "cuda"; torch.manual_seed(0); bad_ids = 0; max_w = 0.0; cases = 0
bias = (torch.randn(288, device=dev) * 0.05).float()
for trial in range(400):
    m = [1, 3, 4, 8, 33][trial % 5]
    g = torch.randn(m, 288, device=dev) * 2
    if trial % 4 == 0: g = g.round(decimals=1)          # exact ties in the biased scores
    if trial % 7 == 0: g[:, 10] = g[:, 20]
    h = torch.empty(m, 4096, device=dev, dtype=torch.bfloat16)
    w0, i0 = grouped_topk(h, g, 8, True, 1, 1, "sigmoid", 2.5, bias)
    w1, i1 = W.sigmoid_bias_topk(g, bias, 8, True, 2.5)
    bad_ids += int(not torch.equal(i0.int(), i1)); max_w = max(max_w, (w0 - w1).abs().max().item()); cases += 1
print(f"ids mismatches {bad_ids}/{cases}, max |weight diff| {max_w:.2e}")
g = torch.randn(4, 288, device=dev); h = torch.empty(4, 4096, device=dev, dtype=torch.bfloat16)
for name, f in (("compiled grouped_topk", lambda: grouped_topk(h, g, 8, True, 1, 1, "sigmoid", 2.5, bias)),
                ("fused HIP", lambda: W.sigmoid_bias_topk(g, bias, 8, True, 2.5))):
    for _ in range(10): f()
    torch.cuda.synchronize(); t = time.time()
    for _ in range(300): f()
    torch.cuda.synchronize(); print(f"{name:22s} {(time.time()-t)/300*1e6:6.1f} us/call (wall, M=4)")
