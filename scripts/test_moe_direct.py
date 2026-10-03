# compare vsh direct decode MoE vs stock pipeline (moe_align + wna16 kernel + clamp-silu + moe_sum)
import os, sys, time, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import vsh_moe_int4 as vm
from vllm.model_executor.layers.fused_moe.fused_moe import invoke_fused_moe_wna16_triton_kernel, try_get_optimal_moe_config
from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
from vllm import _custom_ops as ops
import triton.language as tl
torch.manual_seed(0)
dev = "cuda"; E, H, I, TOPK, L = 64, 4096, 1024, 8, 10.0
w13 = torch.randint(0, 256, (E, 2 * I, H // 2), dtype=torch.uint8, device=dev)
w2 = torch.randint(0, 256, (E, H, I // 2), dtype=torch.uint8, device=dev)
s13 = (torch.rand(E, 2 * I, H // 128, device=dev) * 0.004 + 0.001).bfloat16()
s2 = (torch.rand(E, H, I // 128, device=dev) * 0.004 + 0.001).bfloat16()

def stock(x, tw, ids, hip):
    os.environ["VSH_MOE_INT4_HIP"] = "1" if hip else "0"
    M = x.size(0)
    cfg = try_get_optimal_moe_config(w13.size(), w2.size(), TOPK, "int4_w4a16", M, block_shape=[0, 128])
    c1 = torch.empty(M, TOPK, 2 * I, dtype=torch.bfloat16, device=dev)
    c2 = torch.empty(M * TOPK, I, dtype=torch.bfloat16, device=dev)
    c3 = torch.empty(M, TOPK, H, dtype=torch.bfloat16, device=dev)
    sid, eid, ntp = moe_align_block_size(ids, cfg["BLOCK_SIZE_M"], E, None)
    kw = dict(compute_type=tl.bfloat16, use_int8_w8a16=False, use_int4_w4a16=True, block_shape=[0, 128])
    invoke_fused_moe_wna16_triton_kernel(x, w13, c1, s13, None, None, sid, eid, ntp, False, TOPK, cfg, **kw)
    torch.ops._C.silu_and_mul_with_clamp(c2, c1.view(-1, 2 * I), L, 1.0, 0.0)
    invoke_fused_moe_wna16_triton_kernel(c2, w2, c3, s2, None, tw, sid, eid, ntp, True, 1, cfg, **kw)
    out = torch.empty(M, H, dtype=torch.bfloat16, device=dev)
    ops.moe_sum(c3, out)
    return out

worst = 0.0
for M in (1, 2, 3, 4, 5, 8):
    for trial in range(5):
        x = (torch.randn(M, H, device=dev) * 2).bfloat16()
        if trial % 2:   # force shared experts across tokens (spec-decode-like)
            base = torch.randperm(E, device=dev)[:TOPK]
            ids = torch.stack([base if (t % 2 == 0) else torch.randperm(E, device=dev)[:TOPK] for t in range(M)]).int()
        else:
            ids = torch.stack([torch.randperm(E, device=dev)[:TOPK] for _ in range(M)]).int()
        tw = torch.rand(M, TOPK, device=dev)
        ref = stock(x, tw, ids, True).float()
        tri = stock(x, tw, ids, False).float()
        got = vm.direct_moe(x, w13, w2, s13, s2, tw, ids, L).float()
        d = (got - ref).abs().max().item(); dt = (tri - ref).abs().max().item()
        rel = d / ref.abs().max().item()
        worst = max(worst, rel)
        if trial == 0: print(f"M={M} maxabs={d:.4g} (triton-vs-hip {dt:.4g}) rel={rel:.3g} exact={torch.equal(got, ref)}")
print("worst rel", worst)
# timing (wall per call incl. python)
x = torch.randn(4, H, device=dev).bfloat16(); ids = torch.stack([torch.randperm(E, device=dev)[:TOPK] for _ in range(4)]).int(); tw = torch.rand(4, TOPK, device=dev)
for name, fn in (("stock-hip", lambda: stock(x, tw, ids, True)), ("direct", lambda: vm.direct_moe(x, w13, w2, s13, s2, tw, ids, L))):
    for _ in range(10): fn()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(200): fn()
    torch.cuda.synchronize(); print(name, f"{(time.perf_counter() - t) / 200 * 1e6:.1f} us/call")
