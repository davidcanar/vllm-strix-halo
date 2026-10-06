# vsh_idx_logits vs aiter fp8_mqa_logits (DS4 prefill indexer): exactness of the selected top-k + speed sweep
import os, sys, time, itertools, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from vsh_idx_logits import vsh_idx_logits
except ImportError:   # installed copy
    from vllm.v1.attention.ops.vsh_idx_logits import vsh_idx_logits
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import rocm_fp8_mqa_logits
import vllm._custom_ops  # noqa: F401  (torch.ops._C)
torch.manual_seed(0); dev = "cuda"; R, H, D, TOPK = 512, 64, 128, 512
def tm(fn, it=5):
    fn(); torch.cuda.synchronize(); best = 1e9
    for _ in range(3):
        t = time.perf_counter()
        for _ in range(it): fn()
        torch.cuda.synchronize(); best = min(best, (time.perf_counter() - t) / it)
    return best * 1e3
def inputs(ctx, two_seqs=False):
    nk = ctx // 4
    q = (torch.randn(R, H, D, device=dev) * 0.5).to(torch.float8_e4m3fn)
    k = (torch.randn(nk, D, device=dev) * 0.5).to(torch.float8_e4m3fn)
    ks = torch.rand(nk, 1, device=dev) * 0.01 + 0.001
    w = torch.randn(R, H, device=dev) * 0.1
    ke = (torch.arange(R, device=dev, dtype=torch.int32) + ctx - R) // 4 + 1
    kst = torch.zeros(R, dtype=torch.int32, device=dev)
    if two_seqs:   # rows 0..299 belong to a first sequence (keys [0, 3000)), the rest to a second
        kst[300:] = 3000
        ke[:300] = torch.clamp(torch.arange(300, device=dev, dtype=torch.int32) // 4 + 2000, max=3000)
    return q, k, ks, w, kst, ke
# correctness: same logits (to fp32 rounding) and the same top-k sets
for ctx, two in ((16384, False), (32768, False), (16384, True)):
    q, k, ks, w, kst, ke = inputs(ctx, two)
    ref = rocm_fp8_mqa_logits(q, (k, ks), w, kst, ke)
    got = vsh_idx_logits(q, k, ks, w, kst, ke)
    fin = torch.isfinite(ref)
    same_mask = torch.equal(fin, torch.isfinite(got))
    rel = ((got[fin] - ref[fin]).norm() / ref[fin].norm()).item()
    i1 = torch.empty(R, TOPK, dtype=torch.int32, device=dev); i2 = torch.empty_like(i1)
    torch.ops._C.top_k_per_row_prefill(ref, kst, ke, i1, R, ref.stride(0), ref.stride(1), TOPK)
    torch.ops._C.top_k_per_row_prefill(got, kst, ke, i2, R, got.stride(0), got.stride(1), TOPK)
    s1 = torch.sort(i1, dim=1).values; s2 = torch.sort(i2, dim=1).values
    agree = (s1 == s2).float().mean().item()
    print(f"ctx {ctx//1024}K two_seqs={two}: same -inf mask {same_mask}, rel diff {rel:.1e}, top-k index agreement {agree*100:.2f} %", flush=True)
# speed sweep
for ctx in (32768, 131072):
    q, k, ks, w, kst, ke = inputs(ctx)
    fl = 2.0 * R * H * D * ke.float().mean().item()
    ta = tm(lambda: rocm_fp8_mqa_logits(q, (k, ks), w, kst, ke))
    res = []
    for bq, bk, split, nw in itertools.product((1, 2, 4), (32, 64, 128), (1, 2), (4, 8)):
        try:
            t = tm(lambda: vsh_idx_logits(q, k, ks, w, kst, ke, bq, bk, split, nw), it=3)
        except Exception as e:  # noqa: BLE001 - report compile failures and keep sweeping
            print(f"   bq{bq} bk{bk} s{split} w{nw}: {type(e).__name__} {str(e)[:80]}", flush=True); continue
        res.append((t, bq, bk, split, nw))
    res.sort()
    print(f"ctx {ctx//1024}K: aiter {ta:.2f} ms ({fl/ta/1e9:.1f} TFLOPS) | best vsh: " +
          "  ".join(f"bq{b} bk{c} s{s} w{n} {t:.2f} ms ({fl/t/1e9:.1f} TF, {ta/t:.1f}x)" for t, b, c, s, n in res[:4]), flush=True)
