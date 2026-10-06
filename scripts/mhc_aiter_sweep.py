# aiter mhc_fused_post_pre_gemm_sqrsum tile configs on gfx1151 (DS4: hc 4, hidden 4096) + unfused post+pre at large M
import itertools, torch
import aiter.ops.mhc as m
torch.manual_seed(0); dev = "cuda"; H, n = 4096, 4; n3 = 24
def tm(fn, it=10):
    fn(); torch.cuda.synchronize(); best = 1e9
    for _ in range(3):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(it): fn()
        e.record(); torch.cuda.synchronize(); best = min(best, s.elapsed_time(e) / it)
    return best * 1e3
fn = torch.randn(n3, n * H, device=dev) * 0.01; hs = torch.ones(3, device=dev); hb = torch.randn(n3, device=dev) * 0.1
for M in (1, 6, 32, 128, 512, 1024, 2048, 4096):
    res = torch.randn(M, n, H, device=dev).bfloat16(); sub = torch.randn(M, H, device=dev).bfloat16()
    post = torch.rand(M, n, device=dev); comb = torch.rand(M, n, n, device=dev); proj = torch.empty(M, n, H, device=dev, dtype=torch.bfloat16)
    d = m.get_mhc_fused_post_pre_config(M, H)
    def run(sk, a, b, c):
        g = torch.empty(sk, M, 32, device=dev)[:, :, :n3]; s = torch.empty(sk, M, device=dev)
        return lambda: m.mhc_fused_post_pre_gemm_sqrsum(g, s, proj, sub, res, post, comb, fn, a, b, c, 0)
    t0 = tm(run(*d))
    out = []
    for sk, a, b, c in itertools.product((1, 2, 4, 8, 16, 32, 64), (16, 32, 64), (16, 32), (32, 64)):
        if H % (sk * c) or H // sk < 2 * c:
            continue
        try:
            out.append((tm(run(sk, a, b, c)), sk, a, b, c))
        except Exception:  # noqa: BLE001
            pass
    out.sort()
    line = f"M={M:5d}: default {d} {t0:9.1f} us | best " + "  ".join(f"({a},{b},{c},{e}) {t:.1f}" for t, a, b, c, e in out[:3])
    if M >= 512:   # aiter's unfused path for M >= 1024 on this arch: mhc_post + mhc_pre
        x = sub; pm = post.unsqueeze(-1)
        nxt = torch.empty_like(res)
        tu = tm(lambda: (m.mhc_post(nxt, x, res, pm, comb), m.mhc_pre(nxt, fn, hs, hb, 1e-6, 1e-6, 1e-6, 1.0, 20)))
        line += f" | unfused post+pre {tu:.1f} us"
    print(line, flush=True)
