# aiter mHC unfused pieces on gfx1151: mhc_post, mhc_pre (default) and its gemm stage by split-k; fused best near the crossover
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
for M in (64, 128, 256, 384, 512, 2048):
    res = torch.randn(M, n, H, device=dev).bfloat16(); sub = torch.randn(M, H, device=dev).bfloat16()
    post = torch.rand(M, n, 1, device=dev); comb = torch.rand(M, n, n, device=dev); nxt = torch.empty_like(res)
    tp = tm(lambda: m.mhc_post(nxt, sub, res, post, comb))
    tpre = tm(lambda: m.mhc_pre(nxt, fn, hs, hb, 1e-6, 1e-6, 1e-6, 1.0, 20))
    dsk = m.get_mhc_pre_splitk(M, n * H)
    best = []
    for sk, tk in itertools.product((1, 2, 4, 8, 16, 32, 64), (32, 64)):
        if (n * H) % (sk * tk) or (n * H) // sk < 2 * tk:
            continue
        g = torch.empty(sk, M, 32, device=dev)[:, :, :n3]; sq = torch.empty(sk, M, device=dev)
        try:
            best.append((tm(lambda: m.mhc_pre_gemm_sqrsum(g, sq, nxt, fn, tk, 0)), sk, tk))
        except Exception:  # noqa: BLE001
            pass
    best.sort()
    g = torch.empty(dsk[0], M, 32, device=dev)[:, :, :n3]; sq = torch.empty(dsk[0], M, device=dev)
    tg = tm(lambda: m.mhc_pre_gemm_sqrsum(g, sq, nxt, fn, dsk[1], 0))
    fz = []
    for sk, a, b, c in ((8, 16, 16, 32), (16, 16, 16, 32), (16, 32, 16, 32), (32, 16, 16, 32), (8, 16, 32, 32), (16, 16, 32, 32)):
        gg = torch.empty(sk, M, 32, device=dev)[:, :, :n3]; ss = torch.empty(sk, M, device=dev)
        fz.append((tm(lambda: m.mhc_fused_post_pre_gemm_sqrsum(gg, ss, nxt, sub, res, post.squeeze(-1), comb, fn, a, b, c, 0)), sk, a, b, c))
    fz.sort()
    print(f"M={M:5d}: post {tp:7.1f} + pre {tpre:7.1f} = {tp+tpre:7.1f} us | pre gemm stage default {dsk} {tg:.1f} us, best "
          + " ".join(f"(sk{a},tk{b}) {t:.1f}" for t, a, b in best[:3])
          + f" | fused best ({fz[0][1]},{fz[0][2]},{fz[0][3]},{fz[0][4]}) {fz[0][0]:.1f} us + tail", flush=True)
