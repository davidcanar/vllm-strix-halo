# GPU time (events) of the full aiter mhc_fused_post_pre at decode sizes, default vs gfx1151 config
import os, re, torch
import aiter.ops.mhc as m
here = os.path.dirname(os.path.abspath(__file__))
cands = [os.path.join(here, "vsh-ds4-mhc-cfg.py"), os.path.join(here, "..", "container", "patches", "vsh-ds4-mhc-cfg.py")]
src = open(next(c for c in cands if os.path.exists(c))).read()
code = re.search(r"install = '''(.*?)'''", src, re.S).group(1)
ns = {"torch": torch}; exec(code.replace("\n_vsh_mhc_cfg_install()\n", "\n"), ns)
torch.manual_seed(0); dev = "cuda"; H, n = 4096, 4; n3 = 24
fn = torch.randn(n3, n * H, device=dev) * 0.01; hs = torch.ones(3, device=dev); hb = torch.randn(n3, device=dev) * 0.1
nw = torch.rand(H, device=dev).bfloat16() + 0.5
def tg(f, it=200):
    for _ in range(20): f()
    torch.cuda.synchronize(); best = 1e9
    for _ in range(5):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(it): f()
        e.record(); torch.cuda.synchronize(); best = min(best, s.elapsed_time(e) / it)
    return best * 1e3
cases = {}
for M in (1, 2, 6, 8, 16, 32):
    x = torch.randn(M, H, device=dev).bfloat16(); res = torch.randn(M, n, H, device=dev).bfloat16()
    post = torch.rand(M, n, 1, device=dev); comb = torch.softmax(torch.randn(M, n, n, device=dev), -1)
    cases[M] = (x, res, post, comb, fn, hs, hb, 1e-6, 1e-6, 1e-6, 1.0, 20, nw, 1e-6)
before = {M: tg(lambda: m.mhc_fused_post_pre(*a)) for M, a in cases.items()}
ns["_vsh_mhc_cfg_install"]()
for M, a in cases.items():
    after = tg(lambda: m.mhc_fused_post_pre(*a))
    print(f"M={M:3d}: stream time per call before {before[M]:6.1f} us  after {after:6.1f} us  ({before[M]/after:.2f}x)", flush=True)
