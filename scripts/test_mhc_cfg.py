# aiter mHC with vs without the gfx1151 config/crossover install: same outputs, timing (DS4 shapes, sinkhorn tail included)
import os, re, sys, time, torch
import aiter.ops.mhc as m
here = os.path.dirname(os.path.abspath(__file__))
cands = [os.path.join(here, "vsh-ds4-mhc-cfg.py"), os.path.join(here, "..", "container", "patches", "vsh-ds4-mhc-cfg.py")]
src = open(next(c for c in cands if os.path.exists(c))).read()
code = re.search(r"install = '''(.*?)'''", src, re.S).group(1)
ns = {"torch": torch}; exec(code.replace("\n_vsh_mhc_cfg_install()\n", "\n"), ns)
torch.manual_seed(0); dev = "cuda"; H, n = 4096, 4; n3 = 24
fn = torch.randn(n3, n * H, device=dev) * 0.01; hs = torch.ones(3, device=dev); hb = torch.randn(n3, device=dev) * 0.1
nw = torch.rand(H, device=dev).bfloat16() + 0.5
def tm(f, it=10):
    f(); torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(it): f()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / it * 1e3
def rel(a, b):
    a, b = a.float(), b.float(); return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()
cases = []
for M in (1, 6, 64, 128, 384, 512, 2048):
    x = torch.randn(M, H, device=dev).bfloat16(); res = torch.randn(M, n, H, device=dev).bfloat16()
    post = torch.rand(M, n, 1, device=dev); comb = torch.softmax(torch.randn(M, n, n, device=dev), -1)
    cases.append((M, (x, res, post, comb, fn, hs, hb, 1e-6, 1e-6, 1e-6, 1.0, 20, nw, 1e-6)))
ref = {M: [t.clone() for t in m.mhc_fused_post_pre(*a)] for M, a in cases}
tref = {M: tm(lambda: m.mhc_fused_post_pre(*a)) for M, a in cases}
pre_ref = m.mhc_pre(cases[5][1][1], fn, hs, hb, 1e-6, 1e-6, 1e-6, 1.0, 20, nw, 1e-6)
ns["_vsh_mhc_cfg_install"]()
assert getattr(m, "_vsh_mhc_cfg", False), "install did not run"
for M, a in cases:
    g = m.mhc_fused_post_pre(*a); t = tm(lambda: m.mhc_fused_post_pre(*a))
    errs = " ".join(f"{nm} {rel(gg, rr):.1e}" for nm, gg, rr in zip(("post", "comb", "layer_in", "next_res"), g, ref[M]))
    print(f"fused_post_pre M={M:5d}: {errs} | before {tref[M]:7.3f} ms  after {t:7.3f} ms -> {tref[M]/t:4.1f}x", flush=True)
g = m.mhc_pre(cases[5][1][1], fn, hs, hb, 1e-6, 1e-6, 1e-6, 1.0, 20, nw, 1e-6)
print("mhc_pre M=512: " + " ".join(f"{nm} {rel(gg, rr):.1e}" for nm, gg, rr in zip(("post", "comb", "layer_in"), g, pre_ref)), flush=True)
