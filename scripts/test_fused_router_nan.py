import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import vsh_w8a16 as w
E, K = 288, 8
bias = torch.randn(E, device="cuda") * 0.1
for name, lg in (("all-nan", torch.full((8, E), float("nan"), device="cuda")),
                 ("some-nan", torch.where(torch.rand(8, E, device="cuda") < 0.3, float("nan"), torch.randn(8, E, device="cuda"))),
                 ("inf", torch.where(torch.rand(8, E, device="cuda") < 0.3, float("inf"), torch.randn(8, E, device="cuda"))),
                 ("garbage", torch.empty(8, E, device="cuda").uniform_(-1.5e38, 1.5e38))):
    wt, ids = w.sigmoid_bias_topk(lg, bias, K, True, 2.5)
    ok = bool(((ids >= 0) & (ids < E)).all()) and all(len(set(r.tolist())) == K for r in ids)
    print(name, "valid+distinct" if ok else "BAD", ids[0].tolist())
lg = torch.randn(64, E, device="cuda")
wt, ids = w.sigmoid_bias_topk(lg, bias, K, True, 2.5)
s = lg.sigmoid(); ref = torch.topk(s + bias, K, dim=-1).indices
print("finite inputs: ids match topk (as sets):", all(set(a.tolist()) == set(b.tolist()) for a, b in zip(ids, ref)))
