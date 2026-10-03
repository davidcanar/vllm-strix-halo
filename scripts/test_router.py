import torch, time
def old(x, k):
    return x.sort(dim=-1, descending=True, stable=True).indices[..., :k]
def new(x, k):
    xi = x.float().contiguous().view(torch.int32).to(torch.int64)
    ordk = torch.where(xi >= 0, xi, xi ^ 0x7FFFFFFF)
    idx = torch.arange(x.shape[-1], device=x.device, dtype=torch.int64)
    return torch.topk(ordk * 65536 + (65535 - idx), k, dim=-1, sorted=True).indices
dev = "cuda"; torch.manual_seed(0); bad = 0
for trial in range(300):
    m = [1, 2, 4, 8, 740][trial % 5]
    x = torch.sigmoid(torch.randn(m, 288, device=dev)) + torch.randn(288, device=dev) * 0.01
    if trial % 3 == 0:                      # inject exact ties and -inf masks
        x[:, 17] = x[:, 200]; x[:, 5] = x[:, 6]; x[:, ::7] = float("-inf")
    if trial % 4 == 0: x = x.round(decimals=2)
    for k in (1, 8):
        a, b = old(x, k), new(x, k)
        bad += int(not torch.equal(a, b))
print("mismatches:", bad, "of", 300 * 2)
x = torch.rand(4, 288, device=dev)
for f in (old, new):
    g = torch.compile(f, dynamic=True)
    for _ in range(5): g(x, 8)
    torch.cuda.synchronize(); t = time.time()
    for _ in range(200): g(x, 8)
    torch.cuda.synchronize(); print(f.__name__, "%.1f us/call (compiled, M=4)" % ((time.time() - t) / 200 * 1e6))
