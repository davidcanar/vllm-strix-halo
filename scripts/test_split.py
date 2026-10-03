import sys, time, torch
sys.path.insert(0, "/opt/venv/lib/python3.12/site-packages/vllm/v1/attention/ops")
from vsh_sparse_attn_split import sparse_attn_ragged_split
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import _rocm_sparse_attn_prefill_ragged_triton as orig
torch.manual_seed(0); dev = "cuda"
for nq, nsel, sink_on in ((4, 2048, False), (4, 2048, True), (1, 2048, False), (4, 900, False), (8, 2051, True)):
    nh, d = 32, 512
    q = torch.randn(nq, nh, d, device=dev, dtype=torch.bfloat16)
    kv = torch.randn(60000, d, device=dev, dtype=torch.bfloat16)
    idx = torch.randint(0, 60000, (nq, nsel), device=dev, dtype=torch.int32)
    idx[:, ::37] = -1                                   # padded / invalid entries
    indices = idx.reshape(-1).contiguous()
    indptr = torch.arange(0, nq * nsel + 1, nsel, device=dev, dtype=torch.int32)
    sink = torch.randn(nh, device=dev, dtype=torch.float32) if sink_on else None
    sc = 1.0 / (d ** 0.5)
    a = orig(q, kv, indices, indptr, sc, sink, 512, 0); b = sparse_attn_ragged_split(q, kv, indices, indptr, sc, sink)
    err = (a.float() - b.float()).abs().max().item()
    ts = []
    for f in (lambda: orig(q, kv, indices, indptr, sc, sink, 512, 0), lambda: sparse_attn_ragged_split(q, kv, indices, indptr, sc, sink)):
        for _ in range(3): f()
        torch.cuda.synchronize(); t = time.time()
        for _ in range(50): f()
        torch.cuda.synchronize(); ts.append((time.time() - t) / 50 * 1e3)
    print(f"nq={nq} sel={nsel} sink={sink_on}: max abs diff {err:.2e}  orig {ts[0]:.3f} ms  split {ts[1]:.3f} ms  ({ts[0]/ts[1]:.1f}x)")
