"""Unit test: prefill indexer path (paged gather + rocm_fp8_mqa_logits) on the real kpool layout."""
import torch
from vllm.models.glm5next.amd.ops import kpool_compress as kc
torch.manual_seed(0); dev = "cuda"
PAGE, D, H, KPOOL, NUM_PAGES, n_pools, M = 64, 128, 32, 4, 40, 1500, 8
bt_pages = torch.randperm(NUM_PAGES, device=dev)[: (n_pools + PAGE - 1) // PAGE].to(torch.int32)
kv = torch.zeros((NUM_PAGES, PAGE, D + 4), dtype=torch.uint8, device=dev)
p = torch.arange(n_pools, device=dev)
loc = bt_pages.long()[p // PAGE] * PAGE + p % PAGE
slot_k = (torch.randn(n_pools, KPOOL, D, device=dev) * 3).to(torch.bfloat16)
slot_s = torch.randn(n_pools, KPOOL, D, device=dev).to(torch.bfloat16)
ape = torch.zeros(KPOOL, D, device=dev)
ck, cs = kc.kpool_compress_and_write_cache(kv, slot_k, slot_s, ape, loc, pool_size=KPOOL, head_dim=D,
                                           round_scale=True, return_compressed=True, write_cache=True)
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import cp_gather_indexer_k_quant_cache_triton, rocm_fp8_mqa_logits
kq = torch.empty((n_pools, D), dtype=torch.float8_e4m3fn, device=dev)
ksc = torch.empty((n_pools, 4), dtype=torch.uint8, device=dev)
cp_gather_indexer_k_quant_cache_triton(kv, kq, ksc, bt_pages[None].contiguous(),
                                       torch.tensor([0, n_pools], dtype=torch.int32, device=dev),
                                       token_to_seq=torch.zeros(n_pools, dtype=torch.int32, device=dev))
ksf = ksc.view(torch.float32).squeeze(-1)
print("gather: K bytes equal:", torch.equal(kq.view(torch.uint8), ck.view(torch.uint8)),
      "| scales equal:", torch.equal(ksf, cs))
q = (torch.randn(M, H, D, device=dev) * 2).to(torch.float8_e4m3fn)
w = torch.rand(M, H, device=dev)
ke = torch.arange(n_pools - M + 1, n_pools + 1, dtype=torch.int32, device=dev)
ks = torch.zeros(M, dtype=torch.int32, device=dev)
out = rocm_fp8_mqa_logits(q, (kq, ksf), w, ks, ke).float()
kf = ck.float() * cs[:, None]
ov = []; err = 0
for i in range(M):
    lim = int(ke[i])
    t = ((q[i].float() @ kf[:lim].T).clamp_min(0) * w[i][:, None]).sum(0)
    o = torch.nan_to_num(out[i, :lim], nan=-1e30)
    ov.append(len(set(t.topk(512).indices.tolist()) & set(o.topk(512).indices.tolist())) / 512)
    err = max(err, ((o - t).abs().max() / t.abs().max()).item())
print("prefill logits: top-512 overlap", ["%.3f" % x for x in ov], "max rel err %.2e" % err,
      "| NaN frac %.3f" % out[:, :n_pools - M].isnan().float().mean().item())
