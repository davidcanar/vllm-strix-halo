"""Unit test: decode indexer logits on the real GLM kpool cache layout.

Fills a paged cache with the production writer (kpool_compress_and_write_cache),
takes the writer's own compressed K/scales as ground truth, and compares
  A) the production decode path  rocm_fp8_paged_mqa_logits (aiter stage1 + overlay)
  B) the new Triton reader       kpool_paged_mqa_logits
  C) the torch reference         kpool_paged_mqa_logits_ref
by top-k overlap and max error."""
import sys, torch
sys.path.insert(0, "/opt/venv/lib/python3.12/site-packages/vllm/v1/attention/ops")
from vllm.models.glm5next.amd.ops import kpool_compress as kc
from vsh_kpool_paged_logits import kpool_paged_mqa_logits, kpool_paged_mqa_logits_ref

torch.manual_seed(0)
dev = "cuda"
PAGE, D, H, KPOOL, NEXT_N = 64, 128, 32, 4, 4
NUM_PAGES = 40
n_pools = 1500                         # ~6K tokens of context, 24 pages
bt_pages = torch.randperm(NUM_PAGES, device=dev)[: (n_pools + PAGE - 1) // PAGE].to(torch.int32)
kv = torch.zeros((NUM_PAGES, PAGE, D + 4), dtype=torch.uint8, device=dev)
p = torch.arange(n_pools, device=dev)
loc = (bt_pages.long()[p // PAGE] * PAGE + p % PAGE)
slot_k = (torch.randn(n_pools, KPOOL, D, device=dev) * 3).to(torch.bfloat16)
slot_s = torch.randn(n_pools, KPOOL, D, device=dev).to(torch.bfloat16)
ape = torch.zeros(KPOOL, D, device=dev, dtype=torch.float32)
ck, cs = kc.kpool_compress_and_write_cache(kv, slot_k, slot_s, ape, loc, pool_size=KPOOL,
                                           head_dim=D, round_scale=True,
                                           return_compressed=True, write_cache=True)
print("FP8_MAX in writer:", kc.FP8_MAX, "| max |k| code value:", ck.float().abs().max().item())

q = (torch.randn(1, NEXT_N, H, D, device=dev) * 2).to(torch.float8_e4m3fn)
w = torch.rand(NEXT_N, H, device=dev, dtype=torch.float32)
ctx = torch.tensor([[n_pools - 3, n_pools - 2, n_pools - 1, n_pools]], dtype=torch.int32, device=dev)
bt = bt_pages[None, :].contiguous()
MAXLEN = 4096

# ground truth from the writer's own compressed output
kf = ck.float() * cs[:, None]
truth = torch.full((NEXT_N, MAXLEN), float("-inf"), device=dev)
for j in range(NEXT_N):
    lim = int(ctx[0, j])
    s = (q[0, j].float() @ kf[:lim].T).clamp_min(0) * w[j][:, None]
    truth[j, :lim] = s.sum(0)

def report(name, out):
    out = out[:, :MAXLEN].float()
    ov = []
    for j in range(NEXT_N):
        lim = int(ctx[0, j])
        a = set(truth[j, :lim].topk(512).indices.tolist())
        b = set(torch.nan_to_num(out[j, :lim], nan=-1e30).topk(512).indices.tolist())
        ov.append(len(a & b) / 512)
    fin = torch.isfinite(truth)
    err = (out[fin] - truth[fin]).abs().max().item()
    rel = err / truth[fin].abs().max().item()
    nan = out[fin].isnan().float().mean().item()
    print(f"{name:34s} top-512 overlap {['%.3f' % o for o in ov]}  max rel err {rel:.2e}  NaN frac {nan:.3f}")

report("B) new triton reader", kpool_paged_mqa_logits(q, kv.unsqueeze(-2), w, ctx, bt, MAXLEN))
report("C) torch reference", kpool_paged_mqa_logits_ref(q, kv.unsqueeze(-2), w, ctx, bt, MAXLEN))

# production path last (it reads per-pool rows from a page table; pad like production)
bt_wide = torch.zeros((1, 2048), dtype=torch.int32, device=dev); bt_wide[0, :bt.shape[1]] = bt[0]
torch.cuda.synchronize()
try:
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import rocm_fp8_paged_mqa_logits
    from vllm.v1.worker.workspace import init_workspace_manager
    try:
        init_workspace_manager(torch.device(dev))
    except Exception as e:  # noqa
        print("workspace init:", type(e).__name__, e)
    a = rocm_fp8_paged_mqa_logits(q, kv.unsqueeze(-2), w, ctx, bt_wide, None, max_model_len=MAXLEN)
    report("A) production (aiter stage1)", a)
except Exception as e:
    import traceback; traceback.print_exc()
    print("A) production path failed to run:", type(e).__name__, e)

