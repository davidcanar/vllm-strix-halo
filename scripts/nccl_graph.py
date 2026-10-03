"""2-rank RCCL all-reduce: eager, then captured in a CUDA graph and replayed (same env as serving)."""
import os, sys, time, torch, torch.distributed as dist
rank = int(sys.argv[1]); os.environ.setdefault("MASTER_ADDR", "10.0.2.1"); os.environ.setdefault("MASTER_PORT", "29611")
dist.init_process_group("nccl", rank=rank, world_size=2, device_id=torch.device("cuda:0"))
x = torch.ones(4 * 4096, device="cuda", dtype=torch.bfloat16) * (rank + 1)
for _ in range(3): dist.all_reduce(x)
torch.cuda.synchronize(); print(f"[r{rank}] eager ok, x[0]={x[0].item()}", flush=True)
x.fill_(rank + 1)
s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(2): dist.all_reduce(x)          # warm on side stream
torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
g = torch.cuda.CUDAGraph()
y = torch.ones(4 * 4096, device="cuda", dtype=torch.bfloat16)
with torch.cuda.graph(g):
    dist.all_reduce(y)
print(f"[r{rank}] captured", flush=True)
for i in range(20):
    y.fill_(rank + 1); g.replay()
    torch.cuda.synchronize()
print(f"[r{rank}] graph replay x20 ok, y[0]={y[0].item()} (expect 3.0)", flush=True)
dist.destroy_process_group()
