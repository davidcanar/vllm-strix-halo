"""Two RCCL communicators: graphed all-reduce (comm A, vLLM-style side capture) + eager all-gather (comm B)."""
import os, sys, torch, torch.distributed as dist
rank = int(sys.argv[1]); os.environ.setdefault("MASTER_ADDR", "10.0.2.1"); os.environ.setdefault("MASTER_PORT", "29612")
dist.init_process_group("nccl", rank=rank, world_size=2, device_id=torch.device("cuda:0"))
gA = dist.new_group([0, 1]); gB = dist.new_group([0, 1])
y = torch.ones(4 * 4096, device="cuda", dtype=torch.bfloat16)
z = torch.ones(4 * 77440, device="cuda", dtype=torch.bfloat16); zo = torch.empty(2 * 4 * 77440, device="cuda", dtype=torch.bfloat16)
for _ in range(2): dist.all_reduce(y, group=gA); dist.all_gather_into_tensor(zo, z, group=gB)
torch.cuda.synchronize()
s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(2): dist.all_reduce(y, group=gA)
torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    for _ in range(50): dist.all_reduce(y, group=gA)      # many collectives per "step"
print(f"[r{rank}] captured", flush=True)
for i in range(30):
    g.replay()
    dist.all_gather_into_tensor(zo, z, group=gB)          # eager, other communicator, no sync in between
torch.cuda.synchronize()
print(f"[r{rank}] 30 x (graph of 50 AR on comm A + eager AG on comm B) ok", flush=True)
dist.destroy_process_group()
