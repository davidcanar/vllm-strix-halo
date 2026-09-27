# AGENTS.md — bring up GLM-5.3-Flash (vLLM) on the 2-box gfx1151 OdinLink cluster

Runbook for a person or agent setting this up on a fresh pair of machines, or
repairing the reference rig. Read top to bottom **before** running anything —
several steps are hard to reverse and order matters.

```
        ┌────────────── box1 (ray HEAD, gfx1151) ──────────────┐
        │  toolbox "vllm-glm"  ──►  vllm serve  TP rank 0       │
        │  vsh-glm.service → vsh-cluster-restart.sh             │
        │  API :1235 (glm-5.3-flash)   +  :1234 (ds4, optional) │
        └───────────────┬───────────────────────────────────────┘
                        │  Thunderbolt-4 cable (EXACTLY ONE)
                        │  OdinLink driver = odl_tb5 (/dev/odl_tb5_0)
                        │  IP link thunderbolt0 = 10.0.2.1/.2
        ┌───────────────┴───────────────────────────────────────┐
        │  toolbox "vllm-glm"  ──►  ray worker  TP rank 1        │
        └────────────── box2 (ray WORKER, gfx1151) ─────────────┘
```

Four layers, build/verify in this order:

1. **OdinLink fabric** (`odinlink/`) — the Thunderbolt interconnect driver.
   Foundational and the riskiest; do it first. *(The cluster also runs
   without it on the TCP fallback — `transport: tcp` in vsh-config.yaml —
   use that to de-risk a fresh bring-up before adding the fabric.)*
2. **vLLM engine** (`container/`) — rebuild the serving image, one toolbox per box.
3. **Model weights** (`scripts/download-models.sh`) — on both boxes.
4. **Host orchestration** (`host/`) — the launch scripts, env, systemd units.

## 0. Prerequisites (verify these exist; do NOT try to synthesize them)

- **2× AMD Strix Halo / gfx1151** (~128 GB unified memory each), same LAN.
- **Exactly one** Thunderbolt-4/USB4 cable physically connecting the two boxes
  (OdinLink demultiplexes peers by route — a second cable puts both XDomains
  at the same route and breaks peer demux).
- Linux (**stock kernel**) with kernel headers/devel for the running kernel on
  each box, `podman`, `toolbox`, `git`, build toolchain.
- Root/sudo on both boxes (kernel module, systemd units).
- **Secure Boot disabled on both boxes** — the `odl_tb5` module is unsigned
  unless a MOK is enrolled (the install path signs it when one exists).
- `hf` CLI on both boxes (model download; the AWQ repo is public).

Roles, kept consistent everywhere: **box1 = ray head**, Thunderbolt IP
`10.0.2.1`; **box2 = worker**, `10.0.2.2` (also odl_ar2 rank 1). Site values
live in `host/vsh-config.yaml`, deployed as `~/vsh-config.yaml` on box1.

## 1. OdinLink fabric — both boxes

```bash
cd ~/vllm-strix-halo/odinlink
./build-odinlink.sh                    # no sudo; builds odl_tb5.ko for the running kernel
sudo ./install-odinlink.sh             # installs to /usr/local, loads the module
```

Previously ran the old `tbv`/ibverbs stack? `sudo ./uninstall-tbv.sh --apply`
first (it also removes the `blacklist=thunderbolt` kernel args, without which
no thunderbolt driver loads at all), then reboot BOTH boxes together.

After install, on both boxes:

```bash
odl-state                              # state=ready
ls -l /dev/odl_tb5_0                   # the stream device node
ip -brief addr show thunderbolt0       # 10.0.2.1 (box1) / 10.0.2.2 (box2)
```

The driver is vermagic-locked: rebuild after every kernel update. Details,
single-cable rule, MOK/SELinux notes: `odinlink/README.md`.

## 2. Serving image (box1) → toolbox on both boxes

```bash
cd ~/vllm-strix-halo
container/build.sh                     # ~1-2 h; builds vLLM @ pinned commit
podman save vllm-strix-halo:local | ssh 10.0.2.2 podman load
toolbox create -c vllm-glm -i vllm-strix-halo:local   # BOTH boxes
```

Verify on both boxes:

```bash
toolbox run -c vllm-glm python -c "import torch; print(torch.cuda.get_device_name(0))"
toolbox run -c vllm-glm bash -c 'ls -l /dev/odl_tb5_0 && ls /usr/local/lib/odinlink/'
```

The toolbox container bind-mounts the host `/dev`, so `/dev/odl_tb5_0` is
visible without extra `--device` flags. `/usr/local/lib/odinlink/` must list
`libodl_tb5.so.0` and `librccl_net_odl_tb5.so` (the RCCL net plugin).

## 3. Model weights — both boxes

```bash
scripts/download-models.sh glm53        # ~191 GB each box, resumable
```

GLM-5.3-Flash must be the **AWQ W4A16** checkpoint (PATCHES.md §3 — the
official FP8/BF16 checkpoints do not fit this rig).

## 4. Host orchestration (box1)

```bash
cp host/vsh-config.yaml ~/vsh-config.yaml    # edit model dirs / IPs / ports
host/deploy.sh                                # installs scripts + unit; syncs box2
```

`deploy.sh` installs to `$HOME` on box1 and copies the three cluster-env files
+ `container-heal.sh` to box2 (they must be **byte-identical** on both boxes).

## 5. Launch / operate

```bash
./vllm-strix-halo.sh start      # glm53 bring-up: teardown -> ray -> serve -> verify
./vllm-strix-halo.sh status     # both models, RDMA rail, Ray, memory
./vllm-strix-halo.sh logs       # follow serve journals
./vllm-strix-halo.sh stop
```

Bring-up gates: containers exec-able on both boxes → Ray 2.0 GPU → API 200 on
`:1235` → warmup dispatched. A cold kernel cache adds ~25 min of Triton/LLVM
compiles on first boot. Fabric verification greps the serve journal for the
`odl_ar2: rankN ready` lines (the decode all-reduce fast path,
`glm53_odl_ar2: 1`) and the `odl_mq: writer/reader up` lines (the control-plane
data plane). With `NCCL_DEBUG=INFO` RCCL logs which net plugin it loaded.

## 6. Model switching / coexistence

- glm53 lives on **:1235** (unit `vsh-glm`), the ds4 stack on **:1234** (unit
  `ds4-vllm`); both can run side by side (memory permitting — the GLM AWQ
  weights need ~95 GB/box plus the DS4 stack's ~78 GB/box **does not fit
  together**; stop one before starting the other).
- `./vllm-strix-halo.sh ds4 start|stop|status|logs` drives the deployed
  ds4-vllm stack. If it is not deployed: `git clone https://github.com/AlexKGwyn/ds4-vllm` and
  follow its AGENTS.md (RDMA kernel modules are shared with this repo — build
  once).

## 7. Tuning knobs (vsh-config.yaml)

| key | default | meaning |
|---|---|---|
| `glm53_max_ctx` | 131072 | context cap. 128 K measures 4.01x concurrency with the pin below; the limit past this is TTFT (prefill is a flat 156 tok/s), not memory — PATCHES.md §5.2 |
| `glm53_kv_bytes` | 8589934592 | pinned GPU KV pool (8 GiB = 526,083 tokens). Leave ~20 GiB/box free; `gpu_memory_utilization` is inert while this is set |
| `glm53_max_batched` | 4096 | prefill chunk. 512 -> 4096 is +30% prefill for 0.6 GiB and no decode cost; the old "NOT larger" note was wrong — PATCHES.md §8 |
| `glm53_mtp_tokens` | 0 | MTP draft tokens; 0 = off. Was corrupting structured output on the Sep-3 pin; upstream #58454 (kpool corruption with spec decode) is in the Sep-27 pin — re-enable (`glm53_mtp_tokens: 3`) and validate before relying on it |
| `glm53_max_seqs` | 256 | concurrent sequences. The 1024 default is meaningless here (4.01x concurrency at 128 K) and blocks CUDA graph capture: each decode seq needs one Mamba cache block and there are ~293 — PATCHES.md §11 |
| `glm53_enforce_eager` | 1 | 0 tries torch.compile + CUDA graphs. Currently blocked: rank 1 hits a Triton SystemError during breakable capture, and capture costs 11.4 GiB/rank — PATCHES.md §11 |
| `glm53_tool_parsing` | 1 | `--enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm47`. Needed by coding harnesses; 0 serves raw text. Note the response field is `reasoning`, not `reasoning_content` — see PATCHES.md §7 |
| `glm53_aiter` | 1 | aiter must be ON — the sparse-attention indexer's ROCm path requires it |
| `glm53_odl_ar2` | 1 | odl_ar2 decode all-reduce over OdinLink streams (odl transport only). Prefill-sized collectives ride RCCL over the net plugin |
| `transport` | odl | `odl` \| `tcp`. odl = RCCL over the OdinLink net plugin + odl_ar2 decode all-reduce + odl_mq control plane. tcp = plain sockets over thunderbolt0, no kernel modules |

## 8. Troubleshooting quick map

| symptom | look at |
|---|---|
| API never 200 | `journalctl --user -u vsh-glm -u vsh-glm-manual -n 100` |
| OdinLink not ready | `odl-state`, `ls /dev/odl_tb5_0`, `lsmod \| grep odl_tb5` (rebuild the module after kernel updates) |
| odl_ar2 not ready | serve journal `odl_ar2 init failed:` line; `ODL2_RANK1_IP`/port reachability rank0→rank1 (tbnet must be up) |
| RCCL fell back to sockets | `NCCL_DEBUG=INFO` — look for the net-plugin load line; `NCCL_NET_PLUGIN` path must exist in-container |
| OOM during weight load | raise `glm53_kv_bytes` downward / stop the other model |
| worker dies after `TileLang ... mhc_pre_big_fuse_with_norm` | stale image without the gfx1151 TileLang-MHC gate — rebuild with `container/build.sh` (PATCHES.md) |
| worker dies after `Encoder cache will be initialized` | missing `--skip-mm-profiling` (PATCHES.md) |
| EngineCore SIGSEGV on load | RCCL CQ ENOMEM — PATCHES.md §5 candidate patch |

## 9. Performance

Validated on the reference rig (2× Ryzen AI Max+ 395 / 128 GB each), single
stream, temperature 0, RDMA transport (RCCL `Using network IB` on
`usb4_rdma0`, RoCE, 20 Gbps negotiated). MTP = `glm5_next_mtp`, 3 draft tokens
(`vsh-mtp-ropefree-triton-sparse.patch`, PATCHES.md §1.5 — without it the
worker SIGABRTs on the first speculative batch):

| context | MTP off | MTP on (3 tokens) | MTP + tbv_ar2 |
|---|---|---|---|
| 512 | ~6.7 tok/s | ~7.6 tok/s | ~7.9 tok/s |
| 4.5k | ~2.4 tok/s | ~5.5 tok/s | ~5.6 tok/s |

MTP acceptance on these runs: ~90% of draft tokens accepted, mean
acceptance length up to 4.0, per-position rates 1.000/1.000/1.000 on greedy
repetitive text — from `/metrics` (`vllm:spec_decode_*`) or the periodic
`SpecDecoding metrics` journal lines.

These are first-bring-up baselines with the per-step fp8→fnuz conversion on
the sparse-MQA path (see PATCHES.md §1.3 — an in-kernel conversion is the
next optimization). Treat higher figures quoted elsewhere as unverified.
