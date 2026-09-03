# AGENTS.md — bring up GLM-5.3-Flash (vLLM) on the 2-box gfx1151 RDMA cluster

Runbook for a person or agent setting this up on a fresh pair of machines, or
repairing the reference rig. Read top to bottom **before** running anything —
several steps are hard to reverse and order matters.

```
        ┌────────────── box1 (ray HEAD, gfx1151) ──────────────┐
        │  toolbox "vllm-glm"  ──►  vllm serve  TP rank 0       │
        │  vsh-glm.service → vsh-cluster-restart.sh             │
        │  API :1235 (glm-5.3-flash)   +  :1234 (ds4, optional) │
        └───────────────┬───────────────────────────────────────┘
                        │  Thunderbolt-4 cable
                        │  RoCE-RDMA dev = usb4_rdma0  (tbv stack)
                        │  IP link thunderbolt0 = 10.0.2.1/.2
        ┌───────────────┴───────────────────────────────────────┐
        │  toolbox "vllm-glm"  ──►  ray worker  TP rank 1        │
        └────────────── box2 (ray WORKER, gfx1151) ─────────────┘
```

Four layers, build/verify in this order:

1. **tbv RDMA** (`tbv/`) — the Thunderbolt interconnect. Foundational and the
   riskiest; do it first. *(The cluster also runs without it on a slow TCP
   fallback — `transport: tcp` in vsh-config.yaml — use that to de-risk a
   fresh bring-up before adding RDMA.)*
2. **vLLM engine** (`container/`) — rebuild the serving image, one toolbox per box.
3. **Model weights** (`scripts/download-models.sh`) — on both boxes.
4. **Host orchestration** (`host/`) — the launch scripts, env, systemd units.

## 0. Prerequisites (verify these exist; do NOT try to synthesize them)

- **2× AMD Strix Halo / gfx1151** (~128 GB unified memory each), same LAN.
- A **Thunderbolt-4/USB4 cable** physically connecting the two boxes.
- Linux with **kernel headers/devel** for the running kernel on each box,
  `podman`, `toolbox`, `rdma-core`, `git`, build toolchain.
- Root/sudo on both boxes (kernel modules, systemd units).
- **Secure Boot disabled on both boxes** — the tbv modules are unsigned.
- `hf` CLI logged in on both boxes (model download).

Roles, kept consistent everywhere: **box1 = ray head**, Thunderbolt IP
`10.0.2.1`; **box2 = worker**, `10.0.2.2`. Site values live in
`host/vsh-config.yaml`, deployed as `~/vsh-config.yaml` on box1.

## 1. RDMA (tbv) — both boxes, then reboot both together

```bash
cd ~/vllm-strix-halo/tbv
./build-modules.sh                     # no sudo; builds for the running kernel
sudo ./install-modules.sh              # stages + loads the matched module set
# reboot BOTH boxes together
```

After reboot, on both boxes:

```bash
rdma link show            # usb4_rdma0/1 state ACTIVE
ip -brief addr show thunderbolt0   # 10.0.2.1 (box1) / 10.0.2.2 (box2)
```

Failure modes and recovery are documented in `tbv/README.md` — several
require a coordinated reboot of both boxes. Do not mix module sets across
boxes (core↔net ring ABI panic).

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
toolbox run -c vllm-glm bash -c 'ls /dev/infiniband && ibv_devices'   # must list usb4_rdma0
```

If `ibv_devices` is empty inside the container, the provider ABI is wrong —
see PATCHES.md §1 (provider must match the base's libibverbs ABI, rdmav59).

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
compiles on first boot. RDMA verification greps the serve journal for
`tbv_ar2: rank[0-9] ready (qpn=... peer_qpn=...)`.

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
| `glm53_max_ctx` | 32768 | context cap (KV pin / block size bound) |
| `glm53_kv_bytes` | 4294967296 | pinned GPU KV pool |
| `glm53_mtp_tokens` | 3 | MTP draft tokens; 0 = off |
| `glm53_aiter` | 0 | allow aiter kernels (DS4 ran aiter-less) |
| `transport` | rdma | rdma \| tcp fallback |
| `rdma_hca` | usb4_rdma0 | pin one unambiguous HCA |

## 8. Troubleshooting quick map

| symptom | look at |
|---|---|
| API never 200 | `journalctl --user -u vsh-glm -u vsh-glm-manual -n 100` |
| RDMA not ready | `rdma link show`, GID index 1, `ibv_devices` in container |
| RCCL init "internal error" | two ACTIVE rails on one netdev → pin `rdma_hca` |
| OOM during weight load | raise `glm53_kv_bytes` downward / stop the other model |
| worker dies after `TileLang ... mhc_pre_big_fuse_with_norm` | stale image without the gfx1151 TileLang-MHC gate — rebuild with `container/build.sh` (PATCHES.md) |
| worker dies after `Encoder cache will be initialized` | missing `--skip-mm-profiling` (PATCHES.md) |
| EngineCore SIGSEGV on load | RCCL CQ ENOMEM — PATCHES.md §5 candidate patch |

## 9. Performance

Measured during validation on the reference rig (2× Ryzen AI Max+ 395 / 128 GB
each), single stream, temperature 0 — numbers are recorded here after each
bring-up change. The first measured run after the initial bring-up is the
baseline; treat any figure quoted before that as unverified.
