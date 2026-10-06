# AGENTS.md — bring up GLM-5.3-Flash (vLLM) on the 2-box gfx1151 OdinLink cluster

Runbook for a person or agent setting this up on a fresh pair of machines, or
repairing the reference rig. Read top to bottom **before** running anything —
several steps are hard to reverse and order matters.

```
        ┌────────────── box1 (ray HEAD, gfx1151) ──────────────┐
        │  toolbox "vllm-glm"  ──►  vllm serve  TP rank 0       │
        │  vsh-glm.service → vsh-cluster-restart.sh             │
        │  API :1234  glm-5.3-flash | deepseek-v4-flash         │
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
`:1234` → warmup dispatched. A cold kernel cache adds ~25 min of Triton/LLVM
compiles on first boot. Fabric verification greps the serve journal for the
`odl_ar2: rankN ready` lines (the decode all-reduce fast path,
`glm53_odl_ar2: 1`) and the `odl_mq: writer/reader up` lines (the control-plane
data plane). With `NCCL_DEBUG=INFO` RCCL logs which net plugin it loaded.

## 6. Model switching / coexistence

- One model at a time. GLM-5.3 (unit `vsh-glm`) and DeepSeek-V4-Flash (unit
  `vsh-ds4`, `ds4_engine: native`) both run in the `vllm-glm` container and both
  serve **:1234**. Their weights do not fit together, so stop one before starting
  the other: `./vllm-strix-halo.sh ds4 stop && ./vllm-strix-halo.sh glm53 start`
  (or the reverse).
- `./vllm-strix-halo.sh ds4 start|stop|status|logs` drives DS4. The native engine
  is the 0.31 port in this image (PATCHES.md §35–§43). `ds4_engine: delegate` runs
  the June `AlexKGwyn/ds4-vllm` stack instead (unit `ds4-vllm`; if it is not
  deployed, clone it and follow its AGENTS.md — the RDMA kernel modules are
  shared, so build them once).
- DS4 needs `VSH_W8A16=0 VSH_W8A16_LMHEAD=0` on both ranks. The DS4
  restart/reserve scripts pass them, and every restart script refuses to start
  when the two boxes' env files differ.

## 7. Tuning knobs (vsh-config.yaml)

| key | default | meaning |
|---|---|---|
| `glm53_max_ctx` | 524288 | 512K context (the model's limit is 1M). 2.41 concurrent 512K sessions fit the 16 GiB pin below; a cold 512K prompt takes ~49 min to prefill (128K: 8.5 min) — PATCHES.md §45 |
| `glm53_kv_bytes` | 17179869184 | pinned GPU KV pool (16 GiB = 1,105,488 tokens with the production DFlash2 k=3; 1,183,680 with MTP k=3). Leave ~20 GiB/box free; `gpu_memory_utilization` is inert while this is set |
| `glm53_max_batched` | 8192 | prefill chunk (single-user profile; MiaAI ships 7168) — PATCHES.md §8 |
| `glm53_mtp_tokens` | 3 | draft tokens for `glm53_spec_method`; 0 = off. k=3 is the measured best for DFlash2 (PATCHES.md §28.7) and was for MTP (k=1...4 within 3 %, §17). The live override (`VSH_ADAPTIVE_K_JSON`) sweeps it without a reboot |
| `glm53_spec_method` | dflash | the DFlash2 drafter (`glm53_draft_model`), default since PATCHES.md §28.7: with §27's fixes it beats MTP k=3. `glm5_next_mtp` reverts to the built-in MTP head (§24/§25 measured the opposite, before §27) |
| `glm53_max_seqs` | 32 | concurrent sequences (single-user profile). Each decode seq needs one Mamba cache block and there are ~293, which is what used to block CUDA graph capture — PATCHES.md §11 |
| `glm53_enforce_eager` | 0 | 0 = CUDA graphs: PIECEWISE breakable capture with every TP collective an eager break, -5 to -6 ms/step (PATCHES.md §33); 1 = eager |
| `glm53_cg_mode` | PIECEWISE | `FULL_DECODE_ONLY` / `FULL_AND_PIECEWISE` hang on the first replay (RCCL inside the graph, PATCHES.md §29) |
| `glm53_profiler_dir` / `_active` | `/home/davidcanar/glmprof`, 40 | torch profiler, inert until `POST /start_profile`. Captures land on both ranks |
| `glm53_tool_parsing` | 1 | `--enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm47`. Needed by coding harnesses; 0 serves raw text. Note the response field is `reasoning`, not `reasoning_content` — see PATCHES.md §7 |
| `glm53_aiter` | 1 | aiter must be ON — the sparse-attention indexer's ROCm path requires it |
| `glm53_odl_ar2` | 1 | odl_ar2 decode all-reduce over OdinLink streams (odl transport only). Prefill-sized collectives ride RCCL over the net plugin |
| `glm53_gpu_sclk_max` | 2100 | iGPU SCLK cap in MHz: leaving power to the CPU is faster overall (PATCHES.md §28); empty = stock 2900 |
| `glm53_warmup_ctx` | 32768 | prompt size of the boot warm-up that JIT-warms the prefill/indexer kernels. It holds the engine for ~2.5 min after the API answers, and requests sent in that window wait for it (PATCHES.md §44) |
| `transport` | odl | `odl` \| `tcp`. odl = RCCL over the OdinLink net plugin + odl_ar2 decode all-reduce + odl_mq control plane. tcp = plain sockets over thunderbolt0, no kernel modules |

The `ds4_*` keys (engine, drafter, context, chunk, sampling defaults) are documented
inline in `host/vsh-config.yaml`; current DS4 numbers are in README.md.


Knobs that live in `~/vsh-cluster-env.<transport>.sh` (exported, so they reach
the engine process; the file must be byte-identical on both boxes):

| key | default | meaning |
|---|---|---|
| `VSH_GLM53_APC_ALIGN` | 1 | Resolve the EAGLE last-block-drop set to pure-drafter KV groups instead of upstream's flag-every-group fallback. On this model that is the empty set, which is what stops every prefix-cache hit losing a scheduler page (2304 tokens). Measured: a 13 877-token repeat went 11 520 -> 13 824 cached tokens, TTFT 8.92 s -> 2.05 s — PATCHES.md §18 |
| `VSH_GLM53_APC_RETENTION` | 2304 | `--prefix-cache-retention-interval` (a multiple of the scheduler block size). Stock 0 keeps only the latest replay boundary for the Mamba/KDA groups, which made the *first* identical repeat miss; 2304 makes it hit 11 520 tokens (TTFT 53 s -> 9.6 s) and the second 13 824 (0.95 s) — PATCHES.md §20 |
| `VSH_ADAPTIVE_K` | off | EMA draft-length policy, installed and instrumented. Off because the step is draft-length independent up to k=4; `VSH_ADAPTIVE_K_JSON` (`{"force": 2}`) is the in-run k-sweep instrument — PATCHES.md §17 |
| `VSH_TRITON_PTR_CACHE` | unset | Memoise Triton's per-pointer `hipPointerGetAttribute`. Applied and validated bit-exact, **leave off**: the call is 0.5 us, ~0.3 % of a step — PATCHES.md §21 |
| `VSH_SYNC_INSTR` | 0 | Time the KDA chunk-index host sync. Off; that sync is on the chunked path only and the GPU is 99 % busy while it blocks — PATCHES.md §22 |
| `VSH_SHM_BUSY_LOOP_S` | unset (stock 1.0 s) | SHM reader busy-spin window. Measured null-to-negative here (4.16-4.31 vs 4.41-4.45 steps/s) because the control plane already rides odl_mq — PATCHES.md §19 |
| `VSH_MHC_CFG` | 1 (in code) | gfx1151 tiles and the fused/unfused crossover for aiter's mHC kernels (PATCHES.md §42). GLM and DS4 share these ops and shapes (hidden 4096, hc_mult 4), so both use it; export `0` to restore aiter's fallback |
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

| Boot wedges in Ray bring-up (API never 200, "worker process has died", MemAvailable back to ~120 GB) | transient; recover with a clean teardown: `pkill -f vsh-cluster-restart.sh`, `systemctl --user stop vsh-glm-manual`, kill stray `vllm serve` pids for the model dir, `systemctl --user reset-failed`, then re-run `host/vsh-cluster-restart.sh` |
| Prefix cache hit rate stuck at 0 % | check the boot log for `[vsh-glm53-apc-align] eagle=[]` and `prefix-cache retention interval: 2304`; if either is missing the env file did not reach the engine — PATCHES.md §18/§20 |
## 9. Performance

Measured 2026-10-06 on the reference rig (2x Ryzen AI Max+ 395 / 128 GB each),
single stream, `odl` transport, GLM-5.3-Flash AWQ W4A16, TP=2, DFlash2 k=3,
PIECEWISE CUDA graphs. Method: `scripts/glm_regress.sh` (PATCHES.md §44);
DS4's numbers are in README.md.

| workload | decode | tokens/step | step |
|---|---|---|---|
| prose | 19.6 tok/s | 1.91 | 97 ms |
| JSON output | 23.4 tok/s | 2.26 | 95 ms |
| tool calls | 25.3 tok/s | 2.52 | 99 ms |

| prefill (cold) | 5.8K prompt | 11.5K prompt | 19.9K prompt |
|---|---|---|---|
| | 291-294 tok/s | 281-282 tok/s | 278 tok/s |

Prefix cache: a 16.1K-token prompt takes 57 s cold and 8.4 s on its first
repeat. Hits land on the 2304-token retention boundaries, so 13,824 tokens were
served from cache and the tail past the last boundary was recomputed. A prompt
that ends just past a boundary repeats in under 1 s (PATCHES.md §20). The chat
template renders the reasoning effort first, so changing it between turns misses
the whole cache (§45).

Spec acceptance at k=3 is 1.9-2.5 tokens per step depending on the workload;
read it from `vllm:spec_decode_num_accepted_tokens_per_pos_total`.

**Why it is where it is:** decode is GPU-bound on weight traffic after the
§28-§33 rounds (clock cap, HIP int4 MoE, W8A16 linears, router, graphs).
Prefill is compute-bound in the routed-expert MoE: its AWQ Triton tiles are
tuned only up to M=512, while prefill chunks run M~65K. The remaining levers
are listed in README.md "Known issues and next steps".
