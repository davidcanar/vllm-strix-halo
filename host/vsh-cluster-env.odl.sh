#!/usr/bin/env bash
# OdinLink transport: RCCL rides the odl_tb5 net plugin (NCCL_NET_PLUGIN)
# for the big prefill-sized collectives, odl_ar2 (DS4_ODL_AR2) carries the
# small decode all-reduces over odl streams, and odl_mq (fail-open, in the
# patched shm_broadcast) moves the EngineCore<->worker control plane off
# the per-step TCP round trips.
#
# The kernel driver (odl_tb5.ko, /dev/odl_tb5_0) is host-side, built by the
# repo's odinlink/build-odinlink.sh — this profile only configures the
# userspace. Needs /dev bind-mounted into the container (the toolbox vllm-glm
# container binds the whole /dev, so the node is visible).
source "$HOME/vsh-cluster-env.sh"

# RCCL: no ibverbs at all on this stack; the net plugin replaces it.
export NCCL_IB_DISABLE=1
unset NCCL_IB_HCA NCCL_IB_GID_INDEX 2>/dev/null || true
# Leave NCCL_PROTO to RCCL's per-size selection: LL is wrong for the
# prefill-sized ops (8 bytes of flag per 8 bytes of payload), and decode
# never reaches RCCL (odl_ar2 owns <=1 MiB).
unset NCCL_PROTO
# Point RCCL at the OdinLink net plugin (absolute path; RCCL dlopens it).
export NCCL_NET_PLUGIN=/usr/local/lib/odinlink/librccl_net_odl_tb5.so
export LD_LIBRARY_PATH="/usr/local/lib/odinlink${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# The plugin multiplexes every connection over ONE odl device: RCCL's default
# 26 channels turn into 26x per-message syscall overhead on prefill ops.
export NCCL_MAX_NCHANNELS=4

# Control-plane bootstrap still rides tbnet (thunderbolt0).
export NCCL_SOCKET_IFNAME=${VSH_CONTROL_IFACE:-thunderbolt0}
export GLOO_SOCKET_IFNAME=${VSH_CONTROL_IFACE:-thunderbolt0}

# odl_ar2 decode all-reduce (env read by the patched cuda_communicator).
export DS4_ODL_AR2=${VSH_GLM53_ODL_AR2:-1}
# odl_ar2 rendezvous: rank 0 connects out to rank 1's listener. Rank 1 is the
# box2 ray worker; vsh-cluster-restart.sh exports VSH_ODL_RANK1_IP=<worker_ip>
# on both boxes.
export ODL2_RANK1_IP=${VSH_ODL_RANK1_IP:-10.0.2.2}
export ODL2_PORT=${VSH_ODL_PORT:-18541}
export ODL2_DEV=${VSH_ODL_DEV:-0}

# --- GLM-5.3 cache + draft-length knobs (2026-10-02) -------------------------
# VSH_GLM53_APC_ALIGN: resolve the EAGLE last-block-drop set to *pure drafter*
#   groups instead of the upstream "flag every group" fallback. On this model
#   that resolves to the empty set (the MTP layer shares group 0 with the target
#   MLA layers), which is what stops prefix-cache hits losing a scheduler page
#   per lookup. PATCHES.md 18.
# VSH_GLM53_APC_RETENTION: Mamba/sliding-window prefix-cache checkpoint interval
#   (multiple of the scheduler block size, 2304 here). The stock default of 0
#   keeps only the latest replay boundary, which made the first identical repeat
#   miss. PATCHES.md 20.
# VSH_ADAPTIVE_K: EMA policy for the verified draft-prefix length. Installed and
#   instrumented but OFF: the step time is draft-length independent up to k=4 and
#   acceptance saturates near 2 tokens/step, so trimming only gives tokens away.
#   The JSON override below is the in-run k-sweep instrument. PATCHES.md 17.
# VSH_SYNC_INSTR: times the KDA chunk-index host sync (index.py). Off; the sync
#   is on the chunked path only and the GPU is 99% busy while it blocks.
#   PATCHES.md 22.
export VSH_GLM53_APC_ALIGN=1
export VSH_GLM53_APC_RETENTION=2304
export VSH_ADAPTIVE_K=off
export VSH_ADAPTIVE_K_ALPHA=0.25
export VSH_ADAPTIVE_K_MARGIN=0.5
export VSH_ADAPTIVE_K_MIN_STEPS=4
export VSH_ADAPTIVE_K_SET=1,2,3
export VSH_ADAPTIVE_K_HIST=200
export VSH_ADAPTIVE_K_DEBUG=0
export VSH_ADAPTIVE_K_DEBUG_EVERY=25
export VSH_ADAPTIVE_K_JSON=/home/davidcanar/vsh-adaptive-k.json
export VSH_ADAPTIVE_K_FORCE=0
export VSH_SYNC_INSTR=0

# tool-call drop instrumentation (2026-10-02): dump text the parser discards
# when a tool call is emitted inside an unclosed reasoning span.
export VSH_TOOLCALL_DROP_LOG=1
export VSH_TOOLCALL_DROP_DIR=/tmp/vsh-toolcall-drops

# MLA W_UK/W_UV absorption: aiter's triton fp8 BMM has no fp8 hardware on RDNA3.5
# (emulated, ~0.38 ms x 28 calls = ~10.7 ms/step); 0 = plain bf16 torch.bmm.
# Also skips the 2x1024-shape fp8-BMM precompile at boot. (PATCHES.md 28)
export VLLM_ROCM_USE_AITER_FP8BMM=${VSH_GLM53_FP8BMM:-0}

# Load-time int8 (group 128) for BF16 linears + HIP W8A16 GEMV (PATCHES.md 28.6).
export VSH_W8A16=${VSH_GLM53_W8A16:-1}
