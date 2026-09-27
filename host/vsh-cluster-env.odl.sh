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
