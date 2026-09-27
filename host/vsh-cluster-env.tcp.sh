#!/usr/bin/env bash
# TCP transport (fallback, no OdinLink dependencies): every collective rides
# RCCL over IP sockets on the thunderbolt0 link. Measured single-stream this
# is at parity with the RDMA rail for prefill (the odl plugin's win is at
# concurrency); decode loses the odl_ar2 fast path and falls back to RCCL.
# Use it to bisect fabric issues or when the odinlink driver is unavailable.
source "$HOME/vsh-cluster-env.sh"
export NCCL_IB_DISABLE=1
unset NCCL_IB_HCA NCCL_IB_GID_INDEX NCCL_PROTO 2>/dev/null || true
export NCCL_SOCKET_IFNAME=${VSH_CONTROL_IFACE:-thunderbolt0}
export GLOO_SOCKET_IFNAME=${VSH_CONTROL_IFACE:-thunderbolt0}
export DS4_ODL_AR2=0
