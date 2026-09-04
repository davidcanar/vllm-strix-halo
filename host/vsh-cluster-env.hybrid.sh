#!/usr/bin/env bash
# Hybrid transport: RCCL on sockets (thunderbolt0) for the big prefill-sized
# collectives, tbv_ar2 RDMA (usb4_rdma0) for the small decode ones.
# Measured on the rig: RCCL-over-sockets beats RCCL-over-IB at every size
# (16 KB 0.11 vs 0.40 ms; 32 MB 1.48 vs 0.97 GB/s), so IB is disabled for
# RCCL, while tbv_ar2 keeps its own ibverbs QP and is unaffected.
source "$HOME/vsh-cluster-env.sh"
export NCCL_IB_DISABLE=1
export VSH_TBV_AR2=${VSH_GLM53_TBV_AR2:-1}
