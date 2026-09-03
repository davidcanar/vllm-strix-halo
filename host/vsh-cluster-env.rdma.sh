#!/usr/bin/env bash
# RDMA-transport env for the vllm-strix-halo cluster.
# The base env sets NCCL_IB_HCA=usb4_rdma, a PREFIX that can match more than
# one device (e.g. after a link reset leaves two ACTIVE rails on one netdev),
# which makes RCCL's ncclCommInitRank fail with "internal error". Pin one
# unambiguous HCA here (rdma_hca in vsh-config.yaml). The decode all-reduce
# still runs via tbv_ar2 (VSH_TBV_AR2 from the base env).
source "$HOME/vsh-cluster-env.sh"
export NCCL_IB_HCA=${VSH_RDMA_HCA:-usb4_rdma0}
export NCCL_IB_DISABLE=0
# RCCL logging is off. Re-enable to debug an init failure; expect the
# harmless "GPU Direct RDMA not available for device 0" (no GPUDirect on
# gfx1151 — host staging is the designed path).
#   export NCCL_DEBUG=WARN
#   export NCCL_DEBUG_SUBSYS=INIT,NET,ENV

# This file adds only what is specific to the RDMA transport; everything else
# lives in the base env. Do not restate base knobs here: a bare `export` after
# the source runs LAST and clobbers values passed in via `podman exec --env`.
