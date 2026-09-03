#!/usr/bin/env bash
# TCP-transport env for the vllm-strix-halo cluster: plain TCP over the
# Thunderbolt IP link (thunderbolt0), no RDMA. Use to de-risk a bring-up when
# the RDMA rail is down; throughput is lower but everything else is identical.
source "$HOME/vsh-cluster-env.sh"
export NCCL_IB_DISABLE=1
export VSH_TBV_AR2=0
