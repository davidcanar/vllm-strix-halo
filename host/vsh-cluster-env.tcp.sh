#!/usr/bin/env bash
# TCP-transport env for the vllm-strix-halo cluster: plain TCP over the
# Thunderbolt IP link (thunderbolt0), no RDMA at all. Originally the de-risk
# fallback for a bring-up with the RDMA rail down -- but it is NOT slower.
# Measured (PATCHES.md section 10): RCCL-over-sockets beats RCCL-over-IB at
# every message size on this rail, and this profile matches or beats the full
# RDMA one end to end. A fresh rig can skip the tbv kernel-module build.
source "$HOME/vsh-cluster-env.sh"
export NCCL_IB_DISABLE=1
export VSH_TBV_AR2=0
