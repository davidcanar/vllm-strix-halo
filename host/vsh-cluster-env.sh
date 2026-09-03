#!/usr/bin/env bash
# vsh-cluster-env.sh — canonical env for the vllm-strix-halo TP=2 cluster.
# Sourced by the ray head, the box2 ray worker, and vllm serve (both boxes keep
# an identical copy in each box's home). NCCL/RDMA transport knobs plus the
# gfx1151/RDNA memory settings inherited from the validated DS4 stack.
#
# Why these values: see the comments in AlexKGwyn/ds4-vllm's
# host/ds4-cluster-env.sh (this file is its model-agnostic derivative).

# Stable block-content hashes across restarts (vLLM prefix-cache filenames).
export PYTHONHASHSEED=0
# Bound allocator growth. The caching allocator never returns freed blocks,
# and its GC is off by default, so the pool grows without bound on a UMA box.
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.85
# Silence torch's per-step all_gather_into_tensor FutureWarning (journal spam).
export PYTHONWARNINGS="${PYTHONWARNINGS:+$PYTHONWARNINGS,}ignore::FutureWarning"

# --- fabric: Thunderbolt IP link + RoCE RDMA on top -------------------------
export NCCL_SOCKET_IFNAME=thunderbolt0
export GLOO_SOCKET_IFNAME=thunderbolt0
export NCCL_IB_HCA=usb4_rdma
export NCCL_IB_GID_INDEX=1
export NCCL_IB_DISABLE=0
# gfx1151 has no GPUDirect: RCCL host-stages the big (prefill) all-reduces.
# That is expected, not a fault.
export NCCL_NET_GDR_LEVEL=0
export NCCL_IB_TIMEOUT=23
export NCCL_PROTO=LL
export NCCL_ALGO=Ring
export NCCL_IB_RETRY_CNT=7
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=2400
export TORCH_NCCL_ENABLE_MONITORING=0
export NCCL_TIMEOUT_MS=2400000

# --- ray --------------------------------------------------------------------
export RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1
export RAY_memory_monitor_refresh_ms=0
export RAY_memory_usage_threshold=0.99

# Persistent JIT caches (defaults land in /tmp and recompile every boot).
export TORCHINDUCTOR_CACHE_DIR="$HOME/.cache/torchinductor"
export TRITON_CACHE_DIR="$HOME/.triton/cache"

# --- GPU / ROCm -------------------------------------------------------------
export HIP_VISIBLE_DEVICES=0
# DS4 ran aiter-less on gfx1151; flip via glm53_aiter in vsh-config.yaml.
export VLLM_ROCM_USE_AITER=${VSH_GLM53_AITER:-0}
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800
# Blocking (interrupt-based) GPU waits instead of busy-poll. ROCm 10 supports
# this natively (HSA_ENABLE_INTERRUPT) — no ROCr rebuild needed, unlike the
# DS4 ROCm 7.14 stack.
export HSA_ENABLE_INTERRUPT=${VSH_HSA_INTERRUPT:-1}

# --- TB4-RDMA all-reduce (VSH_TBV_AR2 hook, patched cuda_communicator) ------
export VSH_TBV_AR2=${VSH_TBV_AR2:-1}
export VSH_TBV2_PEER_IP=${VSH_TBV2_PEER_IP:-${VSH_HEAD_IP:-10.0.2.1}}
export VSH_TBV_AR_GPU=${VSH_TBV_AR_GPU:-1}
# Propagate VSH_* to box2 ray workers (not in ray's default copy prefixes).
export VLLM_RAY_EXTRA_ENV_VAR_PREFIXES_TO_COPY=VSH_
