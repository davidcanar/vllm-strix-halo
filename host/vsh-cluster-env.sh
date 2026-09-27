#!/usr/bin/env bash
# vsh-cluster-env.sh — canonical env for the vllm-strix-halo TP=2 cluster.
# Sourced by the ray head, the box2 ray worker, and vllm serve (both boxes keep
# an identical copy in each box's home). General NCCL/memory/ray knobs; the
# transport-specific pieces (odl plugin vs plain sockets) live in
# vsh-cluster-env.odl.sh / vsh-cluster-env.tcp.sh.
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

# --- fabric: the control plane always rides the Thunderbolt IP link --------
# (tbnet / thunderbolt0 coexists with the OdinLink driver). The RCCL data
# transport is chosen by the transport profile: odl (the OdinLink net plugin)
# or tcp (plain sockets over thunderbolt0).
export NCCL_SOCKET_IFNAME=thunderbolt0
export GLOO_SOCKET_IFNAME=thunderbolt0
# gfx1151 has no GPUDirect: RCCL host-stages the big (prefill) all-reduces.
# That is expected, not a fault. (The odl net plugin is host-pointer based
# for the same reason.)
export NCCL_NET_GDR_LEVEL=0
export NCCL_ALGO=Ring
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
# Native crash diagnosis: print the Python stack on SIGSEGV/SIGABRT inside
# kernels (the signal handler is a no-op unless the fault is on a Python
# frame, but it is what we have without a debugger).
export PYTHONFAULTHANDLER=1
# aiter ON: the glm5next sparse-attention indexer's ROCm path requires it
# (VLLM_ROCM_USE_AITER). aiter's MoE kernels do not support gfx1151, so MoE
# stays on the triton path (VLLM_ROCM_USE_AITER_MOE=0). Both are also baked
# into the image ENV; they are re-exported here to stay tunable.
export VLLM_ROCM_USE_AITER=${VSH_GLM53_AITER:-1}
export VLLM_ROCM_USE_AITER_MOE=${VSH_GLM53_AITER_MOE:-0}
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800
# Blocking (interrupt-based) GPU waits instead of busy-poll. ROCm 10 supports
# this natively (HSA_ENABLE_INTERRUPT) — no ROCr rebuild needed, unlike the
# DS4 ROCm 7.14 stack.
export HSA_ENABLE_INTERRUPT=${VSH_HSA_INTERRUPT:-1}

# --- decode all-reduce + control plane over OdinLink -------------------------
# The odl_ar2 hook (patched cuda_communicator, env DS4_ODL_AR2) and the
# odl_mq data plane (patched shm_broadcast, fail-open) are activated by the
# odl transport profile; see vsh-cluster-env.odl.sh.
#
# Propagate VSH_* + DS4_* to box2 ray workers (not in ray's default copy
# prefixes). DS4_ODL_AR2 and ODL2_* must reach the worker (rank 1) too.
export VLLM_RAY_EXTRA_ENV_VAR_PREFIXES_TO_COPY="VSH_ DS4_ ODL2_"

# --- tuned Triton fused-MoE tile configs (gfx1151) --------------------------
# Upstream ships no config for AMD_Radeon_8060S, and the int4_w4a16 path does
# not use the normal tile heuristic: get_moe_wna16_block_config() hardcodes
# BLOCK_SIZE_K=32, i.e. 16 bytes of packed int4 weight per K step, which runs
# the GLM MoE at ~38 GB/s against a ~214 GB/s roofline. The tuned tiles in this
# folder are ~2-3x faster at the decode shapes. Checked before vLLM's built-in
# configs/, so nothing in site-packages is modified.
export VLLM_TUNED_CONFIG_FOLDER=${VSH_MOE_CONFIG_DIR:-$HOME/vsh-moe-configs}
