#!/usr/bin/env bash
# DeepSeek-V4-Flash (Vision-Exp) vLLM launcher on the vllm-strix-halo image
# (vLLM 0.31.0.dev0+git73859fec). Run INSIDE the vllm-glm container; started
# by the vsh-ds4-manual systemd user unit. Sources the canonical RDMA
# cluster-env, then execs vllm serve on ds4_api_port from vsh-config.yaml.
#
# Everything engine-side this checkpoint needs ships in the 0.31.0 pin:
# DeepseekV4ForCausalLM / DeepseekV4ForConditionalGeneration (common/vl_model),
# the DSpark drafter (amd/dspark.py), the deepseek_v4 tokenizer + tool/reasoning
# parsers, and the OdinLink hooks (vsh-odl-ar2 / vsh-odl-mq patches). The old
# AlexKGwyn/ds4-vllm image additionally carried gfx1151 perf modules
# (ds4_tl_indexer, ds4_moe_hip, ...) and the distributed fs_lru disk-KV tier;
# those are NOT in upstream and are not ported here -- this script serves
# without the disk KV cache and on upstream's native ROCm op paths.
#
# Memory flags, since they are the ones that bite (see vsh-manual-serve.sh and
# the old host/ds4-vllm-manual-serve.sh for the full lessons):
#   --kv-cache-memory-bytes  Pin KV; --gpu-memory-utilization is INERT while
#     it is set. The pool is a fixed-size LRU and does not grow with
#     --max-model-len. bf16 KV ≈ 9.3 B/token on this model, so the 6 GiB pin
#     holds ~650K tokens: one 512K-context session with prefix-cache slack.
#   --max-num-batched-tokens 512  NOT 2048; the indexer/top-k workspace scales
#     with batch x context and 2048 costs ~10 GiB more at 256K.
#
# KV dtype: auto (bf16). The old ds4 engine ran fp8 KV, but on this build the
# ROCM_AITER_MLA_SPARSE fp8 decode path asserts on the first request (same
# failure measured on GLM-5.3, vsh-config 2026-09-28 note; V4 uses the same
# aiter sparse backend). Retest fp8 after an upstream fix; ds4_kv_dtype knob.
#
# Do not add comments inside the backslash-continued `vllm serve` command
# below: a '#' there silently comments out every remaining argument.
set -u
source "$HOME/vsh-cluster-env.${VSH_TRANSPORT:-tcp}.sh"
# gfx1151: aiter linear fp8 (GEMM + activation quant) is MI300-only on this
# build (no fp8 tensors in its pybind dtype table; Triton fp8 dot unsupported).
# VLLM_ROCM_USE_AITER stays 1 for the sparse-attention/indexer aiter ops.
export VLLM_ROCM_USE_AITER_LINEAR=0
# DSV4 indexer QAT: official Hadamard-rotation basis (pairs K+Q; the
# 0.31 kernels carry the port behind this flag - see
# vsh-ds4-idx-hadamard-K/Q-031 patches).
export DS4_IDX_OFFICIAL=${VSH_DS4_IDX_OFFICIAL:-1}
echo "[vsh-ds4-serve] HOME=$HOME VLLM_ROCM_USE_AITER=$VLLM_ROCM_USE_AITER DS4_ODL_AR2=${DS4_ODL_AR2:-unset} NCCL_NET_PLUGIN=${NCCL_NET_PLUGIN:-unset}"

MODEL_DIR=${VSH_DS4_MODEL_DIR:?vsh-config.yaml: ds4_model_dir missing}
PORT=${VSH_DS4_API_PORT:-1234}

# Vision checkpoint detection. DeepSeek-V4-Flash-Vision-Exp's config.json
# declares the TEXT architecture, so vLLM would load it text-only; the marker
# is vision_n_layers. When present, select upstream's multimodal wrapper
# (vllm.models.deepseek_v4.common.vl_model.DeepseekV4ForConditionalGeneration;
# the old engine's class was ...V4VForConditionalGeneration). The same
# override declares num_nextn_predict_layers=1: Vision-Exp declares its
# three-stage DSpark drafter as 3 where the text checkpoint declares 1, the
# drafter is one predictor either way (stages are internal), and 1 keeps
# vLLM's num_speculative_tokens check happy on the MTP-5 shape.
VISION_LAYERS=$(/opt/venv/bin/python3 - "$MODEL_DIR" <<'PYEOF'
import json, os, sys
path = os.path.join(sys.argv[1], "config.json")
print(int(json.load(open(path)).get("vision_n_layers") or 0))
PYEOF
) || { echo "[vsh-ds4-serve] WARNING: could not read config.json -- serving as a text checkpoint"; VISION_LAYERS=0; }
HF_OVERRIDE_ARGS=()
if [ "${VISION_LAYERS:-0}" -gt 0 ] && [ "${VSH_DS4_NO_VL:-0}" != "1" ]; then
  echo "[vsh-ds4-serve] vision checkpoint (vision_n_layers=$VISION_LAYERS): multimodal wrapper ON"
  HF_OVERRIDE_ARGS=(--hf-overrides '{"architectures":["DeepseekV4ForConditionalGeneration"],"num_nextn_predict_layers":1}')
fi
# VSH_DS4_NO_VL=1: diagnostic - serve the checkpoint text-only (DeepseekV4ForCausalLM,
# the arch config.json declares) to A/B the multimodal wrapper's sentinel/embedding merge.
[ "${VSH_DS4_NO_VL:-0}" = "1" ] && echo "[vsh-ds4-serve] TEXT-ONLY diagnostic mode (VSH_DS4_NO_VL=1): wrapper OFF"

# DSpark MTP (in-checkpoint drafter). enforce_eager inside the spec config is
# LOAD-BEARING while graphs are off: the drafter manages its own step-0 graphs
# and cannot be captured by the runner's wrapper. Padded drafter batches +
# async scheduling is the old stack's validated default (DS4_ASYNC_SCHED=1).
SPEC=()
if [ "${VSH_DS4_MTP_TOKENS:-5}" -gt 0 ]; then
  # method dspark = upstream in-checkpoint DSpark drafter (DSparkDeepseekV4ForCausalLM).
  # num_speculative_tokens must match dspark_block_size (5). The old stack ran
  # deepseek_mtp k=5 under its own contract; upstream rejects that (n_predict=3
  # divisibility). Async scheduling OFF until validated on this path.
  if [ "${VSH_DS4_ASYNC_SCHED:-0}" = "1" ]; then
    SPEC=(--speculative-config "{\"method\":\"dspark\",\"num_speculative_tokens\":${VSH_DS4_MTP_TOKENS:-5},\"enforce_eager\":true}" --async-scheduling)
  else
    SPEC=(--speculative-config "{\"method\":\"dspark\",\"num_speculative_tokens\":${VSH_DS4_MTP_TOKENS:-5},\"enforce_eager\":true}")
  fi
  echo "[vsh-ds4-serve] DSpark MTP ON (method=dspark, k=${VSH_DS4_MTP_TOKENS:-5})"
else
  echo "[vsh-ds4-serve] DSpark MTP OFF (ds4_mtp_tokens: 0)"
fi

# Tool calling + reasoning separation (upstream deepseek_v4 parsers).
TOOLS=(--enable-auto-tool-choice --tool-call-parser deepseek_v4 --reasoning-parser deepseek_v4 --default-chat-template-kwargs '{"thinking":true,"reasoning_effort":"high"}')
echo "[vsh-ds4-serve] tool-call + reasoning parsers ON (deepseek_v4)"

# KV cache dtype knob (see header: fp8 asserts on this build's aiter path).
KVD=()
if [ -n "${VSH_DS4_KV_DTYPE:-}" ] && [ "${VSH_DS4_KV_DTYPE}" != "auto" ]; then
  KVD=(--kv-cache-dtype "$VSH_DS4_KV_DTYPE")
  echo "[vsh-ds4-serve] KV cache dtype: $VSH_DS4_KV_DTYPE"
fi

# Eager by default: the GLM lesson on this build (capture OK, first replay
# wedges the MTP drafter on the hybrid attention backends) plus the old
# stack's drafter-capture constraint. ds4_enforce_eager: 0 re-enables the
# old stack's PIECEWISE graphs (unproven on 0.31.0 for V4).
EAGER=(--enforce-eager)
if [ "${VSH_DS4_ENFORCE_EAGER:-1}" != "1" ]; then
  EAGER=()
  export VLLM_USE_BREAKABLE_CUDAGRAPH=1
  echo "[vsh-ds4-serve] enforce-eager OFF (PIECEWISE graphs, breakable capture)"
fi

# gfx1151 (FNUZ aiter): upstream fused norm/act fp8-quant fusions feed
# torch.float8_e4m3fn activations into aiter group_fp8_quant (rocm.py
# _bpre_attn_gemm), which asserts on non-FNUZ dtypes. Same fn->fnuz seam as
# the vsh-fp8-fnuz-mqa patch, different call site. Disable the fusions for
# this model until an upstream fnuz seam lands (small perf cost).
exec vllm serve "$MODEL_DIR" \
  --served-model-name deepseek-v4-flash \
  "${COMP[@]}" \
  "${HF_OVERRIDE_ARGS[@]}" \
  --tensor-parallel-size 2 \
  --distributed-executor-backend ray \
  "${EAGER[@]}" \
  "${KVD[@]}" \
  --gpu-memory-utilization ${VSH_DS4_GPU_UTIL:-0.83} \
  --kv-cache-memory-bytes ${VSH_DS4_KV_BYTES:-6442450944} \
  --max-model-len "${VSH_DS4_MAX_CTX:-524288}" \
  --max-num-batched-tokens ${VSH_DS4_MAX_BATCHED:-512} \
  --trust-remote-code \
  --tokenizer-mode deepseek_v4 \
  --override-generation-config '{"temperature":0.0,"top_p":1.0}' \
  "${TOOLS[@]}" \
  "${SPEC[@]}" \
  --kernel-config '{"linear_backend":"triton"}' \
  --host 0.0.0.0 --port "$PORT"
