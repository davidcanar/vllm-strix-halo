#!/usr/bin/env bash
# GLM-5.3-Flash vLLM launcher (AWQ W4A16, TP=2 over TB4-RDMA, MTP on by
# default). Run INSIDE the vllm-glm container; started by the vsh-glm-manual
# systemd user unit. Sources the canonical RDMA cluster-env, then execs
# vllm serve on glm53_api_port from vsh-config.yaml.
#
# Memory flags, since they are the ones that bite (same lessons as the DS4
# stack — see AlexKGwyn/ds4-vllm host/ds4-vllm-manual-serve.sh):
#   --kv-cache-memory-bytes  Pin KV. Inferring it oversubscribes the box: the
#     AWQ weight cache and RDMA buffers are allocated AFTER the profiling pass.
#     The pool is a fixed-size LRU: it does not grow with --max-model-len.
#   --gpu-memory-utilization  INERT while the pin above is set.
#   --max-num-batched-tokens 512  NOT larger: the sparse-attention indexer
#     workspace scales with batch x context.
#
# Do not add comments inside the backslash-continued `vllm serve` command
# below: a '#' there silently comments out every remaining argument.
set -u
source "$HOME/vsh-cluster-env.${VSH_TRANSPORT:-rdma}.sh"

MODEL_DIR=${VSH_GLM53_MODEL_DIR:?vsh-config.yaml: glm53_model_dir missing}
PORT=${VSH_GLM53_API_PORT:-1235}

# MTP speculative decoding (glm5_next_mtp = GLM-5.3's own NextN drafter,
# weights embedded in the checkpoint — no separate draft download).
SPEC=()
if [ "${VSH_GLM53_MTP_TOKENS:-0}" -gt 0 ]; then
  SPEC=(--speculative-config "{\"method\":\"glm5_next_mtp\",\"num_speculative_tokens\":${VSH_GLM53_MTP_TOKENS}}")
  echo "[vsh-serve] MTP speculative decoding ON (${VSH_GLM53_MTP_TOKENS} draft tokens)"
else
  echo "[vsh-serve] MTP speculative decoding OFF (glm53_mtp_tokens: 0)"
fi

exec vllm serve "$MODEL_DIR" \
  --served-model-name glm-5.3-flash \
  --tensor-parallel-size 2 \
  --distributed-executor-backend ray \
  --quantization compressed-tensors \
  --kv-cache-dtype fp8 \
  --enforce-eager \
  --disable-mm-preprocessor-cache \
  --gpu-memory-utilization ${VSH_GLM53_GPU_UTIL:-0.83} \
  --kv-cache-memory-bytes ${VSH_GLM53_KV_BYTES:-4294967296} \
  --max-model-len "${VSH_GLM53_MAX_CTX:-32768}" \
  --max-num-batched-tokens ${VSH_GLM53_MAX_BATCHED:-512} \
  --trust-remote-code \
  "${SPEC[@]}" \
  --host 0.0.0.0 --port "$PORT"
