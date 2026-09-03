#!/usr/bin/env bash
# Full vllm-strix-halo (glm53) cluster teardown: serve unit -> stranded vllm
# serve procs -> ray on both boxes. The stop half of vsh-cluster-restart.sh,
# for callers that need the stack DOWN rather than restarted. Idempotent.
#
# The explicit process reap exists because vsh-glm-manual is a transient unit
# supervising a `podman exec` wrapper, not the vllm serve inside the container:
# stopping the unit alone strands the server, still holding the API port.
# Scoped to the GLM model dir so the co-resident DS4 cluster is never touched.
set -uo pipefail

# Teardown must work even with a broken/missing config -- fall back to defaults.
eval "$("$HOME/vsh-config" "$HOME/vsh-config.yaml" 2>/dev/null)" 2>/dev/null || true
WORKER_IP=${VSH_WORKER_IP:-10.0.2.2}
CTR=${VSH_GLM53_CONTAINER:-vllm-glm}
UNIT=vsh-glm-manual
MODEL_DIR=${VSH_GLM53_MODEL_DIR:-/home/davidcanar/models/GLM-5.3-Flash-AWQ-W4A16}

systemctl --user stop "$UNIT.service" 2>/dev/null
systemctl --user reset-failed "$UNIT.service" 2>/dev/null
sleep 2

# The bracket keeps this grep from matching its own command line.
for p in $(ps -eo pid,cmd --no-headers | grep "bin/[v]llm serve" | grep -F "$MODEL_DIR" | awk '{print $1}'); do
  echo "[vsh-cluster-down] reaping glm53 vllm serve pid=$p"
  kill "$p" 2>/dev/null; sleep 3
  kill -0 "$p" 2>/dev/null && { kill -9 "$p" 2>/dev/null; sleep 2; }
done

timeout 60 podman exec -u 1000:1000 -w "$HOME" "$CTR" \
  bash -lc 'ray stop --force >/dev/null 2>&1' 2>/dev/null
timeout 60 ssh -o BatchMode=yes -o ConnectTimeout=6 "$WORKER_IP" \
  "podman exec -u 1000:1000 -w \$HOME $CTR bash -lc 'ray stop --force >/dev/null 2>&1'" 2>/dev/null

echo "[vsh-cluster-down] done"
exit 0
