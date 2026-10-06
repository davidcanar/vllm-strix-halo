#!/usr/bin/env bash
# Full vllm-strix-halo DeepSeek-V4-Flash (ds4) cluster restart: teardown ->
# Ray on both boxes -> vllm serve -> verify. Run from box1 (the Ray head);
# box2 is driven over ssh. Adapted from vsh-cluster-restart.sh (glm53); the
# same two encoded lessons apply:
#   1. REAPING: vsh-ds4-manual is a transient systemd-run unit supervising a
#      `podman exec` wrapper, NOT the process inside the container. Stop the
#      unit, then explicitly kill surviving DS4 `vllm serve` procs.
#   2. RAY WORKER POOL: --num_prestart_python_workers passthrough, RAY_NUM_CPUS
#      caps the idle pool.
#
# The reap is scoped to the DS4 model dir so the co-resident GLM cluster
# (vsh-glm, port 1235) is never touched. Serves from the SAME vllm-glm
# container/image as glm53 (vLLM 0.31.0) -- see vsh-ds4-manual-serve.sh.
#
# Env-file reuse note: vsh-cluster-env.odl.sh derives DS4_ODL_AR2 and the
# aiter gate from VSH_GLM53_* names, so this script feeds the DS4 knobs into
# those names (values only; the GLM cluster passes its own when IT starts).
set -uo pipefail

# Site specifics (IPs, transport, container, HCA) come from ~/vsh-config.yaml.
eval "$("$HOME/vsh-config" "$HOME/vsh-config.yaml")"
HEAD_IP=${VSH_HEAD_IP:?vsh-config.yaml: head_ip missing}
WORKER_IP=${VSH_WORKER_IP:?vsh-config.yaml: worker_ip missing}
PORT=${VSH_DS4_API_PORT:-1234}
CTR=${VSH_DS4_CONTAINER:-vllm-glm}
TRANSPORT=${VSH_TRANSPORT:-odl}
RAYTMP=$HOME/vsh-ray-tmp
RAY_NUM_CPUS=${RAY_NUM_CPUS:-4}
CENV=$HOME/vsh-cluster-env.$TRANSPORT.sh
SERVE=$HOME/vsh-ds4-manual-serve.sh
UNIT=vsh-ds4-manual
# Model dir doubles as the reap-pattern discriminator.
MODEL_DIR=${VSH_DS4_MODEL_DIR:?vsh-config.yaml: ds4_model_dir missing}
# Exports that must reach the env files on BOTH boxes (sourced at ray start).
# VSH_ODL_RANK1_IP: the odl_ar2 rendezvous target — rank 1 is the box2 worker.
# VSH_GLM53_AITER / VSH_GLM53_ODL_AR2 are the NAMES the shared env files read;
# we feed them the ds4 profile's values.
ENVPASS="export VSH_ODL_RANK1_IP=${WORKER_IP:?} VLLM_HOST_IP=${HEAD_IP:?} VSH_GLM53_AITER=${VSH_DS4_AITER:-1} VSH_GLM53_ODL_AR2=${VSH_DS4_ODL_AR2:-1} VLLM_ROCM_USE_AITER_LINEAR=0 DS4_IDX_OFFICIAL=${VSH_DS4_IDX_OFFICIAL:-1} VSH_GLM53_W8A16=0 VSH_W8A16_LMHEAD=0 VSH_FP8_GEMV=${VSH_FP8_GEMV:-1} VSH_SPARSE_ATTN_SPLIT=${VSH_SPARSE_ATTN_SPLIT:-1};"  # ds4/gfx1151: aiter linear fp8 (GEMM+quant) is MI300-only; stock Triton paths work here

[ -f "$CENV" ] || { echo "!! $CENV missing (transport=$TRANSPORT)"; exit 1; }

box2() { timeout "${2:-120}" ssh -o BatchMode=yes "$WORKER_IP" "$1"; }
inbox() { timeout "${2:-120}" podman exec -u 1000:1000 -w "$HOME" "$CTR" bash -lc "$1"; }

# Both boxes source their OWN copy of $CENV at ray start, so a divergent copy
# silently gives the two TP ranks different settings (2026-10-05: box2 defaulted
# VSH_W8A16 to 0 -> rank 0 ran int8 linears, rank 1 bf16). Refuse to start.
_cenv1=$(md5sum < "$CENV" | cut -c1-32)
_cenv2=$(box2 "md5sum < $CENV" 30 2>/dev/null | cut -c1-32)
if [ -z "$_cenv2" ] || [ "$_cenv1" != "$_cenv2" ]; then
  echo "!! $CENV differs between box1 ($_cenv1) and box2 (${_cenv2:-unreadable}); sync it: scp $CENV $WORKER_IP:"
  exit 1
fi

echo "== teardown =="
systemctl --user stop "$UNIT.service" 2>/dev/null
systemctl --user reset-failed "$UNIT.service" 2>/dev/null
sleep 4

# The bracket keeps this grep from matching its own command line; the second
# pattern scopes the reap to THIS cluster's model only.
for p in $(ps -eo pid,cmd --no-headers | grep "bin/[v]llm serve" | grep -F "$MODEL_DIR" | awk '{print $1}'); do
  echo "   reaping stranded ds4 vllm serve pid=$p"
  kill "$p" 2>/dev/null; sleep 3
  kill -0 "$p" 2>/dev/null && { kill -9 "$p" 2>/dev/null; sleep 2; }
done
residual=$(ps -eo cmd --no-headers | grep "bin/[v]llm serve" | grep -cF "$MODEL_DIR")
[ "$residual" -eq 0 ] || { echo "!! $residual ds4 vllm serve process(es) still alive -- aborting"; exit 1; }

inbox 'ray stop --force >/dev/null 2>&1' >/dev/null 2>&1
box2 "podman exec -u 1000:1000 -w \$HOME $CTR bash -lc 'ray stop --force >/dev/null 2>&1'" >/dev/null 2>&1

for _ in $(seq 1 40); do
  u1=$(free -g | awk '/^Mem:/{print $3}')
  u2=$(box2 "free -g | awk '/^Mem:/{print \$3}'" 20 2>/dev/null || echo 99)
  [ "${u1:-99}" -lt 45 ] && [ "${u2:-99}" -lt 45 ] && break
  sleep 5
done
echo "   drained: box1=${u1}G box2=${u2}G swap=$(free -m | awk '/^Swap:/{print $3}')MB"

echo "== containers =="
"$HOME/container-heal.sh" "$CTR" 2>&1 | sed 's/^/   /'
box2 "\$HOME/container-heal.sh $CTR" 60 2>/dev/null | sed 's/^/   /'
inbox true 20 >/dev/null 2>&1 || { echo "!! box1 $CTR container not exec-able"; exit 1; }
box2 "podman exec $CTR true" 20 >/dev/null 2>&1 || { echo "!! box2 $CTR container not exec-able"; exit 1; }
echo "   $CTR container exec-able on both boxes"

echo "== ray =="
# --include-dashboard is head-only; ray PANICs if it is passed to a worker.
RAYFLAGS="--num-gpus=1 --num-cpus=$RAY_NUM_CPUS --temp-dir=$RAYTMP"
HEADFLAGS="$RAYFLAGS --include-dashboard=false"
if ! out=$(inbox "$ENVPASS source $CENV; ray start --head --node-ip-address=$HEAD_IP --port=6379 $HEADFLAGS" 180 2>&1); then
  echo "!! box1 ray start failed:"; echo "$out" | tail -5 | sed 's/^/     /'; exit 1
fi
echo "   box1 head up"
if ! out=$(box2 "podman exec -u 1000:1000 -w \$HOME $CTR bash -lc '$ENVPASS export VLLM_HOST_IP=$WORKER_IP; source $CENV; ray start --address=$HEAD_IP:6379 --node-ip-address=$WORKER_IP $RAYFLAGS'" 180 2>&1); then
  echo "!! box2 ray start failed:"; echo "$out" | tail -5 | sed 's/^/     /'; exit 1
fi
echo "   box2 worker up"

for _ in $(seq 1 8); do
  g=$(inbox 'ray status 2>/dev/null' 60 | grep -oE "[0-9.]+/[0-9.]+ GPU")
  [ "${g#*/}" = "2.0 GPU" ] && break
  sleep 4
done
[ "${g#*/}" = "2.0 GPU" ] || { echo "!! ray never reached 2 GPUs (saw '${g:-none}') -- aborting"; exit 1; }
echo "   ray: $g  (prestart python workers capped at $RAY_NUM_CPUS/node, dashboard off)"

echo "== serve =="
systemd-run --user --unit="$UNIT" --description="vllm-strix-halo DeepSeek-V4-Flash TP=2" \
  --working-directory="$HOME" \
  /usr/bin/podman exec -u 1000:1000 -w "$HOME" "$CTR" bash -lc \
  "$ENVPASS export VSH_TRANSPORT=$TRANSPORT VSH_DS4_MODEL_DIR=$MODEL_DIR VSH_DS4_API_PORT=$PORT VSH_DS4_MAX_CTX=${VSH_DS4_MAX_CTX:-524288} VSH_DS4_KV_BYTES=${VSH_DS4_KV_BYTES:-6442450944} VSH_DS4_KV_DTYPE=${VSH_DS4_KV_DTYPE:-auto} VSH_DS4_GPU_UTIL=${VSH_DS4_GPU_UTIL:-0.83} VSH_DS4_MAX_BATCHED=${VSH_DS4_MAX_BATCHED:-512} VSH_DS4_MTP_TOKENS=${VSH_DS4_MTP_TOKENS:-5} VSH_DS4_ASYNC_SCHED=${VSH_DS4_ASYNC_SCHED:-1} VSH_DS4_ENFORCE_EAGER=${VSH_DS4_ENFORCE_EAGER:-1} VSH_DS4_CG_MODE=${VSH_DS4_CG_MODE:-PIECEWISE} VSH_DS4_NO_VL=${VSH_DS4_NO_VL:-0}; exec bash $SERVE" >/dev/null 2>&1

# Warm bringup answers in a few minutes; a cold kernel-cache bringup (first
# DS4 boot on this image spends the diff in Triton/LLVM compiles) can take
# much longer -- the old ds4 stack budgeted 45 min for exactly this.
for _ in $(seq 1 140); do
  code=$(curl -s -o /dev/null -m 5 -w "%{http_code}" "http://127.0.0.1:$PORT/v1/models" 2>/dev/null)
  [ "$code" = "200" ] && break
  systemctl --user is-active --quiet "$UNIT.service" || { echo "!! unit died during boot"; exit 1; }
  sleep 20
done
[ "$code" = "200" ] || { echo "!! API never came up"; exit 1; }

# Warm the JIT kernels so the first real request runs at full speed.
# Backgrounded transient unit; best-effort.
systemctl --user reset-failed vsh-warmup.service 2>/dev/null
systemd-run --user --collect --unit=vsh-warmup \
  --setenv=VSH_WARMUP=0 \
  --setenv=VSH_WARMUP_PORT=$PORT --setenv=VSH_WARMUP_MODEL=deepseek-v4-flash \
  --setenv=VSH_WARMUP_CTX=${VSH_DS4_WARMUP_CTX:-32768} \
  /usr/bin/python3 "$HOME/vsh-warmup.py" >/dev/null 2>&1 \
  && echo "   warmup dispatched (journalctl --user -u vsh-warmup)" \
  || echo "   warmup dispatch failed (non-fatal)"

echo "== verify =="
journalctl --user -u "$UNIT.service" --no-pager -o cat --since "-45min" 2>/dev/null \
  | grep -aE "GPU KV cache size|Maximum concurrency" | tail -2 | sed 's/^/   /'
rdma=$(journalctl --user -u "$UNIT.service" --no-pager -o cat --since "-45min" 2>/dev/null \
  | grep -aoE "odl_ar2: rank[0-9] ready" | sort -u | paste -sd' ' -)
mq=$(journalctl --user -u "$UNIT.service" --no-pager -o cat --since "-45min" 2>/dev/null \
  | grep -aoE "odl_mq: (writer up|reader up)" | sort -u | paste -sd' ' -)
echo "   OdinLink decode AR: ${rdma:-!! odl_ar2 NOT ready -- decode all-reduce is not on the fast path}"
echo "   OdinLink ctrl plane: ${mq:-zmq (odl_mq not engaged)}"
echo "   vllm serve procs: $(ps -eo cmd --no-headers | grep 'bin/[v]llm serve' | grep -cF "$MODEL_DIR") (want 1)"
echo "   MemAvailable: $(awk '/MemAvailable/{printf "%d", $2/1024}' /proc/meminfo)MB"
