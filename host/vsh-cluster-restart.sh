#!/usr/bin/env bash
# Full vllm-strix-halo (glm53) cluster restart: teardown -> Ray on both boxes
# -> vllm serve -> verify. Run from box1 (the Ray head); box2 is driven over
# ssh. Each box's vllm-glm container is verified exec-able (container-heal.sh
# starts/reconciles it) before Ray comes up, so this works from a fresh boot.
#
# Two things this encodes that a bare `systemctl restart` gets wrong (carried
# over from the DS4 stack):
#   1. REAPING: vsh-glm-manual is a transient systemd-run unit supervising a
#      `podman exec` wrapper, NOT the process inside the container. Stopping
#      the unit kills the wrapper and leaves `vllm serve` running, holding its
#      port. We stop the unit, then explicitly kill surviving GLM `vllm serve`
#      procs and confirm zero remain before bringing anything back up.
#   2. RAY WORKER POOL: `ray start` passes --num_prestart_python_workers=
#      --num_cpus straight through; RAY_NUM_CPUS below caps the idle pool.
#
# The reap pattern is scoped to the GLM model dir so the co-resident DS4
# cluster (ds4-vllm, port 1234) is never touched.
set -uo pipefail

# Site specifics (IPs, transport, container, HCA) come from ~/vsh-config.yaml.
eval "$("$HOME/vsh-config" "$HOME/vsh-config.yaml")"
HEAD_IP=${VSH_HEAD_IP:?vsh-config.yaml: head_ip missing}
WORKER_IP=${VSH_WORKER_IP:?vsh-config.yaml: worker_ip missing}
PORT=${VSH_GLM53_API_PORT:-1235}
CTR=${VSH_GLM53_CONTAINER:-vllm-glm}
TRANSPORT=${VSH_TRANSPORT:-rdma}
RAYTMP=$HOME/vsh-ray-tmp
RAY_NUM_CPUS=${RAY_NUM_CPUS:-4}
CENV=$HOME/vsh-cluster-env.$TRANSPORT.sh
SERVE=$HOME/vsh-manual-serve.sh
UNIT=vsh-glm-manual
# Model dir doubles as the reap-pattern discriminator.
MODEL_DIR=${VSH_GLM53_MODEL_DIR:?vsh-config.yaml: glm53_model_dir missing}
# Exports that must reach the env files on BOTH boxes (sourced at ray start).
ENVPASS="export VSH_RDMA_HCA=${VSH_RDMA_HCA:-} VLLM_HOST_IP=${HEAD_IP:?} VSH_GLM53_AITER=${VSH_GLM53_AITER:-0} VSH_GLM53_TBV_AR2=${VSH_GLM53_TBV_AR2:-0};"

[ -f "$CENV" ] || { echo "!! $CENV missing (transport=$TRANSPORT)"; exit 1; }

box2() { timeout "${2:-120}" ssh -o BatchMode=yes "$WORKER_IP" "$1"; }
inbox() { timeout "${2:-120}" podman exec -u 1000:1000 -w "$HOME" "$CTR" bash -lc "$1"; }

echo "== teardown =="
systemctl --user stop "$UNIT.service" 2>/dev/null
systemctl --user reset-failed "$UNIT.service" 2>/dev/null
sleep 4

# The bracket keeps this grep from matching its own command line; the second
# pattern scopes the reap to THIS cluster's model only.
for p in $(ps -eo pid,cmd --no-headers | grep "bin/[v]llm serve" | grep -F "$MODEL_DIR" | awk '{print $1}'); do
  echo "   reaping stranded glm53 vllm serve pid=$p"
  kill "$p" 2>/dev/null; sleep 3
  kill -0 "$p" 2>/dev/null && { kill -9 "$p" 2>/dev/null; sleep 2; }
done
residual=$(ps -eo cmd --no-headers | grep "bin/[v]llm serve" | grep -cF "$MODEL_DIR")
[ "$residual" -eq 0 ] || { echo "!! $residual glm53 vllm serve process(es) still alive -- aborting"; exit 1; }

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
systemd-run --user --unit="$UNIT" --description="vllm-strix-halo GLM-5.3-Flash TP=2" \
  --working-directory="$HOME" \
  /usr/bin/podman exec -u 1000:1000 -w "$HOME" "$CTR" bash -lc \
  "$ENVPASS export VSH_TRANSPORT=$TRANSPORT VSH_GLM53_MODEL_DIR=$MODEL_DIR VSH_GLM53_API_PORT=$PORT VSH_GLM53_MAX_CTX=${VSH_GLM53_MAX_CTX:-32768} VSH_GLM53_KV_BYTES=${VSH_GLM53_KV_BYTES:-4294967296} VSH_GLM53_GPU_UTIL=${VSH_GLM53_GPU_UTIL:-0.83} VSH_GLM53_MAX_BATCHED=${VSH_GLM53_MAX_BATCHED:-512} VSH_GLM53_MTP_TOKENS=${VSH_GLM53_MTP_TOKENS:-3} VSH_GLM53_AITER=${VSH_GLM53_AITER:-0} VSH_GLM53_TBV_AR2=${VSH_GLM53_TBV_AR2:-0}; exec bash $SERVE" >/dev/null 2>&1

# Warm bringup answers in a few minutes; a cold kernel-cache bringup (first
# after a cache wipe) spends ~25 min more in Triton/LLVM compiles.
for _ in $(seq 1 105); do
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
  --setenv=VSH_WARMUP_PORT=$PORT --setenv=VSH_WARMUP_CTX=${VSH_GLM53_WARMUP_CTX:-2048} \
  /usr/bin/python3 "$HOME/vsh-warmup.py" >/dev/null 2>&1 \
  && echo "   warmup dispatched (journalctl --user -u vsh-warmup)" \
  || echo "   warmup dispatch failed (non-fatal)"

echo "== verify =="
journalctl --user -u "$UNIT.service" --no-pager -o cat --since "-20min" 2>/dev/null \
  | grep -aE "GPU KV cache size|Maximum concurrency" | tail -2 | sed 's/^/   /'
rdma=$(journalctl --user -u "$UNIT.service" --no-pager -o cat --since "-20min" 2>/dev/null \
  | grep -aoE "tbv_ar2: rank[0-9] ready \(qpn=[0-9]+ peer_qpn=[0-9]+\)" | head -1)
echo "   RDMA: ${rdma:-!! tbv_ar2 NOT ready -- decode all-reduce is not on RDMA}"
echo "   vllm serve procs: $(ps -eo cmd --no-headers | grep 'bin/[v]llm serve' | grep -cF "$MODEL_DIR") (want 1)"
echo "   MemAvailable: $(awk '/MemAvailable/{printf "%d", $2/1024}' /proc/meminfo)MB"
