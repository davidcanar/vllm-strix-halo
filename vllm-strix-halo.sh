#!/usr/bin/env bash
# =============================================================================
# vllm-strix-halo — start / stop / status / logs for the 2-box vLLM cluster
#
#   Machine A (this machine, 10.0.2.1): Ray head + vLLM OpenAI API server
#   Machine B (10.0.2.2):              Ray worker, TP rank 1
#
#   Link : Thunderbolt/USB4 RoCE-RDMA (usb4_rdma0, GID index 1) — the TP
#          all-reduce runs over RCCL-over-IB (prefill) and the tbv_ar2
#          natives (decode) on this rail.
#
#   Models (both first-class):
#     glm53  GLM-5.3-Flash (AWQ W4A16, MTP)  — image vllm-strix-halo:local,
#            container "vllm-glm", API :1235, systemd unit vsh-glm
#     ds4    DeepSeek-V4-Flash (DSpark MTP)  — delegates to the deployed
#            AlexKGwyn/ds4-vllm stack (image ds4-vllm-patched:local,
#            container "vllm", API :1234, systemd unit ds4-vllm)
#
#   Site values (IPs, ports, HCA, model dirs, KV knobs) live in
#   ~/vsh-config.yaml (deployed by host/deploy.sh) - edit there, not here.
#
# Usage:
#   ./vllm-strix-halo.sh                 # glm53 start (default model)
#   ./vllm-strix-halo.sh start           # start the default model cluster
#   ./vllm-strix-halo.sh stop            # stop the default model cluster
#   ./vllm-strix-halo.sh status          # both models: unit / API / RDMA / Ray
#   ./vllm-strix-halo.sh logs            # follow the default model's journals
#   ./vllm-strix-halo.sh glm53 start     # explicit model + action
#   ./vllm-strix-halo.sh ds4 status      # DeepSeek cluster status
# =============================================================================
set -uo pipefail

# ----------------------------- configuration --------------------------------
WORKER_HOST="10.0.2.2"          # box2, driven over ssh (BatchMode)
UNIT_FILE_VSH="$HOME/.config/systemd/user/vsh-glm.service"
CONFIG_YAML="$HOME/vsh-config.yaml"
RDMA_DEV="usb4_rdma0"

# systemd --user over a non-login ssh shell needs this
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"

LOAD_TIMEOUT="2700"             # max seconds to wait for the API (45 min; cold
                                # Triton/LLVM compiles can add ~25 min on the
                                # very first bring-up after a cache wipe)
# -----------------------------------------------------------------------------

log()  { printf '\033[1;32m[vsh]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[vsh]\033[0m %s\n' "$*" >&2; }
die()  { warn "$*"; exit 1; }

remote() { timeout "${2:-30}" ssh -o BatchMode=yes "$WORKER_HOST" "$1"; }

# A RoCEv2 IPv4 GID like 0000:...:ffff:0a00:0201 is "zero" only if nothing
# besides colons and zeros remains.
gid_nonzero() {
    local g
    g=$(cat "/sys/class/infiniband/${RDMA_DEV}/ports/1/gids/1" 2>/dev/null || true)
    [[ -n "$(echo "$g" | tr -d ':0')" ]]
}

# Parse [model] [action]. First arg is a model name if it matches, otherwise
# an action (default: start). Second arg is the action when the first was a
# model. The default model comes from ~/vsh-config.yaml (model: key).
parse_args() {
    MODEL=""
    ACTION=""
    for a in "$@"; do
        case "$a" in
            glm53|ds4) MODEL="$a" ;;
            start|stop|status|logs) ACTION="$a" ;;
            -h|--help|help) ACTION="help" ;;
            *) die "Unknown argument '$a'. Usage: $0 [glm53|ds4] [start|stop|status|logs]" ;;
        esac
    done
    ACTION="${ACTION:-start}"
    if [ -z "$MODEL" ]; then
        MODEL=$(grep -E "^model:" "$CONFIG_YAML" 2>/dev/null | awk '{print $2}' || true)
        MODEL="${MODEL:-glm53}"
    fi
    case "$MODEL" in glm53|ds4) ;; *) die "Unknown model '$MODEL'." ;; esac
}

load_site() {
    eval "$("$HOME/vsh-config" "$CONFIG_YAML" 2>/dev/null)" 2>/dev/null || true
    GLM_PORT=${VSH_GLM53_API_PORT:-1235}
    GLM_CTR=${VSH_GLM53_CONTAINER:-vllm-glm}
    GLM_MODEL_DIR=${VSH_GLM53_MODEL_DIR:-/home/davidcanar/models/GLM-5.3-Flash-AWQ-W4A16}
    DS4_PORT=${VSH_DS4_API_PORT:-1234}
    DS4_UNIT=${VSH_DS4_UNIT:-ds4-vllm}
}

# =============================================================================
do_start_glm53() {
    # ---- preflight checks ---------------------------------------------------
    command -v systemctl >/dev/null 2>&1 || die "systemctl not found."
    [[ -f "$UNIT_FILE_VSH" ]] || die "systemd unit not installed at $UNIT_FILE_VSH (run host/deploy.sh)."
    [[ -f "$CONFIG_YAML" ]] || die "~/vsh-config.yaml missing (site config)."
    [[ -f "$HOME/vsh-cluster-env.rdma.sh" ]] || die "~/vsh-cluster-env.rdma.sh missing on head."
    [[ -f "$GLM_MODEL_DIR/config.json" ]] || die "Model not found: $GLM_MODEL_DIR"
    remote "test -f \$HOME/vsh-cluster-env.rdma.sh" >/dev/null 2>&1 \
        || die "vsh-cluster-env.rdma.sh missing on ${WORKER_HOST} (env files must be byte-identical on both boxes)."
    remote "test -f $GLM_MODEL_DIR/config.json" >/dev/null 2>&1 \
        || die "Model weights not present on ${WORKER_HOST}."

    # Thunderbolt IP link
    ip -brief addr show thunderbolt0 2>/dev/null | grep -q UP \
        || die "thunderbolt0 is down - check the USB4 cable / peer box."
    # RDMA rail + RoCEv2 GID (index 1 carries the 10.0.2.x IP)
    rdma link show "${RDMA_DEV}/1" 2>/dev/null | grep -q "state ACTIVE" \
        || warn "RDMA rail ${RDMA_DEV} not ACTIVE - bring-up continues on the TCP fallback path."
    gid_nonzero || warn "RoCE GID index 1 is empty (no IP on the rail) - RDMA all-reduce may not engage."

    # serving containers alive on both boxes
    "$HOME/container-heal.sh" "$GLM_CTR" 2>&1 | sed 's/^/    /'
    remote "\$HOME/container-heal.sh $GLM_CTR" 60 2>/dev/null | sed 's/^/    /'
    podman exec "$GLM_CTR" true >/dev/null 2>&1 || die "Container '$GLM_CTR' not exec-able on head."
    remote "podman exec $GLM_CTR true" 30 >/dev/null 2>&1 || die "Container '$GLM_CTR' not exec-able on ${WORKER_HOST}."

    # ---- run the validated cluster bring-up ---------------------------------
    log "Starting GLM-5.3-Flash cluster (teardown -> containers -> Ray TP=2 -> vllm serve)..."
    log "Progress logs: journalctl --user -u vsh-glm -u vsh-glm-manual"

    journalctl --user -f -n 0 -q -o cat -u "vsh-glm.service" -u "vsh-glm-manual.service" 2>/dev/null &
    local jl=$!

    # restart, not start: the unit is Type=oneshot + RemainAfterExit, so plain
    # `start` is a silent NO-OP while it is already active and the health check
    # below would then pass against the OLD server.
    systemctl --user restart "vsh-glm.service" &
    local sc=$!

    log "Waiting for the API (model load ~4 min warm; up to ~30 min on a cold kernel cache)..."
    local t0=$SECONDS
    while (( SECONDS - t0 < LOAD_TIMEOUT )); do
        kill -0 "$sc" 2>/dev/null || break   # systemctl finished (ok or failed)
        curl -sf -m 5 "http://127.0.0.1:${GLM_PORT}/v1/models" >/dev/null 2>&1 && break
        sleep 10
    done
    wait "$sc"; local rc=$?
    kill "$jl" 2>/dev/null || true
    (( rc == 0 )) || { journalctl --user -u "vsh-glm.service" --no-pager -n 30 2>/dev/null; die "Cluster bring-up failed (unit exit $rc)."; }

    if curl -sf "http://127.0.0.1:${GLM_PORT}/v1/models" >/dev/null 2>&1; then
        local rdma
        rdma=$(journalctl --user -u "vsh-glm-manual.service" --no-pager -o cat --since "-60min" 2>/dev/null \
               | grep -aoE "tbv_ar2: rank[0-9] ready \(qpn=[0-9]+ peer_qpn=[0-9]+\)" | head -2 | paste -sd' ' -)
        log "==================================================================="
        log " GLM-5.3-Flash (AWQ W4A16 + MTP) TP=2 is ready!"
        log "   API         : http://127.0.0.1:${GLM_PORT}  (OpenAI-compatible)"
        log "   Model name  : glm-5.3-flash"
        log "   Parallelism : TP=2 over Ray (head here, worker ${WORKER_HOST})"
        log "   Transport   : TB4 RoCE-RDMA ${RDMA_DEV}"
        log "   RDMA ranks  : ${rdma:-NOT READY - decode all-reduce NOT on RDMA}"
        log "   Logs        : journalctl --user -u vsh-glm-manual -f"
        log " Stop with    : $0 stop"
        log "==================================================================="
    else
        journalctl --user -u "vsh-glm.service" --no-pager -n 30 2>/dev/null
        die "Unit reports success but the API is not answering on :${GLM_PORT}."
    fi
}

do_stop_glm53() {
    log "Stopping the GLM-5.3-Flash cluster (serve -> ray -> both boxes)..."
    systemctl --user stop "vsh-glm.service" 2>/dev/null
    sleep 3
    local res
    res=$(ps -eo cmd --no-headers | grep "bin/[v]llm serve" | grep -cF "$GLM_MODEL_DIR" || true)
    if [[ "$res" != "0" ]]; then
        warn "$res vllm serve process(es) survived - reaping..."
        pkill -f "bin/vllm serve .*GLM-5.3-Flash" 2>/dev/null || true
        sleep 3
        pkill -9 -f "bin/vllm serve .*GLM-5.3-Flash" 2>/dev/null || true
    fi
    remote "pkill -f 'bin/vllm serve .*GLM-5.3-Flash' 2>/dev/null" 20 2>/dev/null || true
    log "GLM-5.3-Flash cluster stopped."
}

# ---- ds4 delegation: the existing AlexKGwyn/ds4-vllm stack -----------------
do_start_ds4() {
    [[ -f "$HOME/.config/systemd/user/${DS4_UNIT}.service" ]] \
        || die "ds4-vllm systemd unit not installed (${DS4_UNIT}). Clone AlexKGwyn/ds4-vllm and deploy its host kit first."
    log "Starting DeepSeek-V4-Flash cluster via the ds4-vllm stack (unit $DS4_UNIT)..."
    systemctl --user restart "$DS4_UNIT.service" &
    local sc=$!
    local t0=$SECONDS
    while (( SECONDS - t0 < LOAD_TIMEOUT )); do
        kill -0 "$sc" 2>/dev/null || break
        curl -sf -m 5 "http://127.0.0.1:${DS4_PORT}/v1/models" >/dev/null 2>&1 && break
        sleep 10
    done
    wait "$sc"; local rc=$?
    (( rc == 0 )) || { journalctl --user -u "$DS4_UNIT.service" --no-pager -n 30 2>/dev/null; die "ds4 bring-up failed (unit exit $rc)."; }
    curl -sf -m 5 "http://127.0.0.1:${DS4_PORT}/v1/models" >/dev/null 2>&1 \
        && log "DeepSeek-V4-Flash ready on http://127.0.0.1:${DS4_PORT} (unit $DS4_UNIT)." \
        || die "ds4 unit reports success but the API is not answering on :${DS4_PORT}."
}

do_stop_ds4() {
    log "Stopping DeepSeek-V4-Flash cluster (unit $DS4_UNIT)..."
    systemctl --user stop "$DS4_UNIT.service" 2>/dev/null
    sleep 3
    pkill -f "bin/vllm serve .*DeepSeek-V4-Flash" 2>/dev/null || true
    remote "pkill -f 'bin/vllm serve' 2>/dev/null" 20 2>/dev/null || true
    log "DeepSeek-V4-Flash cluster stopped."
}

# =============================================================================
do_status() {
    # link + RDMA rail
    if ip -brief addr show thunderbolt0 2>/dev/null | grep -q UP; then
        log "thunderbolt0 UP: $(ip -brief addr show thunderbolt0 | awk '{print $3}')"
    else
        warn "thunderbolt0 DOWN."
    fi
    if rdma link show "${RDMA_DEV}/1" 2>/dev/null | grep -q "state ACTIVE"; then
        log "RDMA rail: $RDMA_DEV ACTIVE"
    else
        warn "RDMA rail $RDMA_DEV not active."
    fi
    gid_nonzero && log "RoCE GID idx1: $(cat /sys/class/infiniband/${RDMA_DEV}/ports/1/gids/1)" \
                || warn "RoCE GID idx1 empty (no IP on rail)."

    # glm53
    local st
    st=$(systemctl --user is-active "vsh-glm.service" 2>/dev/null || echo unknown)
    [[ "$st" = "active" ]] && log "unit vsh-glm: $st" || warn "unit vsh-glm: $st"
    if curl -sf -m 5 "http://127.0.0.1:${GLM_PORT}/v1/models" 2>/dev/null | grep -q glm; then
        log "GLM API healthy: http://127.0.0.1:${GLM_PORT}"
    else
        warn "GLM API not answering on :${GLM_PORT}."
    fi

    # ds4
    st=$(systemctl --user is-active "${DS4_UNIT}.service" 2>/dev/null || echo unknown)
    [[ "$st" = "active" ]] && log "unit $DS4_UNIT: $st" || warn "unit $DS4_UNIT: $st"
    if curl -sf -m 5 "http://127.0.0.1:${DS4_PORT}/v1/models" 2>/dev/null | grep -q deepseek; then
        log "DS4 API healthy: http://127.0.0.1:${DS4_PORT}"
    else
        warn "DS4 API not answering on :${DS4_PORT}."
    fi

    # ray + processes + memory
    local gpus
    gpus=$(timeout 30 podman exec "$GLM_CTR" ray status 2>/dev/null | grep -oE "[0-9.]+/[0-9.]+ GPU" | head -1)
    [[ -n "$gpus" ]] && log "Ray GPUs (glm ctr): $gpus" || warn "Ray not reachable in container '$GLM_CTR'."
    log "vllm serve procs on head: $(ps -eo cmd --no-headers | grep -cE 'bin/[v]llm serve' || true)"
    log "MemAvailable: $(awk '/MemAvailable/{printf "%d", $2/1024}' /proc/meminfo)MB | worker: $(remote 'awk "/MemAvailable/{printf \"%d\", \$2/1024}" /proc/meminfo' 15 2>/dev/null || echo '?')MB"
}

# =============================================================================
parse_args "$@"
load_site

case "$MODEL:$ACTION" in
    glm53:start)  do_start_glm53 ;;
    glm53:stop)   do_stop_glm53 ;;
    glm53:logs)   exec journalctl --user -f -n 50 -q -u "vsh-glm.service" -u "vsh-glm-manual.service" ;;
    ds4:start)    do_start_ds4 ;;
    ds4:stop)     do_stop_ds4 ;;
    ds4:logs)     exec journalctl --user -f -n 50 -q -u "${DS4_UNIT}.service" -u "ds4-vllm-manual.service" ;;
    *:status)     do_status ;;
    *:help)       sed -n '2,32p' "$0" ;;
    *)            die "Unknown action for $MODEL." ;;
esac
