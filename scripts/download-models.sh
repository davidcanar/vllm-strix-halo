#!/usr/bin/env bash
# download-models.sh — fetch the vllm-strix-halo model weights on BOTH boxes.
#
#   glm53  GLM-5.3-Flash, AWQ W4A16 (~191 GB) — the only HF format that fits
#          2x 128 GB UMA (official FP8 is ~335 GB, BF16 ~643 GB — see PATCHES.md)
#   ds4    DeepSeek-V4-Flash-0731, official BF16 (~156 GB)
#
#   hf download is resumable: re-run after any interruption.
#
# Usage:
#   ./download-models.sh [glm53|ds4|all] [--both|--head|--worker]
set -uo pipefail

WORKER="10.0.2.2"
MODELS="$HOME/models"

GLM53_REPO="wtdcode/GLM-5.3-Flash-AWQ-W4A16"
GLM53_DIR="$MODELS/GLM-5.3-Flash-AWQ-W4A16"
GLM53_SIZE_GB=191

DS4_REPO="deepseek-ai/DeepSeek-V4-Flash-0731"
DS4_DIR="$MODELS/DeepSeek-V4-Flash-0731-hf"
DS4_SIZE_GB=156

TARGET="both"
MODEL_SEL="all"

log()  { printf '\033[1;34m[dl]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[dl]\033[0m %s\n' "$*" >&2; exit 1; }

for a in "$@"; do
    case "$a" in
        glm53|ds4|all) MODEL_SEL="$a" ;;
        --both)  TARGET="both" ;;
        --head)  TARGET="head" ;;
        --worker) TARGET="worker" ;;
        *) die "unknown arg '$a' (use: $0 [glm53|ds4|all] [--both|--head|--worker])" ;;
    esac
done

command -v hf >/dev/null 2>&1 || die "hf CLI not found — pip install huggingface_hub (and \`hf auth login\` once)."

want_glm53() { [ "$MODEL_SEL" != ds4 ]; }
want_ds4()   { [ "$MODEL_SEL" != glm53 ]; }

check_space() { # dir label
    local free_kb free_gb need_gb=0
    free_kb=$(df -P "$1" 2>/dev/null | awk 'NR==2{print $4}')
    free_gb=$(( free_kb / 1024 / 1024 ))
    want_glm53 && need_gb=$(( need_gb + GLM53_SIZE_GB ))
    want_ds4   && need_gb=$(( need_gb + DS4_SIZE_GB ))
    if [ "$free_gb" -lt $(( need_gb + 20 )) ]; then
        die "$2: only ${free_gb} GB free on $1 — need ~$(( need_gb + 20 )) GB for this download set"
    fi
    log "$2: ${free_gb} GB free on $1 (need ~$(( need_gb + 20 )) GB)"
}

dl_here() {
    want_glm53 && { log "downloading $GLM53_REPO -> $GLM53_DIR"; hf download "$GLM53_REPO" --local-dir "$GLM53_DIR"; }
    want_ds4   && { log "downloading $DS4_REPO -> $DS4_DIR"; hf download "$DS4_REPO" --local-dir "$DS4_DIR"; }
}

dl_remote() {
    local host="$1"
    # worker: check its space, then download the same set (sequential, logged)
    ssh -o BatchMode=yes "$host" "bash -s" <<EOF
set -uo pipefail
free_kb=\$(df -P "$MODELS" 2>/dev/null | awk 'NR==2{print \$4}')
free_gb=\$(( free_kb / 1024 / 1024 ))
need_gb=0
$([ "$MODEL_SEL" != ds4 ] && echo "need_gb=\$(( need_gb + $GLM53_SIZE_GB ))")
$([ "$MODEL_SEL" != glm53 ] && echo "need_gb=\$(( need_gb + $DS4_SIZE_GB ))")
[ "\$free_gb" -lt \$(( need_gb + 20 )) ] && { echo "!! $host: only \${free_gb} GB free — need ~\$(( need_gb + 20 )) GB" >&2; exit 1; }
echo "[dl] $host: \${free_gb} GB free (need ~\$(( need_gb + 20 )) GB)"
mkdir -p "$MODELS"
$([ "$MODEL_SEL" != ds4 ] && echo "hf download $GLM53_REPO --local-dir $GLM53_DIR")
$([ "$MODEL_SEL" != glm53 ] && echo "hf download $DS4_REPO --local-dir $DS4_DIR")
echo "[dl] $host: done"
EOF
}

case "$TARGET" in
    head)   check_space "$MODELS" "head"; dl_here ;;
    worker) dl_remote "$WORKER" ;;
    both)
        check_space "$MODELS" "head"
        # worker first in the background (its download is the long pole), head here
        dl_remote "$WORKER" >"$HOME/vsh-dl-worker.log" 2>&1 &
        local wpid=$!
        dl_here
        wait "$wpid" || { tail -5 "$HOME/vsh-dl-worker.log"; die "worker download failed (see ~/vsh-dl-worker.log)"; }
        log "worker done (log: ~/vsh-dl-worker.log)"
        ;;
esac

log "all downloads complete."
