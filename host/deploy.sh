#!/usr/bin/env bash
# One-time deployment of the vllm-strix-halo host orchestration layer.
# Run on box1 (10.0.2.1, the ray head). Drives box2 (10.0.2.2) over ssh.
set -euo pipefail

R="$HOME/vllm-strix-halo"
WORKER="${1:-10.0.2.2}"

echo "== box1: deploy host scripts + site config =="
install -m 0755 "$R/vllm-strix-halo.sh"                    "$HOME/vllm-strix-halo.sh"
install -m 0755 "$R/scripts/download-models.sh"            "$HOME/download-models-vllm.sh"
install -m 0755 "$R/host/vsh-config"                       "$HOME/vsh-config"
install -m 0755 "$R/host/vsh-cluster-restart.sh"           "$HOME/vsh-cluster-restart.sh"
install -m 0755 "$R/host/vsh-cluster-down.sh"              "$HOME/vsh-cluster-down.sh"
install -m 0755 "$R/host/vsh-manual-serve.sh"              "$HOME/vsh-manual-serve.sh"
install -m 0755 "$R/host/container-heal.sh"                "$HOME/container-heal.sh"
install -m 0755 "$R/host/vsh-cluster-env.sh"               "$HOME/vsh-cluster-env.sh"
install -m 0755 "$R/host/vsh-cluster-env.rdma.sh"          "$HOME/vsh-cluster-env.rdma.sh"
install -m 0755 "$R/host/vsh-cluster-env.tcp.sh"           "$HOME/vsh-cluster-env.tcp.sh"
install -m 0644 "$R/host/vsh-warmup.py"                    "$HOME/vsh-warmup.py"
install -m 0644 "$R/host/systemd/vsh-glm.service"          "$HOME/.config/systemd/user/vsh-glm.service"

# Site configuration (edit BEFORE running: IPs, container, HCA, model dir)
[ -f "$HOME/vsh-config.yaml" ] || {
    echo "!! ~/vsh-config.yaml missing — copy host/vsh-config.yaml and edit it" >&2
    exit 1
}

systemctl --user daemon-reload

echo "== box2: deploy env files + container-heal (byte-identical) =="
for f in vsh-cluster-env.sh vsh-cluster-env.rdma.sh vsh-cluster-env.tcp.sh container-heal.sh; do
    scp -q "$HOME/$f" "$WORKER:$HOME/$f"
done
ssh -o BatchMode=yes "$WORKER" 'chmod 0755 ~/container-heal.sh ~/vsh-cluster-env*.sh'

echo "== verify env files are byte-identical across boxes =="
for f in vsh-cluster-env.sh vsh-cluster-env.rdma.sh vsh-cluster-env.tcp.sh; do
    h1=$(md5sum "$HOME/$f" | cut -d" " -f1)
    h2=$(ssh -o BatchMode=yes "$WORKER" "md5sum \$HOME/$f" | cut -d" " -f1)
    [ "$h1" = "$h2" ] && echo "   OK  $f" || { echo "   MISMATCH $f"; exit 1; }
done

echo "== done =="
