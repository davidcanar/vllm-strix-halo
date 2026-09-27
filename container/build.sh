#!/usr/bin/env bash
# Build the vllm-strix-halo serving image (vLLM main pin + GLM-5.3-Flash on
# gfx1151, ROCm 10, with the OdinLink userspace: odl net plugin, odl_ar2
# decode all-reduce, odl_mq control plane).
#
# Usage:
#   container/build.sh                # build with the default vLLM pin
#   VLLM_COMMIT=<sha> container/build.sh
set -euo pipefail

cd "$(dirname "$0")/.."          # repo root = build context

IMAGE=${VSH_IMAGE:-vllm-strix-halo:local}
VLLM_COMMIT=${VLLM_COMMIT:-73859fec5865700c5b2021b22890cb3b2005f66e}

log() { printf '\033[1;34m[build]\033[0m %s\n' "$*"; }

log "building $IMAGE (vLLM @ $VLLM_COMMIT, gfx1151 / ROCm 10 base)"
podman build -t "$IMAGE" \
  --build-arg VLLM_COMMIT="$VLLM_COMMIT" \
  -f container/Dockerfile .

log "post-build checks"
# GLM-5.3-Flash registration (import must not need a GPU).
podman run --rm "$IMAGE" python - <<'EOF' || { echo "FAIL: glm5next not registered"; exit 1; }
from vllm.model_executor.models.registry import ModelRegistry
archs = ModelRegistry.get_supported_archs()
for a in ("Glm5NextForCausalLM", "Glm5NextForConditionalGeneration"):
    assert a in archs, a
print("OK: glm5next registered:", [a for a in archs if a.startswith("Glm5Next")])
EOF

# OdinLink userspace presence (decode all-reduce + RCCL net plugin + stream
# lib). The host driver (odl_tb5.ko) is built separately by
# odinlink/build-odinlink.sh; the libs here must match its ABI pin.
podman run --rm "$IMAGE" python - <<'EOF' || { echo "FAIL: odinlink userspace missing"; exit 1; }
import os
paths = [
    "/opt/venv/lib/python3.12/site-packages/libodl_ar2.so",
    "/usr/local/lib/odinlink/librccl_net_odl_tb5.so",
    "/usr/local/lib/odinlink/libodl_tb5.so.0",
]
for p in paths:
    assert os.path.exists(p), p
import sys
sys.path.insert(0, "/opt/venv/lib/python3.12/site-packages")
import odl_ar2, odl_mq   # wrappers importable (device not needed at import)
print("OK: odinlink userspace present (libodl_ar2, rccl net plugin, libodl_tb5)")
EOF
log "image ready: $IMAGE"
