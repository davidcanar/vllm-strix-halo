#!/usr/bin/env bash
# Build the vllm-strix-halo serving image (vLLM main + GLM-5.3-Flash on
# gfx1151, ROCm 10, with the usb4_rdma provider and tbv_ar2 RDMA natives).
#
# Usage:
#   container/build.sh                # build with the default vLLM pin
#   VLLM_COMMIT=98ed0856 container/build.sh
set -euo pipefail

cd "$(dirname "$0")/.."          # repo root = build context

IMAGE=${VSH_IMAGE:-vllm-strix-halo:local}
VLLM_COMMIT=${VLLM_COMMIT:-8bf3963}

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

# tbv_ar2 native + provider presence (needs /dev/infiniband on the host).
podman run --rm --device /dev/kfd --device /dev/dri --device /dev/infiniband \
  "$IMAGE" python - <<'EOF' || { echo "WARN: tbv_ar2 lib missing (serve still works on TCP fallback)"; exit 0; }
import ctypes, os
p = "/opt/venv/lib/python3.12/site-packages/libtbv_ar2.so"
assert os.path.exists(p), p
print("OK: libtbv_ar2.so present")
EOF
log "image ready: $IMAGE"
