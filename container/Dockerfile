# vllm-strix-halo — vLLM (main, >= GLM-5.3-Flash merge) for AMD Strix Halo
# (gfx1151), TP=2, with the Thunderbolt usb4_rdma provider and the tbv_ar2
# RDMA all-reduce native for the decode collective.
#
# Thin layer over kyuz0's TheRock gfx1151 image (ROCm 10.0 + torch 2.11 +
# triton 3.8 + aiter — the proven gfx1151 toolchain). The stock vLLM 0.28 in
# the base predates the GLM-5.3-Flash merge (vllm-project/vllm#53906,
# 2026-09-03), so this file rebuilds vLLM from source at a pinned post-merge
# commit while keeping the base's torch/triton/aiter untouched.
#
# Why not more patches? See PATCHES.md at the repo root — the review of
# AlexKGwyn/ds4-vllm that this repo builds on.
#
# Build (context = REPO ROOT):
#   container/build.sh                     # -> vllm-strix-halo:local
#   podman build -t vllm-strix-halo:local -f container/Dockerfile .

ARG VSH_BASE="docker.io/kyuz0/vllm-therock-gfx1151:rocm10.0.0-torch2.11.0-vllm0.28.0"
# vLLM source pin. Must be >= 98ed0856 (GLM-5.3-Flash support).
ARG VLLM_REPO="https://github.com/vllm-project/vllm"
ARG VLLM_COMMIT="8bf3963"
# rdma-core branch the usb4_rdma provider is built against. The base ships
# libibverbs 1.16.62.0 (provider ABI rdmav59, copied in by the image author),
# so the provider must be the rdmav59 flavour or libibverbs never loads it.
ARG RDMA_CORE_TAG="v59.0"
# hellas-ai/thunderbolt-ibverbs pin (same pin as tbv/build-modules.sh).
ARG TBV_IBVERBS_COMMIT="76ba39b630a70accb72f19388eefe48844b50eb8"

# ---------- stage 1: build the usb4_rdma libibverbs provider ----------
# libibverbs matches a provider to a device BY NAME. Without these files
# `ibv_devices` is EMPTY inside the container: tbv_ar2 init fails and
# RCCL (NCCL_IB) cannot see the rail, so the TP all-reduce silently falls
# back to TCP over the Thunderbolt IP link.
FROM ${VSH_BASE} AS provider-build
ARG RDMA_CORE_TAG
ARG TBV_IBVERBS_COMMIT
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
        git cmake gcc libnl-3-dev libnl-route-3-dev pkg-config && \
    rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 https://github.com/hellas-ai/thunderbolt-ibverbs /tmp/tbv-ibverbs && \
    git -C /tmp/tbv-ibverbs checkout ${TBV_IBVERBS_COMMIT}
RUN set -eux; \
    git clone --depth 1 -b ${RDMA_CORE_TAG} https://github.com/linux-rdma/rdma-core /tmp/rdma-core && \
    cd /tmp/rdma-core && \
    for p in /tmp/tbv-ibverbs/packaging/rdma-core-patches/0*.patch; do \
        git apply -C1 "$p"; \
    done && \
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DNO_PYVERBS=1 -DNO_MAN_PAGES=1 && \
    cmake --build build --parallel 8 && \
    ls build/lib/libusb4_rdma-rdmav*.so

# ---------- final stage: rebuild vLLM + RDMA pieces on the base ----------
FROM ${VSH_BASE}

# Build prerequisites for the vLLM source build (--no-build-isolation keeps
# the base's torch 2.11 / triton 3.8 / aiter instead of pulling vLLM's
# CUDA-build torch 2.13 pin).
RUN python -m pip install -U --no-cache-dir \
        "setuptools>=77,<81" setuptools-scm setuptools-rust wheel jinja2 packaging

# Stock vLLM 0.28 in the base predates the GLM merge; drop it cleanly.
RUN python -m pip uninstall -y vllm || true

# Build vLLM at the pinned post-merge commit for gfx1151 only.
# - PYTORCH_ROCM_ARCH pins the HIP target so the build never needs a GPU probe
#   (podman build has no /dev/kfd). MAX_JOBS=8 keeps the multi-GB clang
#   processes within budget on a 128 GB box that may be co-serving another model.
# - CMAKE_ARGS points CMake at the pip-installed ROCm SDK's cmake package dir
#   (the TheRock venv layout has no /opt/rocm, so torch's LoadHIP.cmake cannot
#   find FindHIP.cmake without this and vLLM fails with "Can't find CUDA or
#   HIP installation").
# - SETUPTOOLS_SCM_PRETEND_VERSION: the shallow single-commit fetch has no
#   tags, so setuptools-scm needs an explicit version string.
# - Shallow fetch of the exact SHA (GitHub allows it) instead of a full-history
#   clone: same result, minutes instead of ~an hour on slow links.
ARG VLLM_REPO
ARG VLLM_COMMIT
RUN git init -q /tmp/vllm-src && \
    git -C /tmp/vllm-src remote add origin ${VLLM_REPO} && \
    git -C /tmp/vllm-src fetch -q --depth 1 origin ${VLLM_COMMIT} && \
    git -C /tmp/vllm-src checkout -q FETCH_HEAD && \
    cd /tmp/vllm-src && \
    PYTORCH_ROCM_ARCH=gfx1151 VLLM_USE_PRECOMPILED=0 MAX_JOBS=8 \
    SETUPTOOLS_SCM_PRETEND_VERSION="0.29.0.dev0+git.${VLLM_COMMIT}" \
    CMAKE_ARGS="-DCMAKE_PREFIX_PATH=${ROCM_DEVEL_PATH} -DCMAKE_MODULE_PATH=${ROCM_DEVEL_PATH}/lib/cmake/hip" \
    python -m pip install --no-build-isolation --no-deps . && \
    rm -rf /tmp/vllm-src

# Runtime deps for the new vLLM, from its own pinned requirements, minus the
# torch/triton/aiter/amdsmi lines (the base's gfx1151 builds stay in place and
# the base's amdsmi is newer than vLLM's pin). amd-quark / datasets /
# pytest-asyncio / runai-model-streamer are intentionally skipped (unused here).
RUN python -m pip install --no-cache-dir \
        "grpcio==1.78.0" "grpcio-reflection==1.78.0" \
        "numba==0.65.0" peft "tensorizer==2.10.1" \
        "conch-triton-kernels==1.2.1" "timm>=1.0.17" \
        "tilelang==0.1.10" "apache-tvm-ffi==0.1.10" \
        "fastsafetensors>=0.3.3" "transformers>=5.10.4"

# usb4_rdma userspace provider (rdmav59 ABI, matching the base's libibverbs).
COPY --from=provider-build --chown=root:root \
  /tmp/rdma-core/build/lib/libusb4_rdma-rdmav*.so /usr/lib/x86_64-linux-gnu/libibverbs/
RUN printf 'driver usb4_rdma\n' > /etc/libibverbs.d/usb4_rdma.driver

# tbv_ar2 python wrappers (ctypes-load the natives built below). rootfs/
# mirrors "/" — the files live under opt/venv/lib/python3.12/site-packages/.
COPY --chown=root:root container/rootfs/ /

# Build the TB4-RDMA all-reduce natives (hipcc lives in the venv in this
# image; verbs.h ships with the base's rdma-core dev files).
COPY --chown=root:root container/native/ /tmp/vsh-native/
RUN /opt/venv/bin/hipcc -O2 --offload-arch=gfx1151 -shared -fPIC \
      -o /opt/venv/lib/python3.12/site-packages/libtbv_ar2.so \
      /tmp/vsh-native/tbv_ar2.hip -libverbs -lpthread && \
    gcc -O2 -shared -fPIC \
      -o /opt/venv/lib/python3.12/site-packages/libtbv_ar.so \
      /tmp/vsh-native/tbv_ar.c -libverbs && \
    rm -rf /tmp/vsh-native

# RDMA all-reduce hook (VSH_TBV_AR2) in the CUDA communicator. Model-agnostic,
# env-gated, fail-open. Reviewed in PATCHES.md.
COPY container/patches/vsh-rdma-allreduce.patch /tmp/
RUN cd /opt/venv/lib/python3.12/site-packages && \
    git apply -p1 --whitespace=nowarn /tmp/vsh-rdma-allreduce.patch && \
    rm /tmp/vsh-rdma-allreduce.patch

# Drop stale bytecode for the touched package so the first import recompiles
# from the patched sources.
RUN find /opt/venv/lib/python3.12/site-packages/vllm \
         -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true

# com.github.containers.toolbox is what `toolbox create` checks for.
LABEL com.github.containers.toolbox="true" \
      org.opencontainers.image.title="vllm-strix-halo" \
      org.opencontainers.image.description="vLLM (GLM-5.3-Flash + DeepSeek-V4-Flash era) for AMD Strix Halo gfx1151, TP=2 over Thunderbolt RoCE-RDMA" \
      vsh.base="docker.io/kyuz0/vllm-therock-gfx1151:rocm10.0.0-torch2.11.0-vllm0.28.0" \
      vsh.vllm_commit="${VLLM_COMMIT}"
