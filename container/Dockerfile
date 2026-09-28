# vllm-strix-halo — vLLM (main, with the late-Sep-2026 GLM-5.3-Flash
# correctness fixes) for AMD Strix Halo (gfx1151), TP=2, with the OdinLink
# Thunderbolt fabric: RCCL over the odl_tb5 net plugin for the big
# collectives, odl_ar2 for the small decode all-reduce, odl_mq for the
# EngineCore<->worker control plane.
#
# Thin layer over kyuz0's TheRock gfx1151 image (ROCm 10.0 + torch 2.11 +
# triton 3.8 + aiter — the proven gfx1151 toolchain). The stock vLLM 0.30 in
# the base (2026-09-23) predates the late-September GLM-5.3-Flash fixes —
# sparse-indexer topk backend selection (#58594), kpool corruption with
# speculative decoding (#58454), the SparseIndexerTopk dispatcher (#57546),
# ROCm kpool boot-crash alias (#57464-era), FlashKDA prefill (#57162) — so
# this file rebuilds vLLM from source at a pinned commit while keeping the
# base's torch/triton/aiter untouched.
#
# The tbv/thunderbolt-ibverbs RoCE stack this repo carried before is GONE:
# the fabric is the OdinLink driver (odl_tb5.ko on the hosts, built by
# ../odinlink/build-odinlink.sh; the userspace built here must match its
# stream ABI pin).
#
# Build (context = REPO ROOT):
#   container/build.sh                     # -> vllm-strix-halo:local
#   podman build -t vllm-strix-halo:local -f container/Dockerfile .

ARG VSH_BASE="docker.io/kyuz0/vllm-therock-gfx1151:rocm10.0.0-torch2.11.0-vllm0.30.0"
# vLLM source pin. Must include the GLM-5.3-Flash correctness wave of
# 2026-09-18..26 (kpool spec-decode corruption #58454, sparse-indexer topk
# backend selection #58594, cooperative top-k #57327, FlashKDA prefill).
ARG VLLM_REPO="https://github.com/vllm-project/vllm"
ARG VLLM_COMMIT="73859fec5865700c5b2021b22890cb3b2005f66e"
# OdinLink-Five pin — the SAME pin as odinlink/build-odinlink.sh (the
# HOST-side kernel driver build); the userspace built here must match its
# stream ABI. With the same odinlink-local.patch applied.
ARG ODL_COMMIT="4534f5852253503e61f8540a175d7e32cb072c3b"

# ---------- stage 1: build the OdinLink userspace ----------
# libodl_tb5 (the stream API over /dev/odl_tb5_*), the RCCL net plugin RCCL
# dlopens via NCCL_NET_PLUGIN (librccl_net_odl_tb5.so — plain C, RCCL stages
# GPU<->host itself and hands it host pointers), and the odl_ar2 decode
# all-reduce (libodl_ar2.so, HIP for gfx1151).
FROM ${VSH_BASE} AS odinlink-build
ARG ODL_COMMIT
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
        git cmake build-essential && \
    rm -rf /var/lib/apt/lists/*
COPY container/odinlink/odinlink-local.patch /tmp/
RUN git clone https://github.com/Geramy/OdinLink-Five /tmp/odinlink && \
    git -C /tmp/odinlink checkout ${ODL_COMMIT} && \
    git -C /tmp/odinlink apply /tmp/odinlink-local.patch && \
    cd /tmp/odinlink && \
    cmake -S . -B build \
      -DBUILD_KERNEL_MODULE=OFF -DBUILD_DAEMON=OFF -DBUILD_TRAY=OFF \
      -DBUILD_NCCL=OFF -DBUILD_VERBS=OFF && \
    cmake --build build --target odl_tb5 --parallel 8 && \
    ln -sf libodl_tb5.so.0.1.0 build/lib/libodl_tb5.so && \
    ls build/lib/libodl_tb5.so.0.1.0 && \
    cc -O3 -DNDEBUG -fPIC \
      -I third_party/rccl -I lib/include \
      -shared -Wl,-soname,librccl_net_odl_tb5.so \
      -Wl,-rpath,/usr/local/lib/odinlink \
      -o build/librccl_net_odl_tb5.so rccl/src/odl_tb5_plugin.c \
      build/lib/libodl_tb5.so.0.1.0
# odl_ar2 decode all-reduce native (gfx1151; hipcc lives in the venv in
# this image). rpath /usr/local/lib/odinlink resolves libodl_tb5.so.0 at
# runtime in the final stage.
COPY container/native/odl_ar2.hip /tmp/
RUN /opt/venv/bin/hipcc -O3 --offload-arch=gfx1151 -fPIC -shared \
      -I /tmp/odinlink/lib/include -Wl,-rpath,/usr/local/lib/odinlink \
      -o /tmp/libodl_ar2.so /tmp/odl_ar2.hip \
      -L /tmp/odinlink/build/lib -lodl_tb5

# ---------- final stage: rebuild vLLM + OdinLink userspace on the base ----------
FROM ${VSH_BASE}

# Build prerequisites for the vLLM source build (--no-build-isolation keeps
# the base's torch 2.11 / triton 3.8 / aiter instead of pulling vLLM's
# CUDA-build torch pin).
RUN python -m pip install -U --no-cache-dir \
        "setuptools>=77,<81" setuptools-scm setuptools-rust wheel jinja2 packaging

# Stock vLLM 0.30 in the base predates the GLM fix wave; drop it cleanly.
RUN python -m pip uninstall -y vllm || true

# Build vLLM at the pinned commit for gfx1151 only.
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
    ROCM_PATH=${ROCM_DEVEL_PATH} \
    SETUPTOOLS_SCM_PRETEND_VERSION="0.31.0.dev0+git.${VLLM_COMMIT}" \
    CMAKE_ARGS="-DCMAKE_PREFIX_PATH=${ROCM_DEVEL_PATH}/lib/cmake -DCMAKE_MODULE_PATH=${ROCM_DEVEL_PATH}/lib/cmake/hip -DCMAKE_HIP_COMPILER_ROCM_ROOT=${ROCM_DEVEL_PATH}" \
    python -m pip install --no-build-isolation --no-deps . && \
    cat requirements/rocm.txt requirements/common.txt | \
      grep -vE '^(#|-r|amd-quark|datasets|pytest-asyncio|runai-model-streamer|setuptools|setuptools-scm|setuptools-rust)' | \
      grep -vE '^[[:space:]]*$' > /tmp/vsh-req.txt && \
    python -m pip install --no-cache-dir -r /tmp/vsh-req.txt && \
    rm -rf /tmp/vllm-src /tmp/vsh-req.txt
# Runtime deps come from the pin's own requirements/rocm.txt + common.txt,
# minus the lines the base already provides or this stack does not use:
# amd-quark (GLM here is AWQ, not Quark MXFP4), datasets / pytest-asyncio /
# runai-model-streamer (unused), and the setuptools pins (the build prereqs
# above are newer and must not be downgraded).

# OdinLink userspace: libodl_tb5 + the RCCL net plugin under
# /usr/local/lib/odinlink (the odl env profile points NCCL_NET_PLUGIN at the
# plugin; both resolve libodl_tb5.so.0 via rpath).
COPY --from=odinlink-build --chown=root:root \
  /tmp/odinlink/build/lib/libodl_tb5.so.0.1.0 /usr/local/lib/odinlink/
COPY --from=odinlink-build --chown=root:root \
  /tmp/odinlink/build/librccl_net_odl_tb5.so /usr/local/lib/odinlink/
RUN cd /usr/local/lib/odinlink && \
    ln -sfn libodl_tb5.so.0.1.0 libodl_tb5.so.0 && \
    ln -sfn libodl_tb5.so.0 libodl_tb5.so

# Python wrappers: odl_ar2.py (ctypes-loads libodl_ar2.so next to itself)
# and odl_mq.py (ctypes-loads libodl_tb5.so.0 by soname/rpath), plus the
# aiter gfx1151 GEMM config overlay. rootfs/ mirrors "/" — files live under
# opt/venv/lib/python3.12/site-packages/.
COPY --chown=root:root container/rootfs/ /
COPY --from=odinlink-build --chown=root:root \
  /tmp/libodl_ar2.so /opt/venv/lib/python3.12/site-packages/

# Patch set (see PATCHES.md):
# - vsh-odl-ar2-allreduce / vsh-odl-mq-dataplane: the OdinLink hooks ported
#   from AlexKGwyn/ds4-vllm (decode all-reduce + control-plane data plane),
#   env-gated / fail-open.
# - vsh-moe-router-deterministic-topk: reproducible greedy expert selection
#   (upstream #55514 still open).
# - vsh-aiter-gfx1151-gate / vsh-fp8-fnuz-mqa / vsh-mhc-no-tilelang-gfx1151:
#   gfx1151 enablement fixes.
# (The old vsh-rdma-allreduce patch is superseded by the odl hook; the MTP
# rope-free patch was superseded upstream — _use_rocm_sparse_triton now
# routes spec-decode rows to the Triton path by itself.)
COPY container/patches/vsh-odl-ar2-allreduce.patch \
     container/patches/vsh-odl-mq-dataplane.patch \
     container/patches/vsh-mhc-no-tilelang-gfx1151.patch \
     container/patches/vsh-aiter-gfx1151-gate.patch \
     container/patches/vsh-fp8-fnuz-mqa.patch \
     container/patches/vsh-moe-router-deterministic-topk.patch \
     container/patches/vsh-wna16-layerwise-empty-cache.patch \
     container/patches/vsh-indexer-persistent-topk-rocm.patch \n     container/patches/vsh-glm-eagle3-aux-states.patch /tmp/
RUN cd /opt/venv/lib/python3.12/site-packages && \
    git apply -p1 --whitespace=nowarn \
      /tmp/vsh-odl-ar2-allreduce.patch /tmp/vsh-odl-mq-dataplane.patch \
      /tmp/vsh-mhc-no-tilelang-gfx1151.patch /tmp/vsh-aiter-gfx1151-gate.patch \
      /tmp/vsh-fp8-fnuz-mqa.patch /tmp/vsh-moe-router-deterministic-topk.patch \
      /tmp/vsh-wna16-layerwise-empty-cache.patch \
      /tmp/vsh-indexer-persistent-topk-rocm.patch \n      /tmp/vsh-glm-eagle3-aux-states.patch && \
    rm -f /tmp/vsh-*.patch

# Drop stale bytecode for the touched package so the first import recompiles
# from the patched sources.
RUN find /opt/venv/lib/python3.12/site-packages/vllm \
         -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true

# gfx1151 defaults, baked into the image so EVERY process (raylet, ray
# workers, driver) sees them from birth. vLLM snapshots VLLM_ROCM_USE_AITER
# at module import — propagating it only via the driver->worker RPC arrives
# too late (the worker has already imported vllm). The cluster env files
# (host/vsh-cluster-env.sh) re-export the same values, so they stay tunable.
# - aiter ON: the glm5next sparse-attention indexer's ROCm path requires it
# - aiter MoE OFF: aiter's MoE kernels do not support gfx1151
ENV VLLM_ROCM_USE_AITER=1 \
    VLLM_ROCM_USE_AITER_MOE=0

# com.github.containers.toolbox is what `toolbox create` checks for.
LABEL com.github.containers.toolbox="true" \
      org.opencontainers.image.title="vllm-strix-halo" \
      org.opencontainers.image.description="vLLM (GLM-5.3-Flash era, main pin with the Sep-2026 fix wave) for AMD Strix Halo gfx1151, TP=2 over the OdinLink Thunderbolt fabric" \
      vsh.base="docker.io/kyuz0/vllm-therock-gfx1151:rocm10.0.0-torch2.11.0-vllm0.30.0" \
      vsh.vllm_commit="${VLLM_COMMIT}"
