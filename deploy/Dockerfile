# syntax=docker/dockerfile:1.7
# ArchRender image: CUDA 13.0 runtime (Ampere → Blackwell), Blender 5.2 LTS, app venv, built UI.
# Release build (CI):  docker build -f deploy/Dockerfile .
# Local CPU build:     docker build -f deploy/Dockerfile --build-arg BLENDER_SOURCE=wheel \
#                        --build-arg APP_EXTRAS= --build-arg BASE_IMAGE=ubuntu:24.04@sha256:… .
ARG BASE_IMAGE=nvidia/cuda:13.0.3-runtime-ubuntu24.04@sha256:76f46f3e649824ebd3685d3d86c42b0ea62af04f59242f419f56cd5804a72efb
ARG NODE_IMAGE=node:22-bookworm-slim@sha256:43ac6c60b8f89723f746e8a92ce91abd5017e627ce1ddfe4238355d3a30b772c
ARG BUILDER_IMAGE=ubuntu:24.04@sha256:008173c23f95b170204355c12626cb5a965d779a7e1283b09e9cffbb1bf33ca3

# ---------------------------------------------------------------------------------- UI build
FROM ${NODE_IMAGE} AS ui
WORKDIR /ui
COPY deploy/with-ca.sh /usr/local/bin/with-ca
COPY ui/package.json ui/package-lock.json ./
RUN --mount=type=secret,id=extra_ca,required=false with-ca npm ci --no-audit --no-fund
COPY ui/ ./
RUN npm run build

# ---------------------------------------------------------------------------------- LibreDWG
# GPL-3.0-or-later, used only as the separate dwg2dxf process (ADR-S16); not in Ubuntu 24.04 apt.
FROM ${BUILDER_IMAGE} AS libredwg
RUN apt-get update && apt-get install -y --no-install-recommends build-essential ca-certificates curl pkg-config xz-utils \
    && rm -rf /var/lib/apt/lists/*
COPY deploy/with-ca.sh /usr/local/bin/with-ca
COPY deploy/libredwg/build.sh /usr/local/bin/build-libredwg
RUN --mount=type=secret,id=extra_ca,required=false with-ca build-libredwg /opt/libredwg

# ---------------------------------------------------------------------------------- runtime
FROM ${BASE_IMAGE} AS runtime
ARG BLENDER_SOURCE=tarball
ARG BLENDER_VERSION=5.2.2
# From a third-party mirror (ADR-S15); the release workflow compares it with the official
# download.blender.org .sha256 before building, and the build fails on any mismatch.
ARG BLENDER_SHA256=84098912789dc450e95697c4184fb8a90acbe5111c2ba4aede3fecb57806a168
ARG UV_VERSION=0.8.17
ARG UV_SHA256=920cbcaad514cc185634f6f0dcd71df5e8f4ee4456d440a22e0f8c0f142a8203
ARG LITESTREAM_VERSION=0.5.17
ARG LITESTREAM_SHA256=cfb371176d164437ae869f8351cfde49bd1804ae71c61923f75c9cba9c9c006d
ARG PYTHON_VERSION=3.12.11
ARG APP_EXTRAS=gpu
ARG INSTALL_VLLM=0
ARG BAKE_MODELS=
ARG GIT_COMMIT=unknown

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=all \
    UV_PYTHON_INSTALL_DIR=/opt/python UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/opt/venv UV_COMPILE_BYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl xz-utils tini supervisor \
        libx11-6 libxi6 libxxf86vm1 libxfixes3 libxrender1 libxext6 libxkbcommon0 libsm6 libice6 \
        libgl1 libegl1 libglvnd0 libfontconfig1 libgomp1 \
        libheif-examples libheif-plugin-libde265 \
        tesseract-ocr tesseract-ocr-tur tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*
COPY deploy/with-ca.sh /usr/local/bin/with-ca

# uv and Litestream (release tarballs, SHA-256 verified)
RUN --mount=type=secret,id=extra_ca,required=false with-ca sh -c 'set -eux; cd /tmp; \
    curl -fsSLo uv.tgz "https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-x86_64-unknown-linux-gnu.tar.gz"; \
    echo "${UV_SHA256}  uv.tgz" | sha256sum -c -; tar -xzf uv.tgz; install -m 0755 uv-x86_64-unknown-linux-gnu/uv /usr/local/bin/uv; \
    curl -fsSLo ls.tgz "https://github.com/benbjohnson/litestream/releases/download/v${LITESTREAM_VERSION}/litestream-${LITESTREAM_VERSION}-linux-x86_64.tar.gz"; \
    echo "${LITESTREAM_SHA256}  ls.tgz" | sha256sum -c -; tar -xzf ls.tgz litestream; install -m 0755 litestream /usr/local/bin/litestream; \
    rm -rf /tmp/uv.tgz /tmp/ls.tgz /tmp/uv-x86_64-unknown-linux-gnu /tmp/litestream'

WORKDIR /opt/archrender
COPY deploy/blender/requirements.lock deploy/blender/requirements.lock

# Blender 5.2 LTS: official tarball (release) or the official bpy wheel of the same version.
RUN --mount=type=secret,id=extra_ca,required=false with-ca sh -c 'set -eux; \
    if [ "$BLENDER_SOURCE" = "tarball" ]; then \
      curl -fsSLo /tmp/blender.tar.xz "https://download.blender.org/release/Blender${BLENDER_VERSION%.*}/blender-${BLENDER_VERSION}-linux-x64.tar.xz"; \
      echo "${BLENDER_SHA256}  /tmp/blender.tar.xz" | sha256sum -c -; \
      tar -xJf /tmp/blender.tar.xz -C /opt; mv "/opt/blender-${BLENDER_VERSION}-linux-x64" /opt/blender; rm /tmp/blender.tar.xz; \
      /opt/blender/blender -b --factory-startup --version | head -n1; \
    else \
      uv venv -p 3.13 /opt/venv-blender; \
      uv pip install -p /opt/venv-blender/bin/python --require-hashes -r deploy/blender/requirements.lock; \
      /opt/venv-blender/bin/python -c "import bpy; print(bpy.app.version_string)"; \
    fi'
RUN if [ "$BLENDER_SOURCE" = "wheel" ]; then echo module > /etc/archrender-blender-mode; else echo binary > /etc/archrender-blender-mode; fi

# App environment from the lockfile (dependencies first for layer caching)
RUN --mount=type=secret,id=extra_ca,required=false with-ca uv python install ${PYTHON_VERSION}
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=secret,id=extra_ca,required=false with-ca sh -c 'set -eux; extras=""; \
    if [ -n "$APP_EXTRAS" ]; then extras="--extra $APP_EXTRAS"; fi; \
    uv sync --frozen --no-dev --no-install-project --python ${PYTHON_VERSION} $extras'
COPY src/ src/
COPY configs/ configs/
COPY deploy/ deploy/
COPY scripts/ scripts/
COPY ui/package-lock.json ui/package-lock.json
RUN --mount=type=secret,id=extra_ca,required=false with-ca sh -c 'set -eux; extras=""; \
    if [ -n "$APP_EXTRAS" ]; then extras="--extra $APP_EXTRAS"; fi; \
    uv sync --frozen --no-dev --no-editable --python ${PYTHON_VERSION} $extras; \
    /opt/venv/bin/python -c "import archrender.api.app, archrender.pipeline.worker, OpenEXR, manifold3d, cv2"'

# vLLM in its own environment (ADR-S02); enabled once a vLLM-served role is implemented.
RUN --mount=type=secret,id=extra_ca,required=false with-ca sh -c 'set -eux; if [ "$INSTALL_VLLM" = "1" ]; then \
      uv venv -p ${PYTHON_VERSION} /opt/venv-vllm; \
      uv pip install -p /opt/venv-vllm/bin/python --require-hashes -r deploy/vllm/requirements.lock; fi'

COPY --from=ui /ui/dist /opt/archrender/ui/dist

# Optional: bake model weights for faster cold starts (needs --secret id=hf_token).
RUN --mount=type=secret,id=hf_token,required=false set -eux; if [ -n "$BAKE_MODELS" ]; then \
      HF_TOKEN="$(cat /run/secrets/hf_token 2>/dev/null || true)" ARCHRENDER_DATA_DIR=/opt/baked \
      /opt/venv/bin/python -m archrender.ops.models download --profile "$BAKE_MODELS"; fi

COPY --from=libredwg /opt/libredwg /opt/libredwg

ENV PATH=/opt/venv/bin:/opt/libredwg/bin:$PATH \
    ARCHRENDER_CONFIGS_DIR=/opt/archrender/configs \
    ARCHRENDER_UI_DIST=/opt/archrender/ui/dist \
    ARCHRENDER_DATA_DIR=/workspace \
    ARCHRENDER_DB_PATH=/var/lib/archrender/db/archrender.sqlite \
    ARCHRENDER_BLENDER_BIN=/opt/blender/blender \
    ARCHRENDER_BLENDER_PYTHON=/opt/venv-blender/bin/python \
    ARCHRENDER_GIT_COMMIT=${GIT_COMMIT} \
    HF_HOME=/workspace/cache/hf

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=600s CMD curl -fsS http://127.0.0.1:8000/healthz || exit 1
ENTRYPOINT ["/usr/bin/tini", "--", "/opt/archrender/deploy/runpod/entrypoint.sh"]
