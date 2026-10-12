# Build context is the REPOSITORY ROOT (see docker-compose.yml: build.context: .),
# so this image can install the reviewscope_ml package alongside the backend.
#
# Base: bookworm by default. ltgpu1 overrides it to bullseye
# (docker-compose.ltgpu.yml): its Docker 20.10.7 seccomp profile blocks clone3,
# which bookworm's glibc (>=2.34) uses, while bullseye's glibc 2.31 does not.
# Bullseye is EOL and its security mirror now 404s, so the bullseye build must
# not touch apt — which works because amd64 needs no compiler (see below).
ARG BASE_IMAGE=python:3.12-slim-bookworm
FROM ${BASE_IMAGE}

WORKDIR /app

# Every dependency ships amd64 wheels, but hdbscan publishes no aarch64 wheel and
# compiles from source there — so the compiler is installed on arm64 only.
ARG TARGETARCH
RUN if [ "$TARGETARCH" = "arm64" ]; then \
        apt-get update \
        && apt-get install -y --no-install-recommends build-essential \
        && rm -rf /var/lib/apt/lists/*; \
    fi

# Pin torch to 2.7.1+cu126 BEFORE installing reviewscope_ml. Left unpinned,
# sentence-transformers pulls the latest torch, which has dropped Pascal (sm_61)
# kernels — this box's GPUs are TITAN X (Pascal), so ops fail with "no kernel
# image is available for execution on the device". torch 2.7.1 ships sm_60 cubins
# that run on sm_61 (verified against the host .venv). Installing it first pins
# the resolver so the reviewscope_ml install below keeps this build.
# arm64 (Apple Silicon, Graviton) gets the CPU build: PyTorch publishes no CUDA
# wheel of 2.7.1 for aarch64, and Docker on a Mac has no GPU anyway. TARGETARCH is
# set by BuildKit/buildx only; the legacy builder (ltgpu1) leaves it empty → CUDA.
RUN if [ "$TARGETARCH" = "arm64" ]; then TORCH_INDEX=cpu; else TORCH_INDEX=cu126; fi \
    && pip install --no-cache-dir torch==2.7.1 --index-url "https://download.pytorch.org/whl/$TORCH_INDEX"

# Install the reviewscope_ml package first (its heavy deps — sentence-transformers,
# torch, umap, hdbscan — are the slow, cache-worthy layer). Only the package
# sources are copied so unrelated changes don't bust this layer.
COPY pyproject.toml /opt/reviewscope_ml/pyproject.toml
COPY src/reviewscope_ml /opt/reviewscope_ml/src/reviewscope_ml
RUN pip install --no-cache-dir /opt/reviewscope_ml

# Backend dependencies.
COPY src/backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Backend application code. The compose dev bind-mount (./src/backend:/app)
# overlays this for hot-reload; the installed reviewscope_ml in site-packages
# is not shadowed by that mount.
COPY src/backend/ /app/

ENV PYTHONUNBUFFERED=1
