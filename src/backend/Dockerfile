# Build context is the REPOSITORY ROOT (see docker-compose.yml: build.context: .),
# so this image can install the reviewscope_ml package alongside the backend.
# Pinned to bullseye (not the bare -slim tag, which tracks bookworm/trixie).
# This host's Docker daemon is 20.10.7, whose bundled seccomp profile blocks the
# clone3 syscall. bookworm+ glibc (>=2.34) uses clone3 to spawn sub-processes, so
# apt/dpkg die mid-build ("Problem executing scripts APT::Update::Post-Invoke ...
# Sub-process returned an error code"). bullseye ships glibc 2.31, which does not
# use clone3, so the build works without touching the shared daemon's seccomp.
FROM python:3.12-slim-bullseye

WORKDIR /app

# build-essential: hdbscan / llvmlite (umap) compile native extensions on install.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Pin torch to 2.7.1+cu126 BEFORE installing reviewscope_ml. Left unpinned,
# sentence-transformers pulls the latest torch, which has dropped Pascal (sm_61)
# kernels — this box's GPUs are TITAN X (Pascal), so ops fail with "no kernel
# image is available for execution on the device". torch 2.7.1 ships sm_60 cubins
# that run on sm_61 (verified against the host .venv). Installing it first pins
# the resolver so the reviewscope_ml install below keeps this build.
RUN pip install --no-cache-dir torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126

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
