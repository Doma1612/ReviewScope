# Build context is the REPOSITORY ROOT (see docker-compose.yml: build.context: .),
# so this image can install the reviewscope_ml package alongside the backend.
FROM python:3.12-slim

WORKDIR /app

# build-essential: hdbscan / llvmlite (umap) compile native extensions on install.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

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
