#!/usr/bin/env bash
# Run this LOCALLY to push code and data to the GPU server.
# Usage: bash scripts/sync_to_gpu.sh

set -euo pipefail

REMOTE_USER="4dmarten"
JUMP_HOST="rzssh1.informatik.uni-hamburg.de"
GPU_HOST="ltgpu2.informatik.uni-hamburg.de"
REMOTE_DIR="~/ReviewScope"

LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SSH_OPTS="-o ProxyJump=${REMOTE_USER}@${JUMP_HOST}"

echo "==> Syncing code to ${GPU_HOST}:${REMOTE_DIR}..."
rsync -avz --progress \
    --exclude='.git' \
    --exclude='.venv' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.ipynb_checkpoints' \
    --exclude='data/raw' \
    --exclude='data/cache/embeddings' \
    --exclude='data/cache/umap' \
    --exclude='data/cache/clustering' \
    --exclude='data/cache/bertopic' \
    -e "ssh $SSH_OPTS" \
    "$LOCAL_ROOT/" \
    "${REMOTE_USER}@${GPU_HOST}:${REMOTE_DIR}/"

echo ""
echo "==> Uploading sample data files..."
# Create remote cache dir first
ssh $SSH_OPTS "${REMOTE_USER}@${GPU_HOST}" "mkdir -p ${REMOTE_DIR}/data/cache"

rsync -avz --progress \
    -e "ssh $SSH_OPTS" \
    "$LOCAL_ROOT/data/cache/sample_5k.jsonl" \
    "$LOCAL_ROOT/data/cache/sample_hotels_5k.jsonl" \
    "$LOCAL_ROOT/data/cache/sample_hotels_10k.jsonl" \
    "$LOCAL_ROOT/data/cache/sample_restaurants_5k.jsonl" \
    "$LOCAL_ROOT/data/cache/sample_restaurants_10k.jsonl" \
    "${REMOTE_USER}@${GPU_HOST}:${REMOTE_DIR}/data/cache/"

echo ""
echo "==> Done. Now run on the server:"
echo "    ssh -J ${REMOTE_USER}@${JUMP_HOST} ${REMOTE_USER}@${GPU_HOST}"
echo "    cd ${REMOTE_DIR} && bash scripts/setup_gpu.sh"
