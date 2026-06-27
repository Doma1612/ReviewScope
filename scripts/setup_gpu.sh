#!/usr/bin/env bash
# Run this ON the GPU server after syncing the project.
# Usage: bash scripts/setup_gpu.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$PROJECT_ROOT/.venv"

echo "==> Project root: $PROJECT_ROOT"

# ── 1. Create data directories ─────────────────────────────────────────────────
echo "==> Creating data directories..."
mkdir -p "$PROJECT_ROOT/data/cache/embeddings"
mkdir -p "$PROJECT_ROOT/data/cache/umap"
mkdir -p "$PROJECT_ROOT/data/cache/clustering"
mkdir -p "$PROJECT_ROOT/data/cache/bertopic"
mkdir -p "$PROJECT_ROOT/data/raw"

# ── 2. Create virtual environment ─────────────────────────────────────────────
if [ ! -d "$VENV" ]; then
    echo "==> Creating virtual environment..."
    python3 -m venv "$VENV"
else
    echo "==> Virtual environment already exists, skipping."
fi

source "$VENV/bin/activate"
echo "==> Python: $(which python) ($(python --version))"

# ── 3. Install dependencies ────────────────────────────────────────────────────
echo "==> Upgrading pip..."
pip install --upgrade pip --quiet

echo "==> Installing requirements (this may take a few minutes)..."
# hdbscan must come before bertopic
pip install hdbscan --quiet
pip install InstructorEmbedding --quiet
pip install -r "$PROJECT_ROOT/requirements.txt" --quiet

# ── 4. Register venv as Jupyter kernel ────────────────────────────────────────
echo "==> Registering Jupyter kernel..."
python -m ipykernel install --user --name reviewscope --display-name "ReviewScope"

# ── 5. Verify GPU ─────────────────────────────────────────────────────────────
echo "==> Checking GPU..."
python -c "
import torch
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('  WARNING: No CUDA GPU detected.')
"

echo ""
echo "==> Setup complete."
echo "    Activate with: source $VENV/bin/activate"
