#!/usr/bin/env bash
# One-command local dev: Postgres + Redis in Docker, API + Celery worker +
# frontend native. Reads the repo-root .env. Ctrl-C stops everything.
#
#   scripts/dev-local.sh
#
# Prereqs (one-time): a repo-root .env, src/backend/.venv with requirements
# installed, and `npm install` in src/frontend. See src/backend/README.md.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ ! -f "$ROOT/.env" ]; then
    echo "No .env at repo root. Copy .env.example to .env first." >&2
    exit 1
fi
set -a; . "$ROOT/.env"; set +a

BACKEND_VENV="$ROOT/src/backend/.venv"
if [ ! -x "$BACKEND_VENV/bin/uvicorn" ]; then
    echo "Backend venv missing. Run: python -m venv src/backend/.venv && src/backend/.venv/bin/pip install -r src/backend/requirements.txt" >&2
    exit 1
fi

echo "==> Ensuring Postgres + Redis are up..."
docker compose up -d db redis

echo "==> Applying migrations..."
( cd "$ROOT/src/backend" && "$BACKEND_VENV/bin/alembic" upgrade head )

pids=()
cleanup() { echo; echo "==> stopping..."; for p in "${pids[@]}"; do kill "$p" 2>/dev/null || true; done; }
trap cleanup EXIT INT TERM

echo "==> Starting API, worker, frontend..."
( cd "$ROOT/src/backend" && exec "$BACKEND_VENV/bin/uvicorn" app.main:app --reload --port 8000 ) &
pids+=($!)
( cd "$ROOT/src/backend" && exec "$BACKEND_VENV/bin/celery" -A app.worker.celery_app worker --loglevel=info ) &
pids+=($!)
( cd "$ROOT/src/frontend" && exec npm run dev ) &
pids+=($!)

echo ""
echo "  API:      http://localhost:8000  (docs at /docs)"
echo "  Frontend: http://localhost:5173"
echo "  Ctrl-C to stop all three."
echo ""
wait
