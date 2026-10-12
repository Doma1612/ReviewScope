# Running ReviewScope from published images

ReviewScope is published to Docker Hub as three images (amd64 + arm64):

| Image | Contents |
|---|---|
| `dmnkmrtns/reviewscope-backend` | FastAPI API **and** Celery ML worker (same image, different command) |
| `dmnkmrtns/reviewscope-frontend` | React UI (Vite dev server) |
| `dmnkmrtns/reviewscope-ollama` | Ollama, pulls the label model on first boot |

Postgres (pgvector) and Redis come from their official images. The images
contain code only — **no datasets, no projects, no secrets**. Each user
registers an account and uploads their own CSV/JSONL.

## Run it (Linux, Windows, macOS incl. Apple Silicon)

Requires Docker Desktop (Windows/macOS) or Docker Engine with the Compose plugin
(Linux). About 15 GB of free disk space.

1. Download `docker-compose.release.yml` and `.env.example` from the repository
   into an empty folder. Rename `.env.example` to `.env`.
2. In `.env`, set at least `SECRET_KEY`, `POSTGRES_USER`, `POSTGRES_PASSWORD`,
   `POSTGRES_DB`. To use a pre-release, add `REVIEWSCOPE_VERSION=0.1.0-rc1`
   (default: `latest`, the newest stable release).
3. Start:
   ```
   docker compose -f docker-compose.release.yml up -d
   ```
   The first start downloads the images and the label model (~2 GB); the
   worker waits until the model is ready.
4. Open http://localhost:5173, register, and create a project.

Stop with `docker compose -f docker-compose.release.yml down`. Data lives in
Docker volumes and survives restarts; `down -v` deletes it.

### With an NVIDIA GPU (Linux / Windows WSL2)

Needs the NVIDIA driver and `nvidia-container-toolkit`. Download
`docker-compose.gpu.yml` as well and start with:
```
docker compose -f docker-compose.release.yml -f docker-compose.gpu.yml up -d
```
Each pipeline run claims one idle GPU (`ML_MAX_GPUS`, default 1) and falls back
to CPU when every card is busy. The amd64 image ships CUDA 12.6 torch, which
covers Pascal through RTX 40xx — not RTX 50xx (Blackwell). arm64 is CPU-only.

## What to expect on CPU

Embedding and sentiment are the GPU-accelerated steps; on CPU they dominate
the runtime. UMAP (the "Reducing" step) is single-threaded on every machine
because the seed is fixed for reproducibility (see methodology §4). A few
thousand reviews take minutes on a laptop; 50k reviews is a GPU-sized job.

## Publishing a release

Images are built by `.github/workflows/docker-publish.yml` on GitHub's native
x86 and arm64 runners:

| Push | Docker tags |
|---|---|
| tag `v0.1.0` | `0.1.0`, `latest` |
| tag `v0.1.0-rc1` | `0.1.0-rc1` (pre-releases never move `latest`) |
| branch `main` | `main`, `sha-<short>` |

The workflow needs the repository variable `DOCKERHUB_USERNAME` and the secret
`DOCKERHUB_TOKEN` (a Docker Hub personal access token with Read & Write).
Besides the final tags it pushes per-architecture helper tags
(`<tag>-amd64`, `<tag>-arm64`); they can be deleted on Docker Hub once the
multi-arch tag exists. Old or test tags are removed under the repository's
**Tags** page on hub.docker.com.

## Development vs. release compose files

| File | Use |
|---|---|
| `docker-compose.yml` | development: builds from source, mounts the code, CPU-only |
| `docker-compose.gpu.yml` | override: NVIDIA GPU via nvidia-container-toolkit |
| `docker-compose.ltgpu.yml` | override: the ltgpu1 university box (old Docker, no toolkit) |
| `docker-compose.release.yml` | end users: published images, no source needed |
