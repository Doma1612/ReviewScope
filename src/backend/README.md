# Backend

FastAPI application and Celery worker. The API handles accounts, projects, uploads, cluster editing and chat. The worker runs the ML pipeline from `reviewscope_ml` and stores the results in PostgreSQL.

## Structure

| Path | Responsibility |
|---|---|
| `app/main.py` | FastAPI app, CORS, router registration, `GET /api/health` |
| `app/api/auth.py` | Register, login, logout, current user. JWT in an httpOnly cookie |
| `app/api/projects.py` | Projects, upload, schema, pipeline status, embeddings, clusters, documents, segments, members, edit history, cluster chat |
| `app/api/project_chat.py` | Project chat over all documents (plain and streaming) |
| `app/api/system.py` | `GET /api/models`: embedding and label model in use |
| `app/api/deps.py` | Current user and project permission dependencies |
| `app/models.py` | SQLAlchemy tables (see below) |
| `app/schemas.py`, `app/project_chat_schemas.py` | Pydantic request and response models |
| `app/core/config.py` | Settings from environment variables |
| `app/core/security.py` | Password hashing and JWT handling |
| `app/db/` | Async engine for the API, sync engine for the worker |
| `app/worker.py` | Celery app; one task per worker process |
| `app/ml_pipeline.py` | Celery task that runs `reviewscope_ml` on an upload and persists the result |
| `app/ml_mapping.py` | Converts the upload schema and the pipeline result to and from database rows; reports step progress |
| `app/tasks.py` | Simulated pipeline for tests and runs without ML dependencies (`SIMULATE_ML=true`) |
| `app/services/edits.py` | Writes the cluster edit log |
| `app/services/recompute.py` | Recomputes cluster size, terms and sentiment after edits |
| `app/services/replay.py` | Reapplies manual edits after a pipeline rerun |
| `app/services/metrics.py` | Cluster cohesion metric |
| `app/services/project_rag.py` | Retrieval of relevant documents for the project chat |
| `app/services/project_chat_tools.py` | Tools the chat model may call (cluster count, ranking, related clusters, evidence search), validated on the server |
| `app/services/query_embeddings.py` | Embeds chat queries on CPU with the project's embedding model |
| `app/services/ollama_chat.py` | Client for Ollama chat completions |
| `alembic/` | Database migrations |
| `tests/` | Unit tests; no database or GPU required |

## Data model

| Table | Content |
|---|---|
| `users` | Accounts |
| `projects`, `project_members`, `project_schema` | Project metadata, owner and viewer roles, confirmed column types |
| `documents` | Uploaded rows: primary key, text, all original columns, primary cluster, sentiment |
| `segments` | Sentences of each document; the unit that is clustered |
| `embeddings` | Vectors (pgvector) and 2D/3D map coordinates |
| `clusters` | Label, summary, top terms, word frequencies, size, sentiment, label source |
| `cluster_edits` | Log of manual edits, replayed after reruns |
| `pipeline_jobs` | Status and message per pipeline step |

## Pipeline run

1. `POST /api/projects` stores the file, saves the confirmed schema and queues `run_ml_pipeline`.
2. The worker claims an idle GPU if `ML_DEVICE=cuda`, otherwise runs on CPU.
3. `reviewscope_ml.app.run_from_upload` runs the steps Ingest, Preprocess, Embed, Reduce, Cluster, Sentiment, Label, Finalize. Each step updates `pipeline_jobs`, which the frontend polls.
4. The result replaces the project's clusters, documents and segments. Earlier manual edits are then reapplied, and the project becomes `ready`.

## Configuration

Settings are read from the repository root `.env` (template: `.env.example`).

| Variable | Default | Meaning |
|---|---|---|
| `SECRET_KEY` | required | JWT signing key |
| `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`, `POSTGRES_HOST`, `POSTGRES_PORT` | | Database connection |
| `REDIS_URL` | `redis://localhost:6379/0` | Celery broker and result backend |
| `SIMULATE_ML` | `true` | `false` runs the real pipeline; Docker Compose sets `false` |
| `ML_DEVICE` | `cpu` | `cuda` requests a GPU per run |
| `ML_MAX_GPUS` | `1` | Idle GPUs one run may claim; `0` means all idle GPUs |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server for labels and chat |
| `OLLAMA_CHAT_MODEL` | `llama3.2` | Model for cluster and project chat |

## Development

The usual way is the Docker Compose stack described in the root README; it runs migrations on start. Without Docker for the API and worker:

```
docker compose up -d db redis
python -m venv src/backend/.venv
src/backend/.venv/bin/pip install -r src/backend/requirements.txt
npm --prefix src/frontend install
scripts/dev-local.sh
```

`scripts/dev-local.sh` loads `.env`, runs the migrations and starts API, worker and frontend. With `SIMULATE_ML=true` no ML dependencies are needed. For the real pipeline, also run `pip install -e .` from the repository root and set `SIMULATE_ML=false`.

New migration after a model change:

```
docker compose run --rm api alembic revision --autogenerate -m "describe change"
```

Tests:

```
cd src/backend && python -m pytest
```
