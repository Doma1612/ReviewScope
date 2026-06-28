# ML integration changelog — wiring `reviewscope_ml` into the backend

**Branch:** `integrate-ml-into-app` · **Date:** 2026-06-27

This records the change that replaced the backend's *simulated* pipeline with a real
call into the `reviewscope_ml` integration seam (`docs/integration-guide.md`). It is the
"what was done / what changed / why" companion to that contract doc.

## Goal

The app already ran end-to-end on fabricated data (`app/tasks.run_simulated_pipeline`).
The ML package shipped a complete, framework-agnostic seam (`reviewscope_ml.app.run_from_upload`)
that the backend never called. This change calls it for real — runnable end-to-end via Docker
Compose, Ollama labeling included — while keeping the simulated path behind a flag so tests and
lightweight runs need no GPU or heavy deps.

**`reviewscope_ml/` was not modified** (per `AGENTS.md`: don't touch the ML source; simulate ML
in tests rather than implementing it there).

## What changed

### New backend modules

- **`app/ml_mapping.py`** — pure adapters (no Celery / FastAPI / torch), so the mapping logic
  is unit-testable in isolation:
  - `derive_roles(columns)` / `build_upload_schema(columns)` — turn the stored
    `project_schema.columns` into the seam's `UploadSchema`. The seam needs a designated
    **text column** (the column table only implies it) and an optional **rating column**;
    `derive_roles` picks a non-PK `text` column (preferring conventional names — a PK is
    identity, not content) and a rating-named numeric column.
  - `DbProgressSink` — implements the seam's `ProgressSink` port. Writes each step to
    `pipeline_jobs` via a short-lived sync session and commits immediately so
    `GET /pipeline/status` advances mid-run. The seam emits Capitalized canonical names
    (`Ingest`…`Finalize`) and only an explicit `done` for `Finalize`; the sink lowercases to
    match the pre-created rows and marks every earlier step `done` on each `running` event so
    the dashboard bar fills monotonically.
  - `result_to_orm(result)` / `persist_run_result(session, result)` — map a finished
    `RunResult` onto `Cluster` / `Document` / `Embedding` rows. UUIDs are assigned eagerly so
    the pipeline's **integer** cluster ids resolve to row UUIDs and embeddings join to
    documents by `primary_key_value` **before** any flush; noise documents (`cluster_id is None`)
    keep a `NULL` FK (integration-guide §3).
- **`app/ml_pipeline.py`** — the real Celery task `run_ml_pipeline`. Sets status `processing`,
  lazily imports `reviewscope_ml.app` (so torch/sentence-transformers/umap load *only* on this
  path), calls `run_from_upload(... progress=DbProgressSink, device=settings.ml_device,
  label_clusters=True)`, persists, sets `ready`. `IngestError` → `failed` with the per-row
  `.errors`; any other exception → `failed`; running jobs are marked failed.
- **`app/api/system.py`** — `GET /api/models` (was missing vs the app spec / integration-guide §6).
  Returns static names when simulating, else the frozen `app_default_spec()` embedding + label model.

### Edited

- **`app/core/config.py`** — `SIMULATE_ML` (default **true** — safe; the Docker worker opts into
  the real path) and `ML_DEVICE` (default `cpu`) settings.
- **`app/db/session.py`** — added a **sync** engine + `SyncSessionLocal`. The real run is a long
  blocking call that emits progress between stages, so the worker and progress sink use a sync
  session (psycopg) rather than asyncio. The async path is untouched.
- **`app/api/projects.py`** — `create_project` now dispatches `run_ml_pipeline` vs
  `run_simulated_pipeline` based on `settings.simulate_ml`.
- **`app/worker.py`** — explicitly imports both task modules (after `celery_app` is defined, so
  the `app.* → app.worker` import resolves) to register them in the worker.
- **`app/models.py`** + **`alembic/versions/0002_*.py`** — `clusters` gains `label_source`
  (`ollama:<model>` | `terms_fallback`, so the UI can flag non-LLM labels) and `mean_stars`.
- **`app/schemas.py`** — `ClusterRead` surfaces the two new fields; new `ModelsRead`.

### Infrastructure

- **`src/backend/Dockerfile`** — build context is now the repo root; installs the
  `reviewscope_ml` package (heavy ML deps as a cache-worthy layer) then the backend
  requirements, then copies the backend app. The compose dev bind-mount overlays the app code
  but not the installed package.
- **`docker-compose.yml`** — `api`/`worker` build from root with `SIMULATE_ML=false`; worker also
  sets `ML_DEVICE`, `REVIEWSCOPE_ROOT=/workspace`, and mounts `./data:/workspace/data` for the ML
  cache. New **`ollama`** service. The worker uses **`network_mode: "service:ollama"`** (see
  below). `.env.example` documents `SIMULATE_ML` / `ML_DEVICE`.

## Notable decisions

- **Sync worker session.** Progress must be visible *during* the synchronous ML run, which can't
  `await` between stages — hence a dedicated sync engine rather than `asyncio.run`.
- **Step-name casing.** The seam's canonical `Ingest…Finalize` is mapped to the existing
  lowercase `pipeline_jobs` rows in the sink, avoiding a migration of pre-created rows.
- **Ollama `localhost` via shared netns.** `reviewscope_ml`'s `OllamaLabeler` hard-codes
  `http://localhost:11434` and exposes no base-URL override, and the package is frozen. To reach a
  separate `ollama` container without editing ML code, the worker shares the ollama container's
  network namespace (`network_mode: "service:ollama"`); `localhost:11434` then hits Ollama while
  `db`/`redis` still resolve. If Ollama/model is absent, labeling **fails soft** to term-fallback
  by design (`label_source` records which).

## Follow-ups (out of scope here)

- `embeddings.vector` is still JSONB, not a pgvector-typed column. Switch to `vector(768)` +
  the extension when similarity queries are needed.
- Cleaner Ollama wiring: add a `base_url` env hook to `OllamaLabeler` once the `reviewscope_ml`
  freeze lifts, and drop the `network_mode` trick.
- Sentence-level (Phase 2): a `segments` table + per-segment scatter; the seam rejects
  `sentence_level` until then.
- Frontend still renders terms as plain spans (no word cloud), hard-codes a 3-D scatter, and
  doesn't consume `/api/models` — tracked separately from this ML wiring.

## Verification

- **Unit (no DB/GPU):** `python -m pytest src/backend/tests/test_ml_integration.py` — covers
  `derive_roles`, the `DbProgressSink` step logic, and `result_to_orm` cluster-UUID resolution /
  embedding join. Passed.
- **Import smoke:** all touched modules import and both Celery tasks register (no circular import).
- **Simulated path unchanged:** with `SIMULATE_ML=true`, upload still fills the dashboard via the
  mock.
- **Real end-to-end (run by the user — needs Docker/GPU/Ollama, not available in CI):**
  ```bash
  docker compose up -d db redis ollama
  docker compose exec ollama ollama pull llama3.2      # else term-fallback labels
  docker compose run --rm api alembic upgrade head
  docker compose up --build
  ```
  Upload a small CSV/JSONL, watch `pipeline_jobs` advance ingest→finalize, confirm real
  clusters/labels/embeddings render and `GET /api/models` returns the real model names.
