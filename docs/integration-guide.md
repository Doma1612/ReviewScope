# Integration guide — wiring the ML pipeline into the backend

This is the contract between the `reviewscope_ml` package and the FastAPI/Celery
backend. It describes the one entry point the worker calls, the two interfaces
the backend implements, and how the pipeline's output maps onto the app spec's
database tables. Companion docs: `application-spec.md` (the app), `pipeline-guide.md`
(how the ML works), `methodology.md` (why).

The integration layer lives in **`src/reviewscope_ml/app/`** and is
framework-agnostic: no FastAPI, Celery, SQLAlchemy or pydantic. The backend owns
those; this layer turns an uploaded file into DB-ready records by driving the
existing pipeline runner.

---

## 1. The seam in one call

```python
from reviewscope_ml.app import run_from_upload, UploadSchema, ColumnSpec

result = run_from_upload(
    file_path="/data/uploads/proj42.jsonl",
    schema=UploadSchema(
        columns=[
            ColumnSpec("review_id", "text", is_primary_key=True),
            ColumnSpec("text", "text"),
            ColumnSpec("stars", "integer"),
            ColumnSpec("date", "date"),
        ],
        text_column="text",        # the column the NLP pipeline embeds/clusters
        rating_column="stars",     # optional; drives star display + Tier-3 + sentiment
    ),
    project_id=str(project.id),
    progress=DbProgressSink(job),  # YOU implement this (ProgressSink)
    device="cuda",                 # "cpu" | "cuda"
)
# result is a RunResult of plain dataclasses — persist it:
repository.save(result)            # YOU implement this (ResultRepository), or insert directly
```

`run_from_upload` runs all eight pipeline steps end to end. If you already hold an
ingested corpus, call `run_project_pipeline(corpus=..., ...)` instead (skips steps 1–2).

**Validation:** `run_from_upload` raises `IngestError` (carrying `.errors`, a list of
per-row messages) when the file violates its schema — map it to a 4xx and a `failed`
pipeline status. No auto-fixing, per the upload-modal spec.

---

## 2. What you implement: two ports

Both are `typing.Protocol`s in `reviewscope_ml.app.ports` — structural, so just match
the shape; no base class to inherit.

### `ProgressSink` — drives `GET /pipeline/status`

```python
class DbProgressSink:
    def __init__(self, job): self.job = job
    def step(self, name, status, message="", index=0, total=8):
        # upsert a pipeline_jobs row: step=name, status=status, message=message
        update_pipeline_job(self.job, step=name, status=status,
                            message=message, index=index, total=total)
```

The service calls `step(...)` as each stage starts, using the **canonical 8-step
vocabulary** (`ports.PIPELINE_STEPS`): `Ingest, Preprocess, Embed, Reduce, Cluster,
Sentiment, Label, Finalize`. `index` is the 1-based step number, so the frontend can
render `Clustering… step 5/8`. The final call is always `("Finalize", "done", index=8)`.
Wrap the whole task in try/except and emit `(current_step, "failed", message=str(err))`
on exception.

> Granularity note: the MVP runs the pipeline as **one Celery task** that emits these
> progress events. That satisfies the polling contract without the plumbing of passing
> large intermediates between eight separate tasks. Decompose later only if you need
> per-stage restartability.

### `ResultRepository` — optional persistence helper

```python
class SqlAlchemyRepo:
    def save(self, result: RunResult) -> None: ...
```

You can implement this or just insert the DTO lists directly in the task. Either way the
mapping is §3.

---

## 3. Output → database mapping

`RunResult` (in `reviewscope_ml.app.dto`) is three lists of dataclasses that mirror the
app-spec tables field-for-field.

| DTO | Table | Notes |
|---|---|---|
| `DocumentRecord` | `documents` | `primary_key_value`, `text`, `raw_data` (all original columns, type-coerced JSON), `cluster_id`, `sentiment_score` |
| `EmbeddingRecord` | `embeddings` | `vector` (768-d `list[float]` → cast to pgvector), `umap_x/y/z` |
| `ClusterRecord` | `clusters` | `label`, `summary`, `label_source`, `top_terms` `[{term,score}]`, `word_frequencies`, `size`, `sentiment_avg`, `mean_stars`, `sample_doc_ids` |

Plus `result.manifest` (provenance: spec, seed, per-stage cost, label sources) and
`result.metrics` (three-tier metrics + failure flags) for `pipeline_jobs` / audit.

**Two things the backend must resolve:**

1. **Integer cluster id → UUID.** `DocumentRecord.cluster_id` is the pipeline's integer
   id (or `None` for noise/unassigned). Insert the `ClusterRecord`s first, build an
   `{int → cluster.uuid}` map, then set each document's FK. `None` stays `NULL`.
2. **Coordinates.** `umap_x/y/z` carry the **3-D** UMAP projection; the 2-D scatter uses
   `(x, y)`. The pipeline also computes a dedicated 2-D projection (in the run-directory
   artifact `assignments.csv`), but the spec's `embeddings` table has only one coordinate
   triple, so we store 3-D and let the 2-D view drop `z`. If you want a truer 2-D layout,
   add `umap_x2/y2` columns and surface `RunArtifacts.coords_2d` — see §6.

The join key between documents and embeddings is `primary_key_value`.

---

## 4. Config & caching are namespaced per project

`service.project_config(project_id, n_docs=...)` builds a `PipelineConfig` whose
`corpus_slug` is `proj-<slug-of-project-id>`. Every cached artifact (embeddings, UMAP,
clustering) and the run directory are keyed on it, so projects never collide with each
other or with the research benchmark (`hotels`). The synthetic `data_file` name is never
read — the pipeline receives its data in memory. Re-running the same project with the
same data + seed is almost free (cache hits); a crash mid-run resumes from the last
cached stage.

Embeddings for pgvector are reloaded from that cache by `service.load_run_embeddings`
(no recompute, no model reload) immediately after the run.

---

## 5. The frozen pipeline (and the doc-level decision)

The app runs **one** configuration, defined in `app/defaults.py`:
`APP_DEFAULT_VARIANT = "custom_hdbscan"` (embed → UMAP → HDBSCAN, `all-mpnet-base-v2`).
Swapping the winner after the doc-level sweep / human sign-off
(see `quality-roadmap.md`) is a one-line change there.

**Why document-level, not the better-scoring `sentence_level`:** the spec's schema is
one-cluster-per-document and one-point-per-document. `sentence_level` clusters *mentions*
(sentence segments) — a document then maps to several clusters and the scatter shows
segments, not documents. That needs a `segments` table the spec doesn't have, so
`run_project_pipeline` **rejects `sentence_level`** today. It's a deliberate Phase-2
extension (§6), not an oversight.

---

## 6. Deferred to the backend / Phase 2

Reference scaffolding to write (examples can follow on request):

- **Celery task** wrapping `run_from_upload` with a `DbProgressSink` + repo, plus
  try/except → `failed` status.
- **SQLAlchemy models + Alembic** for the spec's tables; pgvector extension.
- **`GET /api/models`** — expose the available embedding model (`app_default_spec().embedding_model`)
  and the LLM (`spec.label_model`).
- **Ollama**: labeling calls Ollama; it **fails soft** to term-join labels when Ollama is
  down (every cluster carries `label_source`, so unreviewed/fallback labels are always
  distinguishable). Wire Ollama as a compose service; pass `label_clusters=True` (default).
- **GPU**: `device="cuda"` makes the embed stage claim an idle GPU via the runtime
  etiquette already in the package (≤50% VRAM, idle devices only). Nothing extra to do.
- **Phase-2 sentence-level**: add a `segments` table, per-segment scatter points, and use
  the runner's `doc_membership.json` (primary cluster per review) to fill
  `documents.cluster_id`. Then allow `sentence_level` in `defaults.py`.

**TODO — DTO single source of truth (decide with whoever owns persistence).** The DTO
shape is currently defined twice: here (`app/dto.py`) and in the backend's ORM models,
which can drift. Either keep these DTOs as the canonical contract and map DTO → ORM
(status quo), or extract a tiny shared `reviewscope_contracts` package both sides import.
Note the dependency direction: the ML package must **not** import the backend (ports &
adapters — the core has no outward dependencies), so a backend-owned base class the DTOs
inherit from is *not* an option; a shared definition must live in a neutral package. See
the matching TODO in `app/dto.py`.

---

## 7. Tests

`tests/test_app_*.py` cover the schema, ingest (CSV/JSONL, filters, dedup, validation
errors), the persistence mapping, and the service orchestration (with the runner mocked,
so they stay pure-logic — no GPU, no model downloads). Run them with the rest:

```bash
pytest tests/test_app_schema.py tests/test_app_ingest.py \
       tests/test_app_persistence.py tests/test_app_service.py
```

A full end-to-end smoke (tiny CSV → real CPU run → DTOs) is the natural next test once
the backend can call it — that's the Phase-5 handover check in the task-3 plan.
