# ReviewScope — Implementation Plan (standalone work packages)

This plan turns `docs/frontend-feature-gap-analysis.md` into **self-contained work
packages (WPs)**. Each WP is written so you can paste it into a fresh Claude Code
session and implement it without the others in context.

## How to use this file
1. Copy the **Shared Preamble** below.
2. Copy the **one WP** you want to build (e.g. `B1`).
3. Paste both into a new session (preamble first). That's all the context the
   session needs.

Before starting a WP, check its **Prereqs** are `✅ Done` in the Progress Tracker.
WPs are grouped and ordered by dependency: backend WPs (B*) generally come before
the frontend WPs (F*) that consume them.

---

## Progress Tracker
Single source of truth for status. Update the box here **and** the WP's own
`**Status:**` line together. `☐` not started · `▶` in progress · `✅` done.

- ✅ **B1** — cluster_edits audit table + model + migration
- ✅ **B2** — recompute service for cluster aggregates
- ✅ **B3** — document reassignment endpoints
- ✅ **B4** — cluster CRUD + merge + from-selection + rename/junk
- ✅ **B5** — GET/POST project schema endpoint
- ✅ **B6** — re-run survival (replay edits) + label_source protection
- ✅ **B7** — rich hover payload on embeddings
- ☐ **F0** — API client + types (`api.ts`)
- ☐ **F1** — scatter real hover
- ☐ **F2** — scatter 2D/3D toggle + click→highlight
- ☐ **F3** — cluster word cloud + star rating + distribution chart
- ☐ **F4** — "show all documents" paginated table
- ☐ **F5** — editing UX (selection toolbar, card actions, edit-mode)
- ☐ **F6** — optimistic updates + invalidation + WebGL cap
- ☐ **F7** — undo / edit-history panel
- ☐ **F8** — upload step 2 real schema confirm
- ☐ **F9** — owner email + member role-change + models/health
- ☐ **T1** — provenance badge + confirm/sign-off
- ☐ **T2** — export (CSV/JSON/PNG/report)
- ☐ **T3** — search / filter / noise triage
- ☐ **T4** — re-run / re-cluster controls + append data

---

## Shared Preamble (prepend to every WP)

> **Repo:** ReviewScope — an NLP qualitative-text-analysis platform (masters
> course project). Ports-and-adapters architecture. A React/Vite frontend talks
> to a FastAPI backend (async SQLAlchemy + Postgres, Celery worker), which wraps a
> separate `reviewscope_ml` Python package via an integration seam.
>
> **Key paths:**
> - Backend app: `src/backend/app/` — API routers in `api/` (`projects.py`,
>   `system.py`, `auth.py`, `deps.py`), ORM in `models.py`, Pydantic DTOs in
>   `schemas.py`, ML→ORM adapters in `ml_mapping.py`, Celery ML task in
>   `ml_pipeline.py`, simulated task in `tasks.py`.
> - Alembic migrations: `src/backend/alembic/versions/` (latest is
>   `0002_cluster_fields`).
> - Frontend: `src/frontend/src/` — API client `api.ts`, views in `views/`
>   (`Dashboard.tsx`, `ProjectView.tsx`, `ClusterDetail.tsx`, `DeckDashboard.tsx`,
>   `SettingsView.tsx`, `PipelineView.tsx`), routes in `main.tsx`. Uses
>   `@tanstack/react-query` + `react-plotly.js` + `react-router-dom`.
> - ML reference (DO NOT import into the backend casually — heavy deps; import
>   lazily): HITL editing semantics in `src/reviewscope_ml/hitl/feedback.py` and
>   `hitl/apply_feedback.py`; term/word-freq recompute in
>   `src/reviewscope_ml/represent/terms.py` (`ctfidf_terms`, `tfidf_top_terms`,
>   `word_frequencies`). The Streamlit prototype `hitl/app.py` is the UX
>   reference only — reuse the semantics, not the code.
>
> **Conventions:**
> - Routers are mounted under `/api/projects` (see `main.py`). Mutations must call
>   `require_project_role(db, project_id, current_user.id, {ProjectRole.owner})`
>   from `api/deps.py` — viewers are read-only per spec.
> - Python venv: use `.venv/bin/python` (not conda/system). Backend has its own
>   `src/backend/.venv`.
> - Frontend mutations use react-query `useMutation` + `queryClient.invalidateQueries`.
> - Cluster ids are row UUIDs in the DB; the ML package uses int ids. "Noise" =
>   `Document.cluster_id IS NULL`.
> - Add a matching Alembic migration for any model change; keep `models.py` and
>   the migration in sync.
>
> **House rules:** match surrounding code style, keep adapters free of heavy ML
> imports, add/update tests under `src/backend/tests/` where one exists for the
> area, and run the backend test suite after backend changes.
>
> **When you finish this WP:** update `docs/frontend-feature-implementation-plan.md`
> — set this WP to `✅` in the **Progress Tracker** list AND on the WP's own
> `**Status:**` line (add the date and a one-line note of anything that deviated
> from the plan, e.g. a renamed endpoint). If you discover a new prerequisite or a
> follow-up, note it on the relevant WP so the next session sees it. Keep the two
> status locations in sync.

---

# Group B — Backend mutation layer (do first)

## B1 — `cluster_edits` audit table + model + migration
**Status:** ✅ Done (2026-06-28) · **Prereqs:** none

**Notes:** Implemented as planned. `action` is a `String(50)` column constrained by
a DB `CheckConstraint` (`ck_cluster_edits_action`) over `EDIT_ACTIONS` in
`models.py` (kept in sync with the migration's `ACTIONS` tuple) — not a Postgres
ENUM. Subject columns (`cluster_id`/`target_cluster_id`/`document_id`) are plain
UUIDs **without** FKs by design: cluster ids are regenerated on re-run and rows are
deleted, so the audit trail must outlive them (needed for B6 replay / F7 undo).
`actor_id` is a FK→users (no `ondelete`). Helper `record_edit` lives in
`app/services/edits.py` (new `services` package) and only stages the row — callers
commit it inside their own transaction. Endpoint `GET /api/projects/{id}/edits`
(owner+viewer) returns edits newest-first. Tests in
`tests/test_cluster_edits.py` (no-DB, like the existing suite). Migration verified
live against the dev Postgres (docker, port 5533): `upgrade head` →
`downgrade -1` → `upgrade head` all clean; table/index/FKs/CheckConstraint present
and the constraint rejects unknown actions (`frobnicate` → CHECK violation).

**Goal:** an append-only audit log of every cluster/document edit, mirroring the
ML package's `FeedbackRecord` (`src/reviewscope_ml/hitl/feedback.py`). This is the
backbone for re-run survival (B6) and the undo/history UI (F7).

**Do:**
1. Add a `ClusterEdit` model to `src/backend/app/models.py`:
   - `id` UUID PK; `project_id` FK→projects (CASCADE, indexed); `actor_id`
     FK→users; `created_at` timestamptz server_default now.
   - `action` String — constrain to the same vocabulary as `feedback.ACTIONS`
     plus the app-only actions: `reassign_doc`, `bulk_reassign`,
     `merge_clusters`, `create_cluster`, `create_from_selection`, `rename_label`,
     `approve_label`, `mark_junk`, `split_cluster`, `confirm_run`.
   - Nullable subject columns: `cluster_id` UUID, `target_cluster_id` UUID,
     `document_id` UUID, `new_label` Text, `note` Text.
   - `payload` JSONB (default `{}`) for action-specific extras (e.g.
     `{document_ids: [...]}` for bulk, `before`/`after` snapshots).
2. Add an Alembic migration `0003_cluster_edits` (down_revision
   `0002_cluster_fields`) creating the table + the `project_id` index.
3. Add a `ClusterEditRead` Pydantic schema in `schemas.py`
   (`from_attributes = True`).
4. Add a small helper `record_edit(db, *, project_id, actor_id, action, **fields)`
   (e.g. in a new `src/backend/app/services/edits.py`) that constructs and adds a
   `ClusterEdit`. Other WPs will call it inside their mutation transactions.
5. Add `GET /api/projects/{id}/edits` (owner+viewer) returning the project's
   edits newest-first; wire it into the router in `api/projects.py`.

**Acceptance:** migration applies cleanly up/down; `GET /edits` returns `[]` for a
fresh project; creating an edit row via the helper shows up in the list.

---

## B2 — Recompute service for cluster aggregates
**Status:** ✅ Done (2026-06-28) · **Prereqs:** none (B3/B4 call it)

**Done notes:** `app/services/recompute.py` with `recompute_cluster` /
`recompute_clusters` (async, caller owns the commit) + a `delete_if_empty` flag.
The numeric aggregates (`size`/`sentiment_avg`/`mean_stars`) are factored into a
pure `numeric_aggregates` helper plus `_parse_rating` so they unit-test without a
DB or sklearn (the backend venv has **no** sklearn — heavy ML imports must stay
lazy). `top_terms`/`word_frequencies` go through `reviewscope_ml.represent.terms`
(lazy import, all docs labelled `0`), shapes matching `ml_mapping.result_to_orm`.
Rating column resolved via `derive_roles(project_schema.columns)`. Tests in
`tests/test_recompute.py` (move-a-doc, None/empty means, rating coercion).

**Goal:** after any structural edit, recompute a cluster's `size`, `top_terms`,
`word_frequencies`, `sentiment_avg`, `mean_stars` from current membership —
"append + derive, don't patch" (gap doc §3/§4a). Reuses the same functions the ML
package uses so the app and notebook agree.

**Do:**
1. Create `src/backend/app/services/recompute.py` with:
   - `recompute_cluster(db, project_id, cluster_id)` — load that cluster's
     `Document.text` + per-doc `sentiment_score` + the rating value from
     `raw_data` (the rating column comes from `project_schema`; reuse
     `ml_mapping.derive_roles` to find the rating column name). Recompute:
     - `size` = count of member docs.
     - `sentiment_avg` = mean of non-null `sentiment_score`.
     - `mean_stars` = mean of the rating column parsed from `raw_data` (None if no
       rating column / no numeric values).
     - `top_terms` + `word_frequencies` via
       `reviewscope_ml.represent.terms.ctfidf_terms` and `word_frequencies`
       (import lazily). Build the `labels` array as all docs in this cluster
       labelled `0`; store `top_terms` as `[{"term","score"}]` and
       `word_frequencies` as `{word: count}` to match the existing JSONB shapes
       used in `ml_mapping.result_to_orm`.
   - `recompute_clusters(db, project_id, cluster_ids)` — loop helper.
   - If a cluster ends up empty after an edit, delete it (caller decides; expose a
     `delete_if_empty` flag).
2. Keep this module importable without heavy ML deps at import time (lazy-import
   `reviewscope_ml.represent` inside the function).
3. Add a unit test in `src/backend/tests/` that seeds a couple of docs and asserts
   `size`/`sentiment_avg` update after moving a doc.

**Acceptance:** calling `recompute_cluster` after a membership change yields
correct `size`, non-null terms/word_frequencies when texts exist, and correct
`sentiment_avg`/`mean_stars`.

---

## B3 — Document reassignment endpoints
**Status:** ✅ Done (2026-06-28) · **Prereqs:** B1, B2

**Done notes:** Implemented as planned in `api/projects.py`:
`PATCH /{project_id}/documents/{document_id}` (single) and
`POST /{project_id}/documents/reassign` (bulk), both owner-only, both validate the
doc + target cluster belong to the project (404 otherwise), write a `ClusterEdit`
via `record_edit`, then `recompute_clusters` over the affected set (union of old +
new, `None`/noise skipped) before a single commit. Deviations: (a) added a small
`BulkReassignResult` schema (`{moved:int}`) for the bulk `response_model` alongside
the planned `DocumentReassign`/`BulkReassign` request schemas; (b) the
`bulk_reassign` edit's `payload.document_ids` records the docs **actually moved**
(those found in the project), not the raw requested ids, and sets
`target_cluster_id` on the edit. Tests in `tests/test_document_reassign.py` —
no-DB fake `AsyncSession` driving the route fns with `recompute_clusters`
monkeypatched (backend venv has no sklearn/pytest-asyncio; coroutine tests run via
an `asyncio.run` decorator). Full backend suite: 21 passed. Note for later WPs:
importing `app.api.projects` runs `get_settings()` which `mkdir`s `upload_dir`, so
DB-touching tests must set `UPLOAD_DIR` to a writable temp dir.

**Goal:** move documents between clusters (single + bulk), the most-used edit.

**Do (in `api/projects.py`):**
1. `PATCH /{project_id}/documents/{document_id}` — body `{cluster_id: UUID|null}`
   (null = noise). Owner-only. Validate the doc + target cluster belong to the
   project. Capture `old_cluster_id`, set `cluster_id`, call
   `record_edit(... action="reassign_doc", document_id, cluster_id=old,
   target_cluster_id=new ...)`, then `recompute_clusters` for `{old, new}` (skip
   None), commit. Return the updated `DocumentRead`.
2. `POST /{project_id}/documents/reassign` — body
   `{document_ids: [UUID...], cluster_id: UUID|null}`. Owner-only. Bulk update,
   record one `bulk_reassign` edit with `payload={"document_ids":[...]}`,
   recompute every affected cluster (union of old + new), commit. Return
   `{moved: <count>}`.
3. Add Pydantic request schemas (`DocumentReassign`, `BulkReassign`) to
   `schemas.py`.

**Acceptance:** moving a doc changes its `cluster_id`; both source and target
cluster `size` update; a viewer gets 403; an audit row is written.

---

## B4 — Cluster CRUD + merge + from-selection + rename/approve/junk
**Status:** ✅ Done (2026-06-28) · **Prereqs:** B1, B2

Implemented in `api/projects.py` (schemas in `schemas.py`), tests in
`tests/test_cluster_crud.py` (16 cases, all green). Notes on what deviated:
- `merge` writes **one `ClusterEdit` per source** (`cluster_id=source`,
  `target_cluster_id=target`) — more faithful to the ml feedback semantics and
  easier for B6 replay than a single payload row.
- `PATCH` returns `ClusterRead | None`: for `mark_junk` the cluster is deleted so
  it returns `None` (200, null body); rename/approve return the updated
  `ClusterRead`. A `PATCH` with no field set is a 400.
- `mark_junk` is shared with `DELETE` via a private `_junk_cluster` helper.
- `create`/`from-selection` generate the cluster id eagerly (`id=uuid.uuid4()`)
  so docs can be reassigned without a pre-flush round-trip.

**Goal:** create/merge/relabel/delete clusters from the app, mirroring the HITL
actions in `apply_feedback.py`.

**Do (in `api/projects.py`, all owner-only, all write a `ClusterEdit`):**
1. `POST /{project_id}/clusters` — body `{label: str}` → create an empty cluster
   (`summary=""`, `label_source="hitl_override"`, empty terms/freqs, size 0).
   Return `ClusterRead`. Action `create_cluster`.
2. `POST /{project_id}/clusters/merge` — body `{source_ids: [UUID...],
   target_id: UUID}`. Reassign all docs of each source to target, delete the
   source clusters, `recompute_cluster(target)`. Action `merge_clusters` (one row
   per source, or one row with `payload={"source_ids":[...]}`). Validate all ids
   belong to the project and `target_id ∉ source_ids`.
3. `POST /{project_id}/clusters/from-selection` — body
   `{document_ids: [UUID...], label: str}`. Create a new cluster, reassign those
   docs into it, recompute the new cluster + every previously-owning cluster.
   Action `create_from_selection`. (Lasso → new cluster.)
4. `PATCH /{project_id}/clusters/{cluster_id}` — body
   `{label?: str, approve?: bool, mark_junk?: bool}`:
   - rename → set `label`, `label_source="hitl_override"`, action `rename_label`.
   - approve → set `label_source="hitl_approved"`, action `approve_label`.
   - mark_junk → docs → noise (`cluster_id=NULL`), delete cluster, action
     `mark_junk`. (Same effect as DELETE below; keep one implementation.)
5. `DELETE /{project_id}/clusters/{cluster_id}` — junk: member docs → noise,
   remove cluster. Action `mark_junk`. 204.
6. Add request schemas (`ClusterCreate`, `ClusterMerge`, `ClusterFromSelection`,
   `ClusterUpdate`) to `schemas.py`.

**Acceptance:** merge moves all docs and removes sources; from-selection produces a
cluster with the right size and recomputed terms; rename sets
`label_source="hitl_override"`; delete/junk nulls the docs' `cluster_id`; viewer
gets 403 everywhere; each call writes an audit row.

---

## B5 — `GET/POST /api/projects/{id}/schema`
**Status:** ✅ Done (2026-06-28) · **Prereqs:** none

**Done note:** `GET/POST /{project_id}/schema` in `api/projects.py` (handlers
`get_schema`/`set_schema`). Added `SchemaColumn` (with the `text|integer|float|
date|boolean` type validator), `ProjectSchemaWrite` (model_validator enforcing
exactly one PK), and `ProjectSchemaRead` (tolerant `list[dict]`, reflects stored
columns verbatim) to `schemas.py`. The PK/type rules live on the Pydantic request
model, so FastAPI surfaces violations as 422 automatically. Tests in
`tests/test_project_schema.py`; full backend suite green (46 passed).

**Goal:** the column schema is only submitted inside the multipart upload today and
never re-fetched/edited (gap doc §1 backend gaps). Expose it.

**Do (in `api/projects.py`):**
1. `GET /{project_id}/schema` (owner+viewer) → return the stored
   `ProjectSchema.columns` (404 if none). Add a `ProjectSchemaRead` schema.
2. `POST /{project_id}/schema` (owner-only) → upsert
   `ProjectSchema.columns` from body `{columns: [{name, type, is_primary_key}]}`.
   Validate exactly one PK and that types are in the allowed set
   (`text|integer|float|date|boolean`). Do **not** auto-fix — return inline
   validation errors (422). This is the editable counterpart to upload step 2.

**Acceptance:** GET returns the columns saved at upload; POST with two PKs returns
422; POST with valid columns persists and GET reflects it.

---

## B6 — Re-run survival (replay edits) + label_source protection
**Status:** ✅ Done (2026-06-28) · **Prereqs:** B1, B2, B3, B4

**Deviations from plan:**
- `replay_edits(session, project_id, snapshot)` takes a third arg: a
  `MembershipSnapshot` captured by `snapshot_membership(session, project_id)`
  **before** `persist_run_result` wipes the old rows. It's required because the
  edit log stores *old* document/cluster UUIDs that the re-run regenerates — the
  snapshot bridges old doc UUID → `primary_key_value` → new doc, and resolves an
  old cluster UUID to the new cluster holding the **plurality** of its old members
  (against the fresh run's base assignment). Human-created clusters are recreated
  and tracked in a `remap` so reassignments/merges can target them.
- Replay order: **creates** → merges → junk → splits → label actions → doc
  reassignments → confirm. Creates run first (app-only actions with no notebook
  analogue) because later merges/reassignments may target a recreated cluster.
- `split_cluster` and `confirm_run` are logged and **skipped** — neither has an
  app-side artifact yet (no micro-labels / re-cluster manifest, no
  `human_confirmed` field). Revisit when those land.
- Recompute (B2) is async-only; added a sync `_recompute_clusters_sync` in
  `services/replay.py` for the Celery worker, reusing the pure helpers from
  `services/recompute.py`.
- label_source protection: replay runs *after* persist, so a re-applied human
  rename (`hitl_override`) always wins over the run's machine label. No separate
  relabel pass exists today; any future one must skip `hitl_override` clusters.
- Tests: `tests/test_replay.py` (no DB / no ML stack, fake sync session) covers
  the acceptance scenario plus create-from-selection / merge / junk / approve /
  unresolvable-skip and `snapshot_membership`.

**Goal:** re-processing a project (`persist_run_result` wipes & rewrites
clusters/documents in `ml_mapping.py`) must not destroy manual edits. This is the
app analogue of `apply_feedback.apply_run_feedback`.

**Do:**
1. After `persist_run_result` rebuilds rows in `ml_pipeline.run_ml_pipeline`, load
   the project's `ClusterEdit` rows (B1) in `created_at` order and **replay** them
   over the fresh rows, in the order `apply_feedback` uses: merges → junk → splits
   → label actions → doc reassignments → confirm.
   - Map replayed edits onto new rows by stable identity: documents by
     `primary_key_value` (cluster ids are regenerated each run, so match clusters
     by replaying doc membership / by stored `new_label` for renames).
   - Implement this as `src/backend/app/services/replay.py:replay_edits(session,
     project_id)` and call it from `run_ml_pipeline` after persist, before the
     `status=ready` commit. Recompute affected clusters (B2) afterward.
2. Ensure a future LLM relabel pass never clobbers human labels: when persisting /
   relabeling, skip clusters whose `label_source == "hitl_override"` (column
   already exists; renames in B4 set it).

**Acceptance:** seed a project, reassign a doc + rename a cluster, trigger a
re-run, and confirm the doc lands in the right place and the human label survives.

Reference: `src/reviewscope_ml/hitl/apply_feedback.py`.

---

## B7 — Rich hover payload on embeddings (backend half of gap §2)
**Status:** ✅ Done (2026-06-28) · **Prereqs:** none
Implemented as planned: `EmbeddingPoint` gained `snippet`/`primary_key_value`/
`sentiment_score`/`cluster_label` (all optional); `GET /embeddings` now LEFT JOINs
`Cluster` (so noise points get `cluster_label = null`), caps snippet to 120 chars
server-side, and accepts optional `?limit=`. No "stars" field was added — there is
no per-document stars column on `Document` (mean_stars lives only on `Cluster`), so
that part of the goal was dropped. Covered by new `tests/test_embeddings.py`.

**Goal:** the scatter only gets `document_id` today. Provide snippet + sentiment +
label + PK + stars per point so the frontend hover (F1) can be rich, **built only
for displayed points** (perf).

**Do (in `api/projects.py`):**
1. Extend `EmbeddingPoint` in `schemas.py` with optional
   `snippet: str | None`, `primary_key_value: str | None`,
   `sentiment_score: float | None`, `cluster_label: str | None`.
2. Update `GET /{project_id}/embeddings` to join `Document` (text→snippet[:120],
   `primary_key_value`, `sentiment_score`) and the cluster `label`. Keep the
   query a single join; cap snippet length server-side.
3. (Optional, for very large projects) add `?limit=` so the frontend can request
   only the capped set of points.

**Acceptance:** `GET /embeddings` returns the new fields populated; noise points
have `cluster_label = null`.

---

# Group F — Frontend

## F0 — API client + types (`api.ts`)
**Status:** ☐ Not started · **Prereqs:** B3, B4, B5, B7 (to function; can be written speculatively)

**Goal:** expose all new endpoints + fix the frontend `Cluster`/`EmbeddingPoint`
types. (NB: backend `ClusterRead` already has `mean_stars` and `label_source`; the
frontend `Cluster` type just omits them.)

**Do (in `src/frontend/src/api.ts`):**
1. Add to the `Cluster` type: `label_source: string`, `mean_stars: number | null`.
2. Add to `EmbeddingPoint`: `snippet`, `primary_key_value`, `sentiment_score`,
   `cluster_label` (all nullable) — matches B7.
3. Add `ClusterEdit` type matching `ClusterEditRead`.
4. Add API methods:
   `reassignDocument(projectId, documentId, clusterId|null)`,
   `bulkReassign(projectId, documentIds, clusterId|null)`,
   `createCluster(projectId, label)`,
   `mergeClusters(projectId, sourceIds, targetId)`,
   `createClusterFromSelection(projectId, documentIds, label)`,
   `updateCluster(projectId, clusterId, {label?, approve?, markJunk?})`,
   `deleteCluster(projectId, clusterId)`,
   `getSchema(projectId)`, `saveSchema(projectId, columns)`,
   `edits(projectId)`.

**Acceptance:** types compile; methods hit the right paths/verbs. (Pure client
layer — no UI yet.)

---

## F1 — Scatter: real hover (gap §2)
**Status:** ☐ Not started · **Prereqs:** B7, F0

**Goal:** replace the bare-UUID hover with a rich `hovertemplate` in both
`ProjectView.tsx` and `DeckDashboard.tsx`.

**Reference:** the prototype hover string in
`src/reviewscope_ml/hitl/app.py` (~line 286–307): `"<b>{cluster}</b><br>{snippet}
<br><i>{doc_id}</i> · {sentiment}"` with `hovertemplate="%{text}<extra></extra>"`.

**Do:**
1. After F0/B7, build a per-point `text[]` array: cluster label, text snippet,
   primary-key value, sentiment label (derive from `sentiment_score` sign, or show
   the score), star rating if present.
2. Set Plotly `text` to that array and `hovertemplate: "%{text}<extra></extra>"`
   (drop the raw `document_id` text). Build payloads only for the points actually
   plotted (respect the cap from F6).
3. Apply identically in `DeckDashboard.tsx`.

**Acceptance:** hovering a dot shows label + snippet + PK + sentiment, not a UUID,
in both views.

---

## F2 — Scatter: 2D/3D toggle + click→highlight cluster
**Status:** ☐ Not started · **Prereqs:** none (pairs with F0 for types)

**Goal:** `ProjectView.tsx` hardcodes `scatter3d` and the click→cluster
interaction is unwired (only `DeckDashboard` highlights today).

**Do (in `ProjectView.tsx`):**
1. Add a 2D/3D toggle (state). 3D → `scatter3d` with `z`; 2D → `scattergl` with
   `x`/`y` only. Keep the colour-by-cluster mapping.
2. Add a `highlightedClusterId` state. On Plotly `onClick`, read the clicked
   point's cluster and set it; visually emphasise that cluster's card in the right
   panel (scroll-to / outline) and de-emphasise others' points (opacity).
3. Mirror the pattern already in `DeckDashboard.tsx` (`highlightedClusterId`).

**Acceptance:** toggling switches trace type without reload; clicking a point
highlights its cluster card and dims other points.

---

## F3 — Cluster cards & detail: word cloud + star rating + distribution chart
**Status:** ☐ Not started · **Prereqs:** F0

**Goal:** fill the card/detail depth gaps (§1): word cloud (`word_frequencies`
already provided), star rating (`mean_stars`), and a sentiment/star distribution
chart on the detail page.

**Do:**
1. Add a small word-cloud component (term size ∝ frequency) fed by
   `cluster.word_frequencies`. Use on both the cluster card (compact) and
   ClusterDetail (full-width). Pick a lightweight approach (CSS font-size scaling
   or a tiny lib); no heavy dependency.
2. Show `mean_stars` (e.g. ★ 4.2) on the card + detail when non-null (needs F0
   type addition).
3. On `ClusterDetail.tsx`, add a sentiment/star distribution chart (Plotly bar /
   histogram) from the cluster's documents.

**Acceptance:** cards render a word cloud sized by frequency and a star rating when
present; detail shows a distribution chart.

---

## F4 — "Show all documents" paginated table (all schema columns)
**Status:** ☐ Not started · **Prereqs:** B5

**Goal:** the spec's full document table with **all schema columns** doesn't exist
anywhere (§1). Build it, paginated, reusing `GET /documents` (already supports
`limit`/`offset`) and `GET /schema` (B5) for column headers.

**Do:**
1. Add a reusable `DocumentsTable` component: columns come from the project schema
   (`api.getSchema`); each row renders `raw_data[col.name]` (PK + text +
   everything). Server-side pagination via `limit`/`offset` with prev/next.
2. Add a "Show all documents" toggle on `ProjectView.tsx` (project-wide) and use
   the same table in `ClusterDetail.tsx` (cluster-scoped via `cluster_id`),
   replacing its current load-all behaviour.

**Acceptance:** table shows every schema column; pagination fetches pages; works
both project-wide and per-cluster.

---

## F5 — Editing UX: selection toolbar, card actions, edit-mode toggle
**Status:** ☐ Not started · **Prereqs:** F0, B3, B4

**Goal:** the core "edit clusters from the app" UX (gap §4b/§4c), owner-only.

**Do (in `ProjectView.tsx` + `ClusterDetail.tsx`):**
1. **Edit-mode toggle** (owner only, hidden for viewers) so the read-only
   experience stays clean.
2. **Scatter lasso/box select** (Plotly `dragmode: "lasso"/"select"`, capture
   `onSelected` point ids) → a selection toolbar: "N points selected →
   Reassign to… / New cluster…". Wire to `bulkReassign` /
   `createClusterFromSelection`.
3. **Per-cluster card actions:** rename (inline), "merge into…" dropdown
   (`mergeClusters`), mark junk (`deleteCluster`/`updateCluster markJunk`), split
   (call the split path / flag).
4. **Multi-select clusters → "Merge selected"** (N→1) via `mergeClusters`.
5. **ClusterDetail:** row-level "move to cluster…" (single `reassignDocument`) and
   bulk-select rows → reassign (`bulkReassign`).

**Acceptance:** as owner you can lasso points and reassign/create a cluster, rename
a cluster, merge N→1, mark junk, and move single/bulk docs; viewers see no edit
controls.

---

## F6 — Cross-cutting: optimistic updates, invalidation, WebGL cap
**Status:** ☐ Not started · **Prereqs:** F5

**Goal:** make edits feel instant and keep large projects performant (§4d).

**Do:**
1. For every mutation in F5, use react-query optimistic update + rollback, then
   `invalidateQueries` for `["clusters", projectId]`, `["embeddings", projectId]`,
   `["cluster-docs", ...]`, and `["edits", projectId]`.
2. Apply a **~12k-point WebGL cap**: when `embeddings.length > 12000`, sample down
   for display and use `scattergl` (2D) — mirror the cap rationale in the gap doc
   §3 / prototype. Show a "showing N of M points" note.

**Acceptance:** an edit updates the UI before the server responds and rolls back on
error; a >12k-point project renders smoothly with a capped/WebGL scatter.

---

## F7 — Undo / edit-history panel
**Status:** ☐ Not started · **Prereqs:** B1, F0

**Goal:** surface the audit log (`GET /edits`, B1) as a history panel; support undo
where feasible (e.g. re-apply the inverse reassign).

**Do:**
1. Add a collapsible "Edit history" panel on `ProjectView.tsx` listing edits
   newest-first (actor, action, when, subject).
2. Implement undo for reversible actions (single/bulk reassign → move back;
   rename → restore previous label from the edit's `before` payload). Merges/junk
   can be "not undoable" for v1 with a clear note.

**Acceptance:** history lists recent edits; undo on a reassign moves the doc back
and records a new edit.

---

## F8 — Upload Step 2: real schema confirm
**Status:** ☐ Not started · **Prereqs:** none (B5 optional, for server-side validation)

**Goal:** fix the broken type selector in `Dashboard.tsx`'s `UploadModal` (§1).

**Current bugs (`Dashboard.tsx` `detectColumns`):** type default is
`index === 0 ? "text" : "text"` (always `text`); PK detection is just
`id`/first-column.

**Do:**
1. **Real per-column type inference** from sampled rows (parse a handful of CSV/
   JSONL rows): classify each column as integer/float/date/boolean/text.
2. **Better PK detection:** prefer a column with unique, non-null values and an
   id-like name; fall back to first column. Allow the user to change the PK
   (radio), enforcing exactly one.
3. **Submit-time validation with inline errors, no auto-fixing** (spec
   requirement): block submit if no text column, no/duplicate PK, etc.
4. (Optional) split into the spec's two-step modal flow.
5. Reuse the same validation server-side via B5's `POST /schema` if available.

**Acceptance:** numeric/date columns are inferred (not all `text`); the PK is
selectable and validated; invalid schemas show inline errors and block submit.

---

## F9 — Smaller spec fills: owner email, members role-change, models/health
**Status:** ☐ Not started · **Prereqs:** none

**Goal:** quick wins from §1.

**Do:**
1. **`Dashboard.tsx` `ProjectCard`:** render `project.owner_email` on shared
   projects (already on the `Project` type, just unused).
2. **`SettingsView.tsx`:** add a member **role-change** control calling the
   existing `PATCH /members/{uid}` (currently unused). Add `updateMember` to
   `api.ts`.
3. **Surface system info:** call `GET /api/models` (and add `GET /api/health` if
   missing) and show available models / health in Settings or the upload modal.
   Add `models()`/`health()` to `api.ts`.

**Acceptance:** shared-project cards show the owner email; an owner can change a
viewer's role from Settings; available models are visible in the UI.

---

# Group T — Trust, export & scale (lower priority, §5)

These are "worth planning for" items. Each is standalone; build after Groups B/F.

## T1 — Label-provenance badge + confirm/sign-off
**Status:** ☐ Not started · **Prereqs:** F0, B1

Badge on each cluster (LLM `ollama:*` vs `terms_fallback` vs `hitl_override`/
`hitl_approved`) from `label_source` (already on `ClusterRead`; add to frontend
`Cluster` type via F0). Add a per-project "confirmed" state: a `confirm_run`-style
edit (B1) + a project-level flag/badge. Reference `confirm_run` semantics in
`apply_feedback.py`.

## T2 — Export
**Status:** ☐ Not started · **Prereqs:** B5

`GET /{project_id}/export?format=csv|json` → documents with (edited) cluster
assignments + all schema columns. Frontend "Export" button. Optionally scatter
PNG (Plotly `toImage`) and a cluster-summary report.

## T3 — Search / filter / noise triage
**Status:** ☐ Not started · **Prereqs:** B3, F4

Global text search across documents; filter scatter/cluster list by sentiment /
stars / date / schema column; a dedicated **noise triage view** for
`cluster_id IS NULL` docs to reassign in bulk (reuses B3 bulk reassign + F4
table).

## T4 — Re-run / re-cluster controls + append data
**Status:** ☐ Not started · **Prereqs:** B6

In-app controls (# clusters, embedding model, min cluster size) tied to
`GET /api/models`, triggering a re-run (which now survives edits via B6).
"Append data to existing project" is the spec's own future extension.

---

## Suggested build order (matches gap-doc §6)
1. **B1 → B2 → B3 → B4** (backend mutation core), then **B5**, **B7**.
2. **F0**, then **F5 + F6** (editing MVP), **F4**, **F2**.
3. **F1** (real hover), **F3** (word cloud / stars / distribution).
4. **B6** (re-run survival), **F7** (undo/history), **F8/F9** (spec fills).
5. **Group T** (trust + export + scale) last.

> **Open question for the team (from the gap doc):** should app edits also write
> the ML `feedback/` JSONL contract, or only the `cluster_edits` table?
> Recommendation: **DB table for the app (B1), plus a JSONL exporter** so the app
> and notebook tooling stay reconcilable.
