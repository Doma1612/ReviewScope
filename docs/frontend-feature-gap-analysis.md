# ReviewScope — Frontend Feature Gap Analysis & Action Items

_Scope: what the spec (`application-spec.md`) asks for, what the React frontend actually
does today, what the ML/HITL prototype shows us, and the new requirement to
**reassign data points / merge clusters / create new clusters** from the app._

Legend: ✅ done · 🟡 partial · ❌ missing · ☐ action item

---

## 1. Spec vs. what is actually built (gaps)

- 🟡 **Dashboard cards** — show status, role, doc count, date, error.
  - ☐ Render **owner email on shared projects** (`owner_email` is in the API type but `ProjectCard` never displays it).
- 🟡 **Upload Step 2 (schema confirm)** — type selector renders but is broken/fake.
  - ☐ Fix the **type default bug** (`index === 0 ? "text" : "text"` always yields `text`).
  - ☐ Add **real type inference** per column (integer/float/date/boolean/text).
  - ☐ Add **PK detection** beyond `id`/first-column heuristic.
  - ☐ Add **submit-time validation** with inline errors, no auto-fixing (spec requirement).
  - ☐ (Optional) split into the spec's **two-step** modal flow.
- 🟡 **Cluster View — scatter** (`ProjectView.tsx`)
  - ☐ Add **2D/3D toggle** (currently hardcoded `scatter3d`).
  - ☐ Wire **click point → highlight its cluster** in the right panel (unwired today; only `DeckDashboard` highlights).
  - ☐ **Fix hover to show real information** (see §2 — currently only the raw `document_id` is passed as hover text).
- 🟡 **Cluster View — cluster cards** — label, count, sentiment, summary, top terms, samples ✅.
  - ☐ Add **word cloud** (term size ∝ frequency; `word_frequencies` already in API).
  - ☐ Show **star rating** (`mean_stars` exists in DB/`Cluster` model but isn't in the API response type or card).
- ❌ **Cluster View — "Show all documents" toggle**
  - ☐ Build the **full paginated document table** with **all schema columns** (does not exist anywhere).
- 🟡 **Cluster Detail**
  - ☐ Add **sentiment/star distribution chart**.
  - ☐ Add **full-width word cloud**.
  - ☐ Add **pagination** (currently loads all docs at once).
  - ☐ Show **all schema columns** in the table (only PK + text today).
- 🟡 **Project Settings**
  - ☐ Add **member role-change UI** (`PATCH /members/{uid}` exists but is unused).
- 🟡 **System endpoints** (`/api/models`, `/api/health`) exist but aren't surfaced.
  - ☐ Surface **available models** / health somewhere (settings or upload).

### Backend endpoint gaps vs. spec
- ☐ Add **`GET/POST /api/projects/{id}/schema`** (missing from `projects.py`; schema is only submitted inside the multipart upload, never re-fetched/edited).
- ☑ Everything else in the spec's endpoint table exists (projects, clusters, documents, members, embeddings, pipeline status).

**Bottom line:** the read happy-path is wired end-to-end, but the spec's depth is stubbed
(word clouds, 2D/3D toggle, documents table, distribution charts, real schema confirm,
point→cluster interaction, real hover) and **no view can mutate results yet.**

---

## 2. Scatter hover must show real information

- **Today:** `ProjectView` passes only `text: embeddings.data?.map(p => p.document_id)` → hovering a dot shows a bare UUID.
- **Reference:** the HITL prototype (`hitl/app.py`) already builds a rich hover string per point: **cluster name, text snippet (~120 chars), doc id, sentiment label**.

Action items:
- ☐ Build a **per-point hover payload**: cluster label, text snippet, primary-key value, sentiment label/score, star rating.
- ☐ Use a Plotly **`hovertemplate`** with `text` per point (not the raw id).
- ☐ Ensure the embeddings/clusters queries provide the **snippet + sentiment + label** the hover needs (extend `EmbeddingPoint` or join cluster/document data client-side).
- ☐ Build hover payloads **only for displayed points** (perf — see point-cap rule in §3).
- ☐ Apply the same rich hover in **`DeckDashboard`**.

---

## 3. ML / HITL prototype — what to reuse

The Streamlit HITL app (`src/reviewscope_ml/hitl/app.py`) is the design reference for the
editing feature. It is **decoupled by contract**: the GUI only appends `FeedbackRecord`
rows (`hitl/feedback.py`), and `hitl/apply_feedback.py` defines replay semantics. Reuse
the semantics, not the Streamlit code.

Actions the prototype already proves out (carry these into the app):
- ☐ `reassign_doc` — move **one document** to a target cluster (or `-1` = noise).
- ☐ `merge_clusters` — move **all docs** of a source cluster → target; recompute terms/word-freqs.
- ☐ `mark_junk` — cluster's docs → noise; cluster removed.
- ☐ `split_cluster` — promote micro-clusters (two-stage) or flag subset for re-clustering (flat).
- ☐ `rename_label` / `approve_label` — override label or stamp human approval.
- ☐ `confirm_run` — record reviewer sign-off.

Design lessons to honor:
- ☐ **Edits are an audit log, not destructive writes** (append + derive, don't overwrite).
- ☐ **Recompute aggregates** after structural edits (size, top terms, word freqs, sentiment, stars) — don't just patch.
- ☐ **Cap the scatter** (~12k points, WebGL `Scattergl`, two traces) for large/sentence-level runs.
- ☐ Reuse the prototype's **multi-select focus → detail view** and **merge N→1** UX.

> The ML package edits **file artifacts**; the backend app stores results in **Postgres**
> (`clusters`, `documents`). Semantics transfer; the app implementation is a new service
> layer over SQLAlchemy (§4).

---

## 4. Requested feature: edit clusters from the app

> "assign data points or clusters to new clusters / merge them — single data point or
> whole cluster, both possible."

### 4a. Backend (do first — frontend depends on it)
- ☐ `PATCH /projects/{id}/documents/{did}` — reassign one doc `{ cluster_id }` (nullable = noise).
- ☐ `POST /projects/{id}/documents/reassign` — bulk reassign `{ document_ids: [...], cluster_id }`.
- ☐ `POST /projects/{id}/clusters` — create a new (empty/named) cluster.
- ☐ `POST /projects/{id}/clusters/merge` — `{ source_ids: [...], target_id }`.
- ☐ `POST /projects/{id}/clusters/from-selection` — create a cluster **from a set of selected points** (lasso → new cluster).
- ☐ `PATCH /projects/{id}/clusters/{cid}` — rename label / approve / mark junk.
- ☐ `DELETE /projects/{id}/clusters/{cid}` — junk: docs → noise, cluster removed.
- ☐ **Recompute service** — after any structural edit recompute `size`, `top_terms`, `word_frequencies`, `sentiment_avg`, `mean_stars`; reuse `reviewscope_ml/represent/` (`ctfidf_terms`, `word_frequencies`).
- ☐ **Edit/audit table** (`cluster_edits`) mirroring `FeedbackRecord`: who/when/action/before-after.
- ☐ **Owner-only guard** on all mutations (viewers stay read-only per spec).
- ☐ **Re-run survival** — replay stored edits when a project is re-processed (the `apply_feedback` analogue) so re-runs don't wipe manual work.
- ☐ Set `label_source = "hitl_override"` on human renames so a future LLM relabel pass won't clobber them (column already exists).

### 4b. Frontend — Cluster View (`ProjectView`)
- ☐ Add Plotly **lasso/box selection** on the scatter → selection toolbar ("N points selected → Reassign / New cluster").
- ☐ **Click point → highlight + select** its cluster.
- ☐ **Per-cluster card actions** (owner only): rename, merge-into dropdown, mark junk, split.
- ☐ **Multi-select clusters → "Merge selected"** (N→1).
- ☐ Add an **edit-mode toggle** so the viewer/read-only experience stays clean.

### 4c. Frontend — Cluster Detail
- ☐ Row-level **"move to cluster…"** on the document table (single-doc reassign).
- ☐ **Bulk-select rows → reassign**.

### 4d. Cross-cutting frontend
- ☐ **Optimistic updates + invalidate** `clusters`/`embeddings`/`documents` queries on mutation.
- ☐ **Undo / edit-history** panel driven by the audit table.
- ☐ Apply the **12k-point WebGL cap** for large projects.

### 4e. API client (`api.ts`)
- ☐ Add `reassignDocument`, `bulkReassign`, `mergeClusters`, `createCluster`, `createClusterFromSelection`, `renameCluster`, `markJunk`, `listEdits`.
- ☐ Add `mean_stars`, `label_source` (and a `status`) to the `Cluster` type.
- ☐ Extend `EmbeddingPoint` (or add a hover endpoint) with snippet/sentiment/label for hover (§2).

---

## 5. Thoughtful user features worth planning for

**Search / filter / navigate**
- ☐ Global **text search** across all documents (prototype has within-cluster search).
- ☐ **Filter** scatter/cluster list by sentiment, star rating, date, or any schema column.
- ☐ **Cross-cluster compare** (select 2–3 clusters side by side).

**Trust the analysis**
- ☐ **Label-provenance badge** (LLM vs. term-fallback vs. human-approved) — data in `label_source`.
- ☐ **Human sign-off / "confirmed" state** per project (the `confirm_run` analogue).
- ☐ **Re-run / re-cluster controls** in-app (# clusters, embedding model, min cluster size) tied to `/api/models`.

**Get data out**
- ☐ **Export** CSV/JSON of docs with (edited) cluster assignments.
- ☐ **Export** scatter as PNG and a cluster-summary report (PDF) — usually the real deliverable.
- ☐ **Shareable read-only link** / embed.

**Scale & quality of life**
- ☐ **Pagination/virtualization** everywhere docs are listed (Cluster Detail loads all docs today).
- ☐ **Noise/outlier triage view** — dedicated bucket for `cluster_id = null` docs to reassign.
- ☐ **Append data to an existing project** (spec's own "future extension").
- ☐ **Annotations / notes** per cluster or document.
- ☐ Surface **per-step pipeline errors** richly (`pipeline_jobs.message` is there).

---

## 6. Suggested sequencing
1. ☐ **Backend mutation layer** (§4a): reassign-doc + merge endpoints, recompute service, edit-audit table, owner-only guard.
2. ☐ **Frontend editing MVP** (§4b/4c): point selection + reassign, per-card merge/rename/junk, query invalidation.
3. ☐ **Real hover** (§2) + spec-gap fills that make editing usable: word cloud, 2D/3D toggle, "show all documents" table, point→cluster highlight.
4. ☐ **Trust + export** (§5): provenance badge, confirm/sign-off, CSV/report export.
5. ☐ **Re-run survival + append data**: replay stored edits across re-processing.

**Open question for the team:** should app edits write back to the ML `feedback/` JSONL
contract (one shared audit format with the notebook tooling), or use a dedicated
`cluster_edits` table? Recommendation: **DB table for the app, with a JSONL exporter** so
the two worlds stay reconcilable.
