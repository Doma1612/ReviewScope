# ReviewScope — Application Specification

---

## Authentication & Authorization

### Auth
- User registration and login (JWT stored in httpOnly cookie)
- Protected routes — unauthenticated users redirected to login

### Project Permissions
- **Owner:** full access (read, share, delete)
- **Viewer:** read-only access to cluster view and documents
- Project sharing: owner invites collaborators by email and assigns role
- Delete restricted to owner only

---

## Frontend Views

### Login / Register (`/login`, `/register`)
- Email + password form
- Redirect to dashboard on success

---

### Project Dashboard (`/`)
- Grid of project cards (only projects the current user has access to)
  - Project name
  - Owner name (shown when project is shared)
  - Status badge: `Uploading` / `Processing` / `Ready` / `Failed`
  - Document count
  - Created date
  - Permission badge: `Owner` / `Viewer`
  - "Open" button (active only when status is Ready)
- "New Project" button → opens Upload Modal
- Failed cards show last pipeline error message

---

### Upload Modal — Step 1: File
- Project name input
- File drop zone (JSONL or CSV)
- "Next" button → parses file headers, proceeds to Step 2

### Upload Modal — Step 2: Schema Confirmation
- Dataset must already be correctly structured — no column renaming, reordering, or exclusion
- Table of auto-detected columns:
  - Column name (read-only)
  - Data type selector: `text`, `integer`, `float`, `date`, `boolean`
  - Primary key indicator (auto-detected from column name heuristics, read-only)
- Validation runs on submit — errors shown inline, no auto-fixing
- "Upload & Start" button → submits schema + file, closes modal, pipeline starts async

> **Future extensions:** chunk size configuration, append data to existing project

---

### Project Dashboard — Live Status
- Cards for in-progress projects show current pipeline step (e.g. `Clustering… step 4/6`)
- Frontend polls `GET /api/projects/{id}/pipeline/status` every ~3s
- Polling stops automatically on `Ready` or `Failed`

---

### Cluster View (`/projects/:id`)

**Left panel — Scatter Plot**
- Plotly.js 2D / 3D toggle
- Points colored by cluster assignment
- Click a point → highlights corresponding cluster in the right panel

**Right panel — Cluster List (scrollable)**

Per cluster card:
- Cluster label (LLM-generated)
- Document count
- Sentiment score (if sentiment model ran) and/or star rating (if column present in schema)
- Short LLM-generated summary
- Word cloud (term size proportional to frequency within cluster)
- 2–3 sample document snippets
- "View all" button → navigates to Cluster Detail

**Bottom toggle — Show all documents**
- Replaces cluster list panel with a full paginated document table
- All schema columns are shown as table columns

---

### Cluster Detail View (`/projects/:id/clusters/:cid`)
- Back link → Cluster View
- Cluster label + document count
- Sentiment score or star rating distribution chart (if applicable)
- Full LLM-generated summary paragraph
- Word cloud (larger, full-width)
- Top terms list
- Paginated, scrollable document table (all schema columns)

---

### Project Settings (`/projects/:id/settings`)
- Rename project (owner only)
- Share: invite collaborator by email, assign `Viewer` role (owner only)
- Member list with roles and revoke access button
- Danger zone: delete project (owner only, requires confirmation)

---

## Backend API Endpoints

### Auth
| Method | Path | Description |
|---|---|---|
| `POST` | `/api/auth/register` | Create user account |
| `POST` | `/api/auth/login` | Login, returns JWT |
| `POST` | `/api/auth/logout` | Invalidate session |
| `GET` | `/api/auth/me` | Current user info |

### Projects
| Method | Path | Description |
|---|---|---|
| `POST` | `/api/projects` | Create project + upload file + enqueue pipeline (202) |
| `GET` | `/api/projects` | List all projects accessible to current user |
| `GET` | `/api/projects/{id}` | Project metadata and pipeline status |
| `PATCH` | `/api/projects/{id}` | Rename project (owner only) |
| `DELETE` | `/api/projects/{id}` | Delete project (owner only) |

### Project Access / Sharing
| Method | Path | Description |
|---|---|---|
| `GET` | `/api/projects/{id}/members` | List members and roles |
| `POST` | `/api/projects/{id}/members` | Invite by email, assign role |
| `PATCH` | `/api/projects/{id}/members/{uid}` | Change member role |
| `DELETE` | `/api/projects/{id}/members/{uid}` | Revoke access (owner only) |

### Schema
| Method | Path | Description |
|---|---|---|
| `POST` | `/api/projects/{id}/schema` | Submit confirmed column types |
| `GET` | `/api/projects/{id}/schema` | Retrieve stored schema |

### Pipeline
| Method | Path | Description |
|---|---|---|
| `GET` | `/api/projects/{id}/pipeline/status` | Per-step progress and overall status |

### Visualization
| Method | Path | Description |
|---|---|---|
| `GET` | `/api/projects/{id}/embeddings` | UMAP x/y/z coords + cluster_id per document |

### Clusters
| Method | Path | Description |
|---|---|---|
| `GET` | `/api/projects/{id}/clusters` | List: label, size, sentiment, top terms, word frequencies, sample docs |
| `GET` | `/api/projects/{id}/clusters/{cid}` | Detail: full summary, sentiment, top terms, word frequencies |
| `GET` | `/api/projects/{id}/clusters/{cid}/documents` | Paginated documents in cluster |

### Documents
| Method | Path | Description |
|---|---|---|
| `GET` | `/api/projects/{id}/documents` | Paginated all docs, filterable by cluster |
| `GET` | `/api/projects/{id}/documents/{did}` | Single document |

### System
| Method | Path | Description |
|---|---|---|
| `GET` | `/api/health` | Health check |
| `GET` | `/api/models` | Available embedding and LLM models |

---

## Async Upload Flow

```
Step 1  User drops file → auto-detect column headers and types
Step 2  User confirms/adjusts data types per column
        → validation runs on submit
        → errors block submit, no auto-fixing
Step 3  POST /api/projects  (multipart: name + schema + file) → 202 Accepted
Step 4  Modal closes, project card appears with status: Processing
Step 5  Frontend polls GET /api/projects/{id}/pipeline/status every ~3s
Step 6  Card updates per pipeline step
Step 7  Status → Ready: "Open" button activates
        Status → Failed: error message shown on card
```

---

## ML Pipeline (Celery Tasks)

| Step | Task | Description |
|---|---|---|
| 1 | **Ingest** | Parse file against confirmed schema, validate types and primary key uniqueness, reject on error |
| 2 | **Preprocess** | Clean text column, deduplicate by primary key, filter out very short documents |
| 3 | **Embed** | Run sentence-transformers, store vectors in pgvector |
| 4 | **Reduce** | UMAP dimensionality reduction to 2D and 3D coords per document |
| 5 | **Cluster** | HDBSCAN / Agglomerative clustering, assign cluster_id per document |
| 6 | **Sentiment** | BERTopic sentiment model — score per document, aggregate per cluster (skipped if no text column qualifies) |
| 7 | **Label** | Ollama LLM generates label and summary per cluster |
| 8 | **Finalize** | Mark project as Ready, persist all results |

---

## Database Schema (PostgreSQL + pgvector)

### `users`
| Column | Type | Notes |
|---|---|---|
| `id` | UUID | PK |
| `email` | text | unique |
| `password_hash` | text | |
| `created_at` | timestamp | |

### `projects`
| Column | Type | Notes |
|---|---|---|
| `id` | UUID | PK |
| `name` | text | |
| `owner_id` | UUID | FK → users |
| `status` | text | pending / processing / ready / failed |
| `doc_count` | integer | |
| `created_at` | timestamp | |

### `project_members`
| Column | Type | Notes |
|---|---|---|
| `project_id` | UUID | FK → projects |
| `user_id` | UUID | FK → users |
| `role` | text | owner / viewer |

### `project_schema`
| Column | Type | Notes |
|---|---|---|
| `project_id` | UUID | FK → projects |
| `columns` | JSONB | `[{name, type, is_primary_key}]` |

### `documents`
| Column | Type | Notes |
|---|---|---|
| `id` | UUID | PK |
| `project_id` | UUID | FK → projects |
| `primary_key_value` | text | value of the user-defined PK column |
| `text` | text | main text column used for NLP |
| `raw_data` | JSONB | all original columns |
| `cluster_id` | UUID | FK → clusters, nullable |
| `sentiment_score` | float | nullable |

### `embeddings`
| Column | Type | Notes |
|---|---|---|
| `document_id` | UUID | FK → documents |
| `vector` | vector | pgvector |
| `umap_x` | float | |
| `umap_y` | float | |
| `umap_z` | float | nullable |

### `clusters`
| Column | Type | Notes |
|---|---|---|
| `id` | UUID | PK |
| `project_id` | UUID | FK → projects |
| `label` | text | LLM-generated |
| `summary` | text | LLM-generated |
| `top_terms` | JSONB | `[{term, score}]` |
| `word_frequencies` | JSONB | `{term: count}` — drives word cloud |
| `size` | integer | document count |
| `sentiment_avg` | float | nullable |

### `pipeline_jobs`
| Column | Type | Notes |
|---|---|---|
| `id` | UUID | PK |
| `project_id` | UUID | FK → projects |
| `step` | text | ingest / preprocess / embed / … |
| `status` | text | pending / running / done / failed |
| `message` | text | last log line or error |
| `started_at` | timestamp | |
| `finished_at` | timestamp | nullable |

---

## Infrastructure (Docker Compose)

| Service | Image / Tech | Role |
|---|---|---|
| `api` | FastAPI + Python 3.12 | REST API, auth, business logic |
| `worker` | Celery + same image | ML pipeline task execution |
| `db` | PostgreSQL + pgvector | Primary data store |
| `redis` | Redis | Celery broker + result backend |
| `ollama` | Ollama | Local LLM inference for cluster labeling |
| `frontend` | Vite (dev) / Nginx (prod) | React 19 + TypeScript UI |

---

## Tech Stack Summary

| Layer | Technology |
|---|---|
| Frontend | React 19, TypeScript, Vite, TanStack Query, Plotly.js, shadcn/ui |
| Backend | FastAPI, SQLAlchemy 2.0 (async), Alembic |
| Task queue | Celery + Redis |
| Database | PostgreSQL + pgvector |
| NLP / ML | sentence-transformers, BERTopic, HDBSCAN, scikit-learn, UMAP |
| LLM inference | Ollama (local dev), vLLM (university GPU) |
| Auth | JWT (httpOnly cookie) |
| Infra | Docker + Docker Compose |
