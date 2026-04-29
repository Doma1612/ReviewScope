# Tech Stack — Proposal for Discussion

**Project:** ReviewScope  
**Status:** Draft — open for team discussion

---

## Context

ReviewScope is a qualitative text analysis platform. The core pipeline takes a corpus of documents, generates embeddings, reduces dimensionality, clusters the results, and assigns human-readable labels to clusters using an LLM. On top of that sits an application layer: cluster visualization, a RAG-based chat interface, and document management.

Two constraints should shape the tech decisions:

1. **Standalone first.** The application should work entirely on its own — own auth, own storage, own frontend — so it can be developed, tested, and demoed without external dependencies.
2. **Externally pluggable.** Ideally, the platform is pluggable as a sidecar into an existing qualitative analysis tool — for example [DATS](https://github.com/uhh-lt/dats) (Discourse Analysis Tool Suite) by UHH's Language Technology group. In sidecar mode, ReviewScope would read documents and embeddings from the host platform and write cluster annotations back.

The second point is a nice-to-have for the course scope but a strong framing if we want to position the project as a potential open-source contribution to an existing ecosystem.

---

## Proposed Architecture — Ports & Adapters

Rather than microservices (operationally expensive for a project of this size and timeline) or a tightly coupled monolith (hard to extend or plug into DATS later), the proposal is a **modular monolith with hexagonal architecture** (also called ports & adapters).

The core idea: domain logic only ever talks to abstract interfaces (ports). Concrete implementations (adapters) are swapped via environment variables at startup. The clustering pipeline, RAG engine, and API layer would be identical in standalone and DATS-integrated modes — only the adapters at the edges change.

```
domain logic
    ↕  port (abstract interface)
adapter (pgvector | weaviate | dats | local | ...)
```

Every swappable concern gets a port:

| Port | Standalone adapter | External platform adapter (e.g. DATS) |
|---|---|---|
| Document source | Local Postgres | Host platform REST API |
| Embeddings | sentence-transformers | Reused from host platform |
| Cluster annotations | Local DB | Host platform tags/codes API |
| Auth | Internal JWT | Host platform token passthrough |
| Vector store | pgvector | pgvector (unchanged) |

**Open question for the team:** Does this level of abstraction feel like the right investment for the project scope, or is it over-engineering given the timeline?

---

## Backend — FastAPI (Python)

**Proposal:** FastAPI with Python 3.12.

Python is the natural fit for an NLP project — the ML ecosystem lives there. Within Python web frameworks, FastAPI seems like the right choice:

- **Async-native.** Embedding generation and clustering are long-running. FastAPI handles requests asynchronously while jobs run in the background.
- **Pydantic v2.** Data structures are validated and typed end-to-end. OpenAPI schema is generated automatically — useful for the frontend and for DATS integration.
- **Dependency injection.** FastAPI's `Depends()` system is a clean way to wire the ports & adapters pattern without a heavy IoC container.

Flask would need significant scaffolding to reach the same level of async support and typing. Django is the wrong shape for an ML pipeline service — it assumes a different kind of application.

**ORM:** SQLAlchemy 2.0 (async) with Alembic for migrations.

---

## Task Queue — Celery + Redis

The NLP pipeline is not a request-response operation. Embedding a corpus, running UMAP, fitting HDBSCAN, and labeling clusters with an LLM can take minutes. Jobs need to run in the background, support retries, and report progress back to the frontend.

**Proposal:** Celery with Redis as the broker and result backend.

Celery handles task prioritization, progress callbacks, failure retries, and chaining pipeline steps. Redis doubles as a cache layer.

ARQ (async Redis queue) is a lighter alternative worth considering — less overhead, async-native. The tradeoff is less tooling around task introspection, which matters if we want to show pipeline progress in the UI.

**Open question for the team:** Celery vs ARQ — does the added complexity of Celery pay off, or is ARQ sufficient?

---

## Database — PostgreSQL + pgvector

Two things need storing: relational data (projects, documents, clusters, users) and embedding vectors (for similarity search and RAG).

**Proposal:** PostgreSQL with the [pgvector](https://github.com/pgvector/pgvector) extension.

pgvector adds vector storage and approximate nearest-neighbor search to Postgres, keeping everything in one database. For a research-scale project this is more than sufficient, and it avoids running a second database service during development.

The vector store port means switching to Qdrant or Weaviate later is a configuration change, not a refactor. But starting with pgvector keeps the local setup simple.

**Alternatives worth discussing:**
- **Qdrant** — purpose-built vector DB, excellent performance, easy to self-host. Real option if we expect large corpora.
- **Weaviate** — more feature-rich, has built-in RAG primitives. More complex to operate.

---

## Frontend — React + TypeScript (Vite)

**Proposal:** React 19 with TypeScript, built with Vite.

The frontend has some specific requirements that drove this thinking:

- 2D/3D cluster visualization (UMAP scatterplots, interactive exploration)
- RAG chat interface with streaming responses
- Document upload and project management
- A potential integration path into external platforms such as DATS — which is itself a TypeScript application

React has the best ecosystem for all of these, particularly for scientific visualization. Aligning on TypeScript also makes it easier to share components or embed the ReviewScope UI into a host platform later.

**Vite** over Create React App (unmaintained) or Next.js (SSR complexity we don't need).

**Proposed libraries:**

| Need | Library | Notes |
|---|---|---|
| Server state / data fetching | TanStack Query | Cache management, background refetching |
| Cluster visualization | Plotly.js | 2D and 3D scatter, handles large point clouds |
| UI components | shadcn/ui + Tailwind | Composable, accessible, not a black-box library |
| API client | auto-generated from OpenAPI | FastAPI generates the spec; `openapi-ts` generates a typed client |

**Alternatives worth discussing:**
- **Vue 3** — comparable DX, good TypeScript support, slightly thinner ecosystem for scientific viz.
- **Svelte / SvelteKit** — much lighter bundle, but React's maturity is an asset as the app grows.

**Open question for the team:** Is anyone strongly opposed to React, or does someone have a strong case for Vue?

---

## NLP / ML Stack

> **Note:** This section is an outline for future discussion — not part of the current tech selection. Decisions here will follow once the core architecture is agreed on.

- **Preprocessing**
  - Junk removal, deduplication
  - Stemming / lemmatization (to investigate: does it help or hurt embedding quality?)
  - Language detection for multilingual corpora

- **Embeddings**
  - Candidates: sentence-transformers (local), OpenAI API, instructor-xl, multilingual-e5
  - Key question: which model performs best on the target domain (e.g. social science / news / academic text)?
  - University GPU compute worth investigating for local inference

- **Dimensionality Reduction**
  - UMAP (primary candidate — fast, preserves global structure, suitable as clustering input)
  - t-SNE (visualization only — distortions make it unreliable for clustering)
  - PCA (as pre-reduction step before UMAP on very high-dimensional embeddings)

- **Clustering**
  - HDBSCAN (primary candidate — no fixed k, handles noise, variable density)
  - DBSCAN (simpler, less adaptive)
  - KMeans (for cases where k is known)
  - **Hierarchical clustering** (e.g. agglomerative) — worth investigating for producing topic trees rather than flat clusters; could be complementary to HDBSCAN rather than a replacement
  - Reproducibility across runs is an open problem for all density-based methods

- **Cluster Labeling & Summarization (LLM)**
  - Preference: open-source models, self-hosted — proprietary APIs (Anthropic, OpenAI) as fallback only
  - Serving: [vLLM](https://github.com/vllm-project/vllm) on university GPU compute (high-throughput inference, OpenAI-compatible API)
  - Local dev: [Ollama](https://ollama.com) (runs the same model family on CPU/consumer GPU, no infra setup)
  - Model candidates: Llama 3.1 / 3.3, Mistral, Qwen 2.5 — to be evaluated on labeling quality vs. size tradeoff
  - All LLM calls go through a labeler port — switching from vLLM to Anthropic or OpenAI is a config change, not a code change
  - Input context strategy matters: centroid-based sampling vs. TF-IDF representative docs vs. metadata-enriched prompts — to be investigated

- **Cluster Summarization**
  - See section below

---

## Cluster Summarization — Pipeline vs. On-the-fly

Summarization splits into two distinct concerns that warrant different treatment.

**What should be pre-computed in the pipeline:**

The visualization needs something to show immediately when a user opens a cluster. Generating that on demand would mean waiting for a live LLM call on every page load, which is not acceptable UX. The following should be computed once during the pipeline run and stored:

- **Cluster label** — short phrase identifying the topic (e.g. "Climate Policy")
- **TF-IDF keywords** — top-n terms characteristic of the cluster
- **Representative documents** — centroid-based or TF-IDF sampling, stored as doc IDs
- **Base summary** — a short paragraph describing the cluster's content, generated from the representative documents. Stored as nullable — can be generated lazily on first access if skipped during the initial run.

The model version and prompt hash used to generate the base summary should be stored alongside it for reproducibility.

**What should be generated on-the-fly (RAG / application layer):**

Anything user- or query-specific stays in the application layer:

- User questions about a specific cluster ("what does this cluster say about X?")
- Cross-cluster comparisons
- Summaries from a particular analytical angle (sentiment, actors, timeline)
- Follow-up chat in the RAG interface

The pre-computed base summary and representative documents serve as context injected into the RAG prompt, so the LLM does not start cold on every user query.

**In practice:**

```
DB stores per cluster:
  label            ← pipeline
  keywords[]       ← pipeline
  rep_doc_ids[]    ← pipeline
  base_summary     ← pipeline (nullable, generated lazily if skipped)
  summary_model    ← pipeline (model + prompt hash for reproducibility)

RAG layer generates:
  user-specific queries   ← on-the-fly, uses base_summary + rep_docs as context
```

---

## External Platform Integration

The ports & adapters architecture is designed so ReviewScope can attach to any qualitative analysis platform that exposes a REST API — the primary candidate being [DATS](https://github.com/uhh-lt/dats) by UHH, but the pattern is not specific to it.

A platform like DATS already handles document ingestion, preprocessing, embedding generation, and annotation. In sidecar mode, ReviewScope would not duplicate any of this.

The proposed flow:

1. Documents and embeddings are fetched from the host platform via its REST API.
2. The clustering pipeline runs on those embeddings.
3. Cluster assignments and labels are pushed back into the host platform as tags or codes.
4. Auth is delegated to the host platform — ReviewScope validates its session tokens.

One prerequisite for DATS specifically: its REST API would need to expose a batch embedding fetch endpoint. If it does not, that could be a contribution we make to the project as part of the course — worth checking the repo before committing to this integration path.

Domain models would need to be compatible with the host platform's document and annotation schema from the start, not retrofitted later.

---

## Deployment

Local development would run with Docker Compose:

```
postgres     — relational data + pgvector
redis        — task broker + cache
api          — FastAPI application
worker       — Celery worker (runs pipeline jobs)
frontend     — React dev server (or nginx in production)
```

Two compose files: one for standalone mode, one override for DATS sidecar mode (attaches to DATS's Docker network). Same codebase, different environment variables.

University GPU/CPU compute for embedding generation is worth investigating — sentence-transformers benefits from a GPU for large corpora.

---

## What This Proposal Explicitly Avoids (and Why)

| Excluded | Reason |
|---|---|
| Microservices from day one | Too much infra overhead for the timeline; clean module boundaries make extraction possible later |
| Separate vector DB (Qdrant/Weaviate) from day one | pgvector covers the scale; switching is a config change |
| Next.js | SSR complexity the app does not need |
| Django | Wrong shape for an ML pipeline service |
| Flask | Too much scaffolding to reach parity with FastAPI on async and typing |

---

## Open Questions for the Team

1. **Architecture depth:** Is the ports & adapters pattern the right investment, or does it add too much abstraction for a single-semester project?
2. **Celery vs ARQ:** Is Celery's task introspection worth the added complexity?
3. **Vector DB:** Start with pgvector or go straight to Qdrant if we expect larger corpora?
4. **Frontend framework:** Any strong opinions on Vue over React?
5. **External platform integration scope:** Is the sidecar integration (e.g. into DATS) a goal for the course, or a post-course contribution?
6. **LLM model choice:** Which open-source model (Llama 3.x, Mistral, Qwen 2.5) gives the best labeling quality at a reasonable size? Is university GPU compute available for vLLM?
7. **Reproducibility:** How strictly do we need to handle non-deterministic clustering across runs?
8. **NLP/ML stack:** To be discussed separately — see outline in the NLP / ML Stack section.
