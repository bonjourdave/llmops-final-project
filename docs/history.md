# Development History: From Prototype to Production

This document describes how this system was built — what the initial prototype was designed to do, what it proved, and how it evolved into the production stack described in [architecture.md](architecture.md).

---

## Part 1: Local Prototype

### Intent

The prototype was built to answer a focused question before committing to cloud infrastructure: does semantic search over Netflix titles work well enough to build a product on?

Rather than designing for scale upfront, the prototype was intentionally local-first. Everything ran on disk and in memory — no cloud accounts, no managed services, no deployment pipeline. This made it fast to iterate: change an embedding model, rebuild the index, re-run evaluation, compare results. The full cycle took minutes, not hours.

By keeping the stack minimal, the prototype validated the parts that actually needed validating: the quality of the retrieval signal, the usefulness of the abstraction patterns, and whether Langfuse tracing gave enough visibility to make A/B decisions. Once those were confirmed, the path to production was clear.

### What Was Built

**Data and domain.** The source is the public Netflix titles dataset (~8,800 rows). Each row is converted into a single composite text string — `title | type | genre tags | description` — which is what gets embedded and indexed. This composite field construction was deliberately simple and proved effective; it carried forward unchanged into production.

**Core abstractions.** Two abstract interfaces were defined from the start:

- `EmbeddingProvider` — how text becomes a vector. Two implementations: a local SentenceTransformer (offline, no API key) and an OpenAI client (already implemented, not used in the default A/B setup).
- `VectorStoreProvider` — how vectors are stored and searched. One implementation: FAISS, running on disk.

A `Retriever` class composed one of each to produce a single `retrieve(query)` operation. This pattern — define the interface, vary the implementation via config — was the right call from day one and carried forward intact.

**Two FAISS variants for A/B testing.** The prototype ran two retrieval variants in parallel:

| Variant | Index type | Embedding model | Dimensions |
|---|---|---|---|
| v1 | Flat (exact search) | all-MiniLM-L6-v2 | 384 |
| v2 | IVF (approximate) | all-mpnet-base-v2 | 768 |

This was real A/B infrastructure: probabilistic request routing, per-variant response metadata, and an evaluation script that compared average similarity scores across five fixed test queries. Running two variants against real data before production gave confidence in the mechanics before Zilliz was introduced.

**API service.** A FastAPI application exposed two endpoints: `GET /health` (system status, variant config, Langfuse reachability) and `POST /search` (query → top-k results with similarity scores, variant tag, and trace ID). The health endpoint's startup validation — rejecting a mismatched embedding dimension between model and index — was a practical safeguard that also carried forward.

**Langfuse tracing.** Every search request generated a Langfuse trace containing the query, variant used, latency, and score statistics. The health endpoint reported whether Langfuse was reachable; the service started successfully either way. This additive-not-required pattern for monitoring was established in the prototype and kept.

**Configuration.** A single `config.yaml` controlled embedding provider selection, vector store type, A/B split, data paths, and retrieval top-k. No secrets in config files — API keys via environment variables only. This principle held.

### What the Prototype Proved

- Semantic search over the composite Netflix field produces useful, meaningful results
- The pluggable interface pattern (embedder + vector store + retriever) works at this scale and is worth keeping
- Langfuse tracing is lightweight enough to wire in from the start without affecting latency
- A/B mechanics (routing, tagging, evaluation) work correctly with FAISS before introducing a managed vector store
- The field construction and data loading logic is stable and reusable

---

## Part 2: Evolution to Production

### Why Evolve

The prototype answered the question it was built to answer. Moving to production required addressing what local-first deliberately deferred: stateless containers (no disk-based indices), a managed vector store (versioned collections, multi-client access), a generation step (not just retrieval), a chat UI, and a deployment pipeline that makes all of this reproducible.

The evolution was additive. Most of the prototype's good decisions stayed. What changed were the infrastructure choices — and those changes were made one layer at a time.

### Phase 1 — Structural Refactor

The codebase was reorganised into the target folder layout before any new features were added. `shared/`, `ingestion/`, `serving/`, `monitoring/`, `ui/`, and `config/` were created as distinct domains with explicit contracts between them. The monolithic `configs/config.yaml` was split into `config/pipeline.yaml`, `config/serving.yaml`, and `config/monitoring.yaml` to reflect that ingestion and serving have different configuration lifecycles. The abstract interfaces from the prototype were moved into `shared/` as the connective tissue for everything that followed.

### Phase 2 — Ingestion Pipeline

The one-shot build script became a Prefect DAG: load → chunk → embed → write → update `versions.json`. Chunking with configurable overlap was introduced (the prototype embedded full rows; the production pipeline splits into 512-token chunks). OpenAI's `text-embedding-3-small` replaced the local SentenceTransformer models — a deliberate tradeoff for stateless containers at the cost of per-request API spend. The version manifest (`versions.json`) was introduced to track what produced each collection: model, chunk parameters, doc count, and timestamp.

### Phase 3 — Serving Layer and RAG Chain

A generation step was added via LangChain: retrieved chunks are assembled into a context block, passed to `gpt-4o-mini` with a system prompt, and an answer is returned alongside a `trace_id`. The API contract changed: `/search` became `/query` (returns `{ answer, trace_id }`), and a `/feedback` endpoint was added. The feedback endpoint posts a thumbs-up or thumbs-down score to the Langfuse trace identified by `trace_id`, closing the loop between user satisfaction and retrieval version performance.

### Phase 4 — Chainlit UI

A Chainlit chat interface was added as a thin client over the serving API. The UI handles session history, sends queries to `POST /query`, displays answers, and fires `POST /feedback` when the user clicks a rating button. The UI has no business logic; it delegates everything to the serving layer.

### Phase 5 — Docker and Local Compose

All three services were containerised as self-contained multi-stage Dockerfiles. A Docker Compose file brought serving and UI up together for local end-to-end testing, with ingestion under a profile flag so it only runs on demand. This validated the full container stack before touching cloud infrastructure.

### Phase 6 — Zilliz Cloud

The FAISS backend was replaced by Zilliz Cloud (Milvus-compatible) for production. The switch required one new file — `shared/backends/zilliz_store.py` — and a config change. No ingestion or serving code changed. Named collections (`docs_v1`, `docs_v2`, etc.) replaced versioned FAISS index files on disk; the A/B mechanics worked identically against the new backend. FAISS and Chroma remained available as local dev backends via the same config toggle.

### Phase 7 — GCP Cloud Run

The containerised services were deployed to GCP Cloud Run. Serving and UI run as always-on services (scaling to zero when idle). Ingestion runs as a Cloud Run Job triggered on demand. Secrets moved from `.env` files to GCP Secret Manager, injected at runtime via Cloud Run's secret reference mechanism. Each service got a dedicated least-privilege service account. Images are tagged with git SHA and pushed to Artifact Registry.

The ARM64/AMD64 cross-compilation challenge was resolved here: building on an aarch64 development machine for a Cloud Run target (linux/amd64) required switching from `docker compose build` to `docker buildx build --platform linux/amd64`, which produces an authoritative amd64 manifest regardless of the host architecture.

### Phase 8 — GitHub Actions CI

The `make push` workflow was automated as a GitHub Actions pipeline. On every merge to `main`, unit tests run first; if they pass, all three service images are built for `linux/amd64` and pushed to Artifact Registry tagged with the commit SHA. GCP authentication uses Workload Identity Federation — no long-lived service account keys stored as secrets. A dedicated CI service account (`rag-ci-sa`) holds only Artifact Registry write access; it cannot read application secrets or trigger deployments. Human promotion to Cloud Run remains a manual step after CI passes.
