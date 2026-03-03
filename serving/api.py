"""
RAG Serving API — Phase 3.

Endpoints:
  POST /query    → { answer, trace_id, version, sources }
  POST /feedback → { status }
  GET  /health   → { status, active_version, ab_test, llm_model, langfuse_enabled }
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ingestion.embedders import OpenAIEmbedder
from monitoring.feedback import post_feedback
from monitoring.instrumentation import LangfuseClient
from serving.chain import run_chain
from serving.retriever import Retriever
from serving.versioning import pick_version
from shared.backends.chroma_store import ChromaVectorStore
from shared.config_loader import load_config
from shared.logging_config import configure_logging

configure_logging()
logger = logging.getLogger("llmops.api")

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


class QueryResponse(BaseModel):
    answer: str
    trace_id: str
    version: str
    sources: List[Dict[str, Any]]


class FeedbackRequest(BaseModel):
    trace_id: str
    rating: int  # 1 = thumbs up, 0 = thumbs down


# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------

_store_cache: Dict[str, ChromaVectorStore] = {}


def _get_store(collection_name: str, persist_dir: str) -> ChromaVectorStore:
    """Return a cached ChromaVectorStore for the given collection name."""
    if collection_name not in _store_cache:
        store = ChromaVectorStore(persist_directory=persist_dir)
        store.create_collection(collection_name, dimension=0)
        _store_cache[collection_name] = store
    return _store_cache[collection_name]


@asynccontextmanager
async def lifespan(app: FastAPI):
    serving_cfg = load_config("config/serving.yaml")
    monitoring_cfg = load_config("config/monitoring.yaml")
    pipeline_cfg = load_config("config/pipeline.yaml")

    app.state.serving_cfg = serving_cfg
    app.state.pipeline_cfg = pipeline_cfg
    app.state.lf = LangfuseClient.from_config(monitoring_cfg)
    app.state.embedder = OpenAIEmbedder(
        model=pipeline_cfg["pipeline"]["embedding_model"]
    )

    logger.info(
        "Serving API ready — active_version=%s langfuse=%s",
        serving_cfg["vector_store"]["active_version"],
        app.state.lf.enabled,
    )
    yield
    app.state.lf.flush()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="RAG Serving API", version="0.3.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
def health():
    cfg = app.state.serving_cfg
    vs = cfg["vector_store"]
    return {
        "status": "ok",
        "active_version": vs["active_version"],
        "ab_test": vs.get("ab_test", {}),
        "llm_model": cfg["llm"]["model"],
        "langfuse_enabled": app.state.lf.enabled,
    }


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    serving_cfg: dict = app.state.serving_cfg
    lf: LangfuseClient = app.state.lf

    version = pick_version(serving_cfg)
    vs_cfg = serving_cfg["vector_store"]
    collection_name = f"{vs_cfg['collection_prefix']}_{version}"
    llm_cfg = serving_cfg["llm"]

    with lf.trace_context(
        "rag_query",
        {"query": request.query, "version": version, "top_k": request.top_k},
    ) as (trace_id, root_obs):
        try:
            store = _get_store(collection_name, persist_dir=".chroma")
            retriever = Retriever(
                embedder=app.state.embedder, store=store, version=version
            )

            with lf.span_context(
                "retrieve",
                {"query": request.query, "top_k": request.top_k},
            ) as span:
                result = retriever.retrieve(request.query, top_k=request.top_k)
                lf.update(span, {"num_hits": len(result.items)})

            answer = run_chain(
                query=request.query,
                items=result.items,
                model=llm_cfg["model"],
                temperature=llm_cfg["temperature"],
            )
        except Exception as exc:
            lf.update(root_obs, {"error": str(exc)})
            logger.exception("Query failed")
            raise HTTPException(status_code=500, detail=str(exc))

        lf.update(root_obs, {"answer": answer})

    logger.info(
        "query version=%s top_k=%s trace_id=%s",
        version,
        request.top_k,
        trace_id,
    )

    return QueryResponse(
        answer=answer,
        trace_id=trace_id,
        version=version,
        sources=result.items,
    )


@app.post("/feedback")
def feedback(request: FeedbackRequest):
    post_feedback(app.state.lf, request.trace_id, request.rating)
    return {"status": "ok"}
