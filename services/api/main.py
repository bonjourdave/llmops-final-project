from __future__ import annotations

import os
import random
import sys
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

from services.api.settings import get_settings
from services.api.logging_config import configure_logging

load_dotenv()

# -----------------------------
# Path setup (project root import)
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.core.local_embedding_provider import LocalSentenceTransformerProvider
from src.core.faiss_store import FAISSVectorStore
from src.core.retriever import Retriever

# -----------------------------
# Langfuse (safe + matches your installed SDK)
# -----------------------------
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

LANGFUSE_ENABLED = bool(LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY)

langfuse = None
if LANGFUSE_ENABLED:
    try:
        from langfuse import Langfuse  # available in your environment

        langfuse = Langfuse(
            public_key=LANGFUSE_PUBLIC_KEY,
            secret_key=LANGFUSE_SECRET_KEY,
            host=LANGFUSE_HOST,
        )
    except Exception:
        # Never crash API startup because of telemetry
        langfuse = None
        LANGFUSE_ENABLED = False

# -----------------------------
# App + request model
# -----------------------------
app = FastAPI(title="LLMOps Final Project API")

settings = get_settings()
app.state.settings = settings

configure_logging(app.state.settings.log_level)
logger = logging.getLogger("llmops.api")


class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = None
    variant: Optional[str] = None


# -----------------------------
# Release / Variant config
# -----------------------------
RELEASES_DIR = PROJECT_ROOT / "releases"
ACTIVE_RELEASE_FILE = PROJECT_ROOT / "active_release.txt"

VARIANT_CONFIG = {
    "v1": {
        "release_dir": RELEASES_DIR / "v1",
        "embedding_model": app.state.settings.embedding_model_v1,
    },
    "v2": {
        "release_dir": RELEASES_DIR / "v2",
        "embedding_model": app.state.settings.embedding_model_v2,
    },
}


def read_active_release() -> str:
    if not ACTIVE_RELEASE_FILE.exists():
        return "v1"
    return ACTIVE_RELEASE_FILE.read_text(encoding="utf-8").strip() or "v1"


def choose_variant(requested_variant: Optional[str]) -> str:
    if requested_variant:
        requested_variant = requested_variant.strip().lower()
        if requested_variant not in VARIANT_CONFIG:
            raise ValueError(
                f"Invalid variant '{requested_variant}'. Expected one of: {list(VARIANT_CONFIG.keys())}"
            )
        return requested_variant

    split_v2 = app.state.settings.ab_split_v2
    return "v2" if random.random() < split_v2 else "v1"


def load_store_for_variant(variant_name: str) -> FAISSVectorStore:
    if variant_name not in VARIANT_CONFIG:
        raise ValueError(f"Unknown variant '{variant_name}'. Expected one of: {list(VARIANT_CONFIG.keys())}")

    release_dir = VARIANT_CONFIG[variant_name]["release_dir"]
    index_path = release_dir / "faiss.index"
    meta_path = release_dir / "faiss_meta.json"

    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found for {variant_name}: {index_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"FAISS metadata not found for {variant_name}: {meta_path}")

    store = FAISSVectorStore(index_path=str(index_path), metadata_path=str(meta_path))
    store.load()
    return store


def _get_store_dim(store: FAISSVectorStore) -> Optional[int]:
    idx = getattr(store, "index", None)
    d = getattr(idx, "d", None)
    return int(d) if isinstance(d, (int, float)) else None


# -----------------------------
# Load embedders + stores once at import time
# -----------------------------
embedders: Dict[str, LocalSentenceTransformerProvider] = {}
stores: Dict[str, FAISSVectorStore] = {}
retrievers: Dict[str, Retriever] = {}
variant_dims: Dict[str, Dict[str, Optional[int]]] = {}

for variant_name, cfg in VARIANT_CONFIG.items():
    store = load_store_for_variant(variant_name)
    embedder = LocalSentenceTransformerProvider(model_name=cfg["embedding_model"])
    retriever = Retriever(embedder=embedder, store=store)

    store_dim = _get_store_dim(store)
    embed_dim = getattr(embedder, "embedding_dim", None)

    if embed_dim is None:
        try:
            _ = embedder.embed_query("dimension check")
            embed_dim = getattr(embedder, "embedding_dim", None)
        except Exception:
            embed_dim = None

    if store_dim is not None and embed_dim is not None and int(store_dim) != int(embed_dim):
        raise RuntimeError(
            f"Dimension mismatch for {variant_name}: FAISS index dim={store_dim} "
            f"but embedder dim={embed_dim} (model={cfg['embedding_model']}). "
            f"Rebuild releases/{variant_name} with the same embedding model used by the API."
        )

    stores[variant_name] = store
    embedders[variant_name] = embedder
    retrievers[variant_name] = retriever
    variant_dims[variant_name] = {
        "store_dim": store_dim,
        "embed_dim": int(embed_dim) if embed_dim is not None else None,
        "model": cfg["embedding_model"],
    }


# -----------------------------
# Endpoints
# -----------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "active_release": read_active_release(),
        "variants_loaded": list(retrievers.keys()),
        "variant_dims": variant_dims,
        "ab_split_v2": app.state.settings.ab_split_v2,
        "langfuse_enabled": bool(langfuse is not None),
        "langfuse_host": LANGFUSE_HOST if langfuse is not None else None,
        "log_level": app.state.settings.log_level,
    }


@app.post("/search")
def search(request: QueryRequest):
    active_release = read_active_release()

    try:
        variant = choose_variant(request.variant)
    except ValueError as e:
        return {
            "error": str(e),
            "active_release": active_release,
        }

    if variant not in retrievers:
        return {
            "error": f"Unknown variant '{variant}'. Expected one of: {list(retrievers.keys())}",
            "active_release": active_release,
        }

    top_k = int(request.top_k or app.state.settings.default_top_k)

    t0 = time.perf_counter()
    result = retrievers[variant].retrieve(request.query, top_k=top_k)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    embedding_model = VARIANT_CONFIG[variant]["embedding_model"]
    embedding_dim = getattr(embedders[variant], "embedding_dim", None)

    scores: List[float] = []
    for item in result.items:
        s = item.get("_score")
        if isinstance(s, (int, float)):
            scores.append(float(s))

    payload = {
        "query": request.query,
        "variant": variant,
        "active_release": active_release,
        "latency_ms": latency_ms,
        "embedding_model": embedding_model,
        "embedding_dim": int(embedding_dim) if isinstance(embedding_dim, (int, float)) else embedding_dim,
        "results": result.items,
        "score_stats": {
            "max": max(scores) if scores else None,
            "min": min(scores) if scores else None,
            "avg": (sum(scores) / len(scores)) if scores else None,
        },
        "trace_id": None,
    }

    logger.info(
        "search query_len=%s top_k=%s variant=%s active_release=%s latency_ms=%.2f embedding_model=%s",
        len(request.query or ""),
        top_k,
        variant,
        active_release,
        float(latency_ms),
        embedding_model,
    )

    if langfuse is not None:
        try:
            langfuse.create_event(
                name="search_request",
                metadata={
                    "query": request.query,
                    "top_k": top_k,
                    "active_release": active_release,
                    "variant": variant,
                    "latency_ms": latency_ms,
                    "embedding_model": embedding_model,
                    "embedding_dim": payload["embedding_dim"],
                    "score_max": payload["score_stats"]["max"],
                    "score_min": payload["score_stats"]["min"],
                    "score_avg": payload["score_stats"]["avg"],
                },
            )
        except Exception:
            pass

    return payload