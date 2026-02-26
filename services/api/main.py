from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from langfuse import observe, get_client

# Load .env variables
load_dotenv()

# Add project root to path (so we can import src.*)
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.core.local_embedding_provider import LocalSentenceTransformerProvider
from src.core.faiss_store import FAISSVectorStore
from src.core.retriever import Retriever

app = FastAPI(title="LLMOps Final Project API")

# Langfuse client reads LANGFUSE_* keys from environment
langfuse = get_client()

# Read active release
ACTIVE_RELEASE = Path("active_release.txt").read_text().strip()

# --- Initialize components at startup ---
store = FAISSVectorStore(
    index_path=f"releases/{ACTIVE_RELEASE}/faiss.index",
    metadata_path=f"releases/{ACTIVE_RELEASE}/faiss_meta.json",
)
store.load()

embedder = LocalSentenceTransformerProvider()
retriever = Retriever(embedder=embedder, store=store)


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


@app.get("/health")
def health():
    return {"status": "ok", "active_release": ACTIVE_RELEASE}


@app.post("/search")
@observe()
def search(request: QueryRequest):
    # Record request input in the current trace
    langfuse.update_current_trace(
        name="search_request",
        input={
            "query": request.query,
            "top_k": request.top_k,
            "active_release": ACTIVE_RELEASE,
            "embedding_model": "all-MiniLM-L6-v2",
            "vector_store": "faiss",
        },
    )

    result = retriever.retrieve(request.query, top_k=request.top_k)

    # Record a lightweight summary output
    langfuse.update_current_trace(
        output={
            "num_results": len(result.items),
            "top_titles": [item.get("title") for item in result.items],
        }
    )

    # Ensure the trace is sent promptly
    langfuse.flush()

    return {"query": request.query, "active_release": ACTIVE_RELEASE, "results": result.items}