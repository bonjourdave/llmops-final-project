from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict

from shared.embedding_provider import EmbeddingProvider
from shared.vector_store import VectorStoreProvider


@dataclass
class RetrievalResult:
    """Raw retrieval output — a list of metadata dicts from the vector store."""
    items: List[Dict]


class Retriever:
    """
    Combines an EmbeddingProvider and a VectorStoreProvider to answer a query.

    In Phase 3 this will gain Langfuse tracing and version tagging.
    Both provider implementations are injected so they can be swapped via config.
    """

    def __init__(self, embedder: EmbeddingProvider, store: VectorStoreProvider):
        self.embedder = embedder
        self.store = store

    def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        query_vector = self.embedder.embed_query(query)
        hits = self.store.search(query_vector, top_k=top_k)
        return RetrievalResult(items=hits)
