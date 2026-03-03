from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from shared.embedding_provider import EmbeddingProvider
from shared.vector_store import VectorStoreProvider


@dataclass
class RetrievalResult:
    """Raw retrieval output — a list of metadata dicts from the vector store."""

    items: List[Dict]
    version: str = ""


class Retriever:
    """
    Combines an EmbeddingProvider and a VectorStoreProvider to answer a query.

    version is attached to every RetrievalResult so the API can return it
    to the client and record it in the Langfuse trace.

    Tracing is handled at the API layer via LangfuseClient.span_context —
    the retriever itself has no Langfuse dependency.
    """

    def __init__(
        self,
        embedder: EmbeddingProvider,
        store: VectorStoreProvider,
        version: str = "",
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.version = version

    def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        query_vector = self.embedder.embed_query(query)
        hits = self.store.search(query_vector, top_k=top_k)
        return RetrievalResult(items=hits, version=self.version)
