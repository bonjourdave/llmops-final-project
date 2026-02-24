from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict

from .embedding_provider import EmbeddingProvider
from .vector_store_provider import VectorStoreProvider


@dataclass
class RetrievalResult:
    items: List[Dict]


class Retriever:
    """
    Small retrieval wrapper:
    - uses an EmbeddingProvider to embed the query
    - uses a VectorStoreProvider to search
    """

    def __init__(self, embedder: EmbeddingProvider, store: VectorStoreProvider):
        self.embedder = embedder
        self.store = store

    def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        qvec = self.embedder.embed_query(query)
        hits = self.store.search(qvec, top_k=top_k)
        return RetrievalResult(items=hits)