from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

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

    lf_client and trace are optional: when supplied, the retrieve() call is
    wrapped in a Langfuse child span. This keeps tracing out of the
    retriever's constructor so the class stays testable without Langfuse.
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

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        lf_client: Optional[Any] = None,
        trace: Optional[Any] = None,
    ) -> RetrievalResult:
        span = None
        if lf_client is not None and trace is not None:
            span = lf_client.span(
                trace,
                name="retrieve",
                input={"query": query, "top_k": top_k, "version": self.version},
            )

        query_vector = self.embedder.embed_query(query)
        hits = self.store.search(query_vector, top_k=top_k)

        if lf_client is not None and span is not None:
            lf_client.end_span(span, output={"num_hits": len(hits)})

        return RetrievalResult(items=hits, version=self.version)
