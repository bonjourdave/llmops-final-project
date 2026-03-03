from __future__ import annotations

from typing import Dict, List

import numpy as np

from shared.vector_store import VectorStoreProvider


class ChromaVectorStore(VectorStoreProvider):
    """
    Chroma-backed vector store for local dev and Phase 2 ingestion.

    Uses a PersistentClient so data survives between process restarts.
    Cosine similarity is set via collection metadata so that Chroma's
    distances (0 = identical) are converted to similarity scores on search.

    Selected when pipeline.yaml sets vector_store.backend: "chroma".
    """

    def __init__(self, persist_directory: str = ".chroma") -> None:
        import chromadb

        self._client = chromadb.PersistentClient(path=persist_directory)
        self._collection = None

    def create_collection(self, name: str, dimension: int) -> None:
        """
        Get or create a named collection.
        dimension is accepted for interface compatibility (Chroma infers it).
        """
        self._collection = self._client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_embeddings(self, embeddings: np.ndarray, metadata: List[Dict]) -> None:
        if self._collection is None:
            raise RuntimeError("Call create_collection() before add_embeddings().")

        ids = [
            f"{m.get('show_id', '')}_{m.get('chunk_index', 0)}"
            for m in metadata
        ]
        documents = [m.get("text", "") for m in metadata]
        # Chroma stores metadata separately from documents; strip 'text' from
        # the metadata dict to avoid duplication.
        meta_without_text = [
            {k: v for k, v in m.items() if k != "text"} for m in metadata
        ]

        self._collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=documents,
            metadatas=meta_without_text,
        )

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict]:
        if self._collection is None:
            raise RuntimeError("Call create_collection() before search().")

        results = self._collection.query(
            query_embeddings=[query_vector.tolist()],
            n_results=top_k,
            include=["metadatas", "documents", "distances"],
        )

        hits: List[Dict] = []
        for i, meta in enumerate(results["metadatas"][0]):
            hit = dict(meta)
            hit["text"] = results["documents"][0][i]
            # Chroma cosine distance: 0 = identical, 2 = opposite.
            # Convert to a 0–1 similarity score.
            hit["_score"] = 1.0 - results["distances"][0][i] / 2.0
            hits.append(hit)

        return hits
