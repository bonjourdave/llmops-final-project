from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import faiss
import numpy as np

from shared.vector_store import VectorStoreProvider


class FAISSVectorStore(VectorStoreProvider):
    """
    Local FAISS-backed vector store.

    Suitable for development and testing without a cloud vector store account.
    Persists the index and metadata to disk so they survive process restarts.

    Production backend: shared/backends/zilliz_store.py (Phase 6)
    Local dev backend:  shared/backends/chroma_store.py  (Phase 2)
    """

    def __init__(self, index_path: str, metadata_path: str):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.index: Optional[faiss.Index] = None
        self.metadata: List[Dict] = []

    def create_collection(self, name: str, dimension: int) -> None:
        """
        For FAISS the 'collection' is an on-disk index file.
        This is a no-op when paths are already set via __init__,
        kept here to satisfy the VectorStoreProvider interface.
        """
        pass

    def _ensure_parent_dirs(self, path: str) -> None:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
        return vectors / norms

    def _init_index_if_needed(self, dim: int) -> None:
        if self.index is None:
            self.index = faiss.IndexFlatIP(dim)

    def add_embeddings(self, embeddings: np.ndarray, metadata: List[Dict]) -> None:
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be 2D array of shape (n, dim)")
        if len(metadata) != embeddings.shape[0]:
            raise ValueError("metadata length must match number of embeddings")

        embeddings = embeddings.astype("float32")
        embeddings = self._normalize(embeddings)
        self._init_index_if_needed(embeddings.shape[1])
        self.index.add(embeddings)
        self.metadata.extend(metadata)

        self._ensure_parent_dirs(self.index_path)
        self._ensure_parent_dirs(self.metadata_path)
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    def load(self) -> None:
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"FAISS index not found: {self.index_path}")
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"FAISS metadata not found: {self.metadata_path}")
        self.index = faiss.read_index(self.index_path)
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict]:
        if self.index is None:
            raise RuntimeError("FAISS index not loaded. Call load() before search().")

        query_vector = query_vector.astype("float32")
        query_vector = self._normalize(query_vector)
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        scores, indices = self.index.search(query_vector, top_k)
        results: List[Dict] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            item = dict(self.metadata[idx])
            item["_score"] = float(score)
            item["_index"] = int(idx)
            results.append(item)
        return results
