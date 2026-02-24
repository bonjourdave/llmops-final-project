from __future__ import annotations

import json
import os
from typing import List, Dict, Optional

import numpy as np
import faiss

from .vector_store_provider import VectorStoreProvider


class FAISSVectorStore(VectorStoreProvider):
    """
    Minimal FAISS vector store implementation for A/B testing vs Milvus.

    Stores:
    - FAISS index on disk (index_path)
    - metadata list on disk (metadata_path), same order as vectors in FAISS
    """

    def __init__(self, index_path: str, metadata_path: str):
        self.index_path = index_path
        self.metadata_path = metadata_path

        self.index: Optional[faiss.Index] = None
        self.metadata: List[Dict] = []

    def _ensure_parent_dirs(self, path: str) -> None:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        """
        Normalize vectors to unit length for cosine similarity.
        FAISS will use inner product (IP) to approximate cosine after normalization.
        """
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
        return vectors / norms

    def _init_index_if_needed(self, dim: int) -> None:
        if self.index is None:
            # Inner product index; with normalized vectors this behaves like cosine.
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

        # Persist to disk
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
            # If index exists on disk, user should call load(). Otherwise, this is a mistake.
            raise RuntimeError("FAISS index is not loaded. Call load() before search().")

        query_vector = query_vector.astype("float32")
        query_vector = self._normalize(query_vector)

        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        scores, indices = self.index.search(query_vector, top_k)

        results: List[Dict] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            item = dict(self.metadata[idx])  # copy
            item["_score"] = float(score)
            item["_index"] = int(idx)
            results.append(item)

        return results