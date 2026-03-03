from abc import ABC, abstractmethod
from typing import List, Dict

import numpy as np


class VectorStoreProvider(ABC):
    """
    Base interface for all vector store backends.
    Concrete implementations live in shared/backends/:
      - faiss_store.py  (local dev / testing)
      - chroma_store.py (local dev, Phase 2+)
      - zilliz_store.py (production, Phase 6+)

    Backend is selected at runtime via the 'vector_store.backend' config key.
    """

    @abstractmethod
    def create_collection(self, name: str, dimension: int) -> None:
        """
        Create a named collection (or index) for a specific embedding dimension.
        Called once per ingestion run before writing any chunks.
        """
        pass

    @abstractmethod
    def add_embeddings(self, embeddings: np.ndarray, metadata: List[Dict]) -> None:
        """
        Add a batch of embeddings and their associated metadata to the store.
        embeddings: float32 array of shape (n, dim)
        metadata:   list of dicts, one per embedding row
        """
        pass

    @abstractmethod
    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict]:
        """
        Find the top_k nearest neighbours for query_vector.
        Returns a list of metadata dicts, each augmented with '_score'.
        """
        pass
