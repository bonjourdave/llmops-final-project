from abc import ABC, abstractmethod
from typing import List, Dict
import numpy as np


class VectorStoreProvider(ABC):
    """
    Base interface for all vector store providers.
    This allows us to swap FAISS, Milvus, etc.
    without changing retrieval logic.
    """

    @abstractmethod
    def add_embeddings(
        self,
        embeddings: np.ndarray,
        metadata: List[Dict]
    ) -> None:
        """
        Add embeddings and associated metadata to the store.
        """
        pass

    @abstractmethod
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Search for nearest neighbors and return results.
        """
        pass