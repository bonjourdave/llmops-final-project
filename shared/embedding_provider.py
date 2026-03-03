from abc import ABC, abstractmethod
from typing import List

import numpy as np


class EmbeddingProvider(ABC):
    """
    Base interface for all embedding providers.
    Implementations live in ingestion/embedders.py (OpenAI) and
    shared/backends/ for any local alternatives.
    """

    @abstractmethod
    def embed_text(self, texts: List[str]) -> np.ndarray:
        """Takes a list of strings, returns a 2D float32 array of shape (n, dim)."""
        pass

    @abstractmethod
    def embed_query(self, query: str) -> np.ndarray:
        """Takes a single query string, returns a 1D float32 array of shape (dim,)."""
        pass
