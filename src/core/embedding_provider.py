from abc import ABC, abstractmethod
from typing import List
import numpy as np


class EmbeddingProvider(ABC):
    """
    Base interface for all embedding providers.
    This allows us to swap embedding models without
    changing retrieval logic.
    """

    @abstractmethod
    def embed_text(self, texts: List[str]) -> np.ndarray:
        """
        Takes a list of text strings and returns
        a 2D numpy array of embeddings.
        """
        pass

    @abstractmethod
    def embed_query(self, query: str) -> np.ndarray:
        """
        Takes a single query string and returns
        a 1D numpy array embedding.
        """
        pass