from __future__ import annotations

from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

from .embedding_provider import EmbeddingProvider


class LocalSentenceTransformerProvider(EmbeddingProvider):
    """
    Free local embedding provider using SentenceTransformers.
    Default model is small and fast for laptops.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_text(self, texts: List[str]) -> np.ndarray:
        vectors = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return vectors.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        vecs = self.embed_text([query])
        return vecs[0]