from __future__ import annotations

import os
from typing import List

import numpy as np
from openai import OpenAI

from shared.embedding_provider import EmbeddingProvider


class OpenAIEmbedder(EmbeddingProvider):
    """
    OpenAI embeddings provider for the ingestion pipeline.

    Moved from src/core/openai_embedding_provider.py and renamed to match
    the target architecture. Requires OPENAI_API_KEY in environment.

    Default model is text-embedding-3-small per pipeline.yaml.
    """

    def __init__(self, model: str = "text-embedding-3-small") -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set.")
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def embed_text(self, texts: List[str]) -> np.ndarray:
        resp = self.client.embeddings.create(model=self.model, input=texts)
        vectors = [d.embedding for d in resp.data]
        return np.array(vectors, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_text([query])[0]
