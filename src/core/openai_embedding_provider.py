from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import os
from typing import List
import numpy as np
from openai import OpenAI

from .embedding_provider import EmbeddingProvider


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """
    OpenAI embeddings provider.
    Requires OPENAI_API_KEY set in environment (we'll set via .env soon).
    """

    def __init__(self, model: str = "text-embedding-3-small"):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set. Put it in a .env file.")
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def embed_text(self, texts: List[str]) -> np.ndarray:
        resp = self.client.embeddings.create(
            model=self.model,
            input=texts
        )
        vectors = [d.embedding for d in resp.data]
        return np.array(vectors, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        vecs = self.embed_text([query])
        return vecs[0]