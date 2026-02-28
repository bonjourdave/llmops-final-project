from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class LocalSentenceTransformerProvider:
    """
    Local embedding provider using sentence-transformers.

    IMPORTANT:
    - The model you use here MUST match the model used to build the FAISS index.
    - Different models produce different embedding dimensions (e.g., 384 vs 768).
    """

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    _model: Optional[SentenceTransformer] = None

    def _load(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dim(self) -> int:
        m = self._load()
        return m.get_sentence_embedding_dimension()

    def embed_text(self, texts: List[str]) -> np.ndarray:
        m = self._load()
        vecs = m.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        # Ensure float32 for FAISS
        return np.asarray(vecs, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        vec = self.embed_text([query])[0]
        return vec