"""
Zilliz Cloud (managed Milvus) vector store backend.

Uses the high-level MilvusClient API with:
  - enable_dynamic_field=True  → all metadata stored without explicit schema fields
  - COSINE metric              → distance is cosine similarity; no conversion needed
  - AUTOINDEX                  → Zilliz selects the best index type automatically

create_collection semantics:
  dimension=0  → serving mode: connect to an existing collection
  dimension>0  → ingestion mode: drop + recreate for a clean run
"""

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np

from shared.vector_store import VectorStoreProvider

logger = logging.getLogger("llmops.zilliz_store")


class ZillizVectorStore(VectorStoreProvider):
    """
    Zilliz Cloud backend.

    uri and token are read from config/pipeline.yaml via ${ZILLIZ_URI} / ${ZILLIZ_TOKEN}.
    """

    def __init__(self, uri: str, token: str) -> None:
        if not uri:
            raise ValueError(
                "ZILLIZ_URI is not set. Add it to .env and rebuild the container."
            )
        from pymilvus import MilvusClient

        self._client = MilvusClient(uri=uri, token=token)
        self._collection_name: str = ""

    def create_collection(self, name: str, dimension: int) -> None:
        self._collection_name = name
        if dimension == 0:
            # Serving mode: collection was already created by ingestion.
            return
        # Ingestion mode: start fresh.
        from pymilvus import DataType

        if self._client.has_collection(name):
            self._client.drop_collection(name)
            logger.info("Dropped existing collection '%s'", name)

        schema = self._client.create_schema(auto_id=False, enable_dynamic_field=True)
        schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=128)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=dimension)

        index_params = self._client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            metric_type="COSINE",
            index_type="AUTOINDEX",
        )

        self._client.create_collection(
            collection_name=name,
            schema=schema,
            index_params=index_params,
        )
        logger.info("Created collection '%s' (dim=%d, metric=COSINE)", name, dimension)

    def add_embeddings(self, embeddings: np.ndarray, metadata: List[Dict]) -> None:
        if not self._collection_name:
            raise RuntimeError("Call create_collection() before add_embeddings().")

        data = [
            {
                "id": f"{m.get('show_id', '')}_{m.get('chunk_index', 0)}",
                "embedding": embeddings[i].tolist(),
                **m,
            }
            for i, m in enumerate(metadata)
        ]
        self._client.insert(collection_name=self._collection_name, data=data)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict]:
        if not self._collection_name:
            raise RuntimeError("Call create_collection() before search().")

        results = self._client.search(
            collection_name=self._collection_name,
            data=[query_vector.tolist()],
            limit=top_k,
            output_fields=["*"],
            search_params={"metric_type": "COSINE"},
        )

        hits: List[Dict] = []
        for hit in results[0]:
            row = dict(hit["entity"])
            row["id"] = hit["id"]
            # COSINE distance in Milvus is already cosine similarity: range [–1, 1].
            row["_score"] = hit["distance"]
            hits.append(row)
        return hits
