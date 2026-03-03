from typing import Any, Dict, List

from pydantic import BaseModel


class Chunk(BaseModel):
    """
    A single indexed unit produced by the ingestion pipeline.
    Stored in the vector store alongside its embedding vector.
    """

    id: str
    text: str
    vector: List[float]
    metadata: Dict[str, Any]  # source doc, version tag, chunk index, etc.


class RetrievalResult(BaseModel):
    """
    A single result returned by the serving retriever.
    Wraps the stored Chunk with the similarity score from the vector store.
    """

    chunk: Chunk
    score: float  # cosine similarity score from the vector store
