import pytest
from pydantic import ValidationError

from shared.schemas import Chunk, RetrievalResult


def test_chunk_validates():
    chunk = Chunk(
        id="show_123",
        text="Stranger Things | TV Show | Sci-Fi & Fantasy | ...",
        vector=[0.1, 0.2, 0.3],
        metadata={"source": "netflix_titles.csv", "version": "v1"},
    )
    assert chunk.id == "show_123"
    assert len(chunk.vector) == 3
    assert chunk.metadata["version"] == "v1"


def test_chunk_metadata_is_flexible():
    chunk = Chunk(
        id="x",
        text="text",
        vector=[],
        metadata={"arbitrary_key": [1, 2, 3], "nested": {"a": 1}},
    )
    assert chunk.metadata["arbitrary_key"] == [1, 2, 3]


def test_chunk_requires_id():
    with pytest.raises(ValidationError):
        Chunk(text="t", vector=[], metadata={})


def test_retrieval_result_validates():
    chunk = Chunk(id="abc", text="hello", vector=[0.5], metadata={})
    result = RetrievalResult(chunk=chunk, score=0.92)
    assert result.score == 0.92
    assert result.chunk.id == "abc"


def test_retrieval_result_requires_chunk_and_score():
    with pytest.raises(ValidationError):
        RetrievalResult(score=0.5)
