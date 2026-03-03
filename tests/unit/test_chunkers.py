import pytest

from ingestion.chunkers import TextChunker


def test_short_text_is_single_chunk():
    chunker = TextChunker(chunk_size=100, chunk_overlap=10)
    assert chunker.split("Hello world") == ["Hello world"]


def test_text_exactly_at_limit_is_single_chunk():
    chunker = TextChunker(chunk_size=5, chunk_overlap=1)
    assert chunker.split("abcde") == ["abcde"]


def test_long_text_splits_into_multiple_chunks():
    chunker = TextChunker(chunk_size=5, chunk_overlap=1)
    # "abcdefghij" → step = 4
    # chunk 0: [0:5]  = "abcde"
    # chunk 1: [4:9]  = "efghi"
    # chunk 2: [8:13] = "ij"
    chunks = chunker.split("abcdefghij")
    assert len(chunks) == 3
    assert chunks[0] == "abcde"
    assert chunks[1] == "efghi"
    assert chunks[2] == "ij"


def test_chunks_overlap_by_correct_amount():
    chunker = TextChunker(chunk_size=6, chunk_overlap=2)
    text = "abcdefghij"
    # step = 4
    # chunk 0: [0:6]  = "abcdef"
    # chunk 1: [4:10] = "efghij"
    chunks = chunker.split(text)
    assert chunks[0][-chunker.chunk_overlap :] == chunks[1][: chunker.chunk_overlap]


def test_full_text_is_covered():
    chunker = TextChunker(chunk_size=10, chunk_overlap=3)
    text = "x" * 47
    chunks = chunker.split(text)
    # Reconstruct by stepping through the chunks without the overlap
    reconstructed = chunks[0]
    step = chunker.chunk_size - chunker.chunk_overlap
    for chunk in chunks[1:]:
        reconstructed += chunk[chunker.chunk_overlap :]
    assert reconstructed == text


def test_empty_string_returns_single_empty_chunk():
    chunker = TextChunker(chunk_size=100, chunk_overlap=10)
    assert chunker.split("") == [""]


def test_invalid_overlap_raises():
    chunker = TextChunker(chunk_size=5, chunk_overlap=5)
    with pytest.raises(ValueError, match="chunk_overlap"):
        chunker.split("a" * 10)


def test_typical_netflix_text_is_single_chunk():
    # Realistic composite text well within the 512-char default
    text = (
        "Stranger Things | TV Show | Sci-Fi & Fantasy, Horror TV Shows, TV Thrillers | "
        "When a young boy vanishes, a small town uncovers a mystery involving secret "
        "experiments, terrifying supernatural forces and one very strange little girl."
    )
    chunker = TextChunker(chunk_size=512, chunk_overlap=50)
    chunks = chunker.split(text)
    assert len(chunks) == 1
    assert chunks[0] == text
