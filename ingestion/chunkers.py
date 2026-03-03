from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class TextChunker:
    """
    Splits text into overlapping character-level chunks.

    Parameters are read from config/pipeline.yaml (chunk_size, chunk_overlap)
    and passed in at construction time by the DAG.

    Netflix composite texts are typically 100–400 chars, well under the
    default 512-char chunk_size, so most records produce a single chunk.
    The splitter is still needed for correctness with longer descriptions.
    """

    chunk_size: int
    chunk_overlap: int

    def split(self, text: str) -> List[str]:
        """
        Return a list of chunks covering the full text.

        If the text fits within chunk_size it is returned as-is (no split).
        Otherwise chunks advance by (chunk_size - chunk_overlap) characters
        and always cover the tail of the text in the final chunk.
        """
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.chunk_overlap
        if step <= 0:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be less than "
                f"chunk_size ({self.chunk_size})"
            )

        chunks: List[str] = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunks.append(text[start:end])
            if end >= len(text):
                break
            start += step

        return chunks
