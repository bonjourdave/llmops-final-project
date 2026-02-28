import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
from src.core.faiss_store import FAISSVectorStore


def main():
    store = FAISSVectorStore(
        index_path="releases/current/faiss.index",
        metadata_path="releases/current/faiss_meta.json"
    )

    # Create 3 fake vectors (dim=4)
    embeddings = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0.9, 0.1, 0, 0],
    ], dtype=np.float32)

    metadata = [
        {"id": "A", "text": "vector A"},
        {"id": "B", "text": "vector B"},
        {"id": "C", "text": "vector C"},
    ]

    store.add_embeddings(embeddings, metadata)

    # Load from disk to prove persistence works
    store2 = FAISSVectorStore(
        index_path="releases/current/faiss.index",
        metadata_path="releases/current/faiss_meta.json"
    )
    store2.load()

    query = np.array([1, 0, 0, 0], dtype=np.float32)
    results = store2.search(query, top_k=2)

    print("Top results:")
    for r in results:
        print(r["id"], r["_score"])


if __name__ == "__main__":
    main()