from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.core.local_embedding_provider import LocalSentenceTransformerProvider
from src.core.faiss_store import FAISSVectorStore


def main():
    # Read active release name
    release = Path("active_release.txt").read_text().strip()

    store = FAISSVectorStore(
        index_path=f"releases/{release}/faiss.index",
        metadata_path=f"releases/{release}/faiss_meta.json",
    )
    store.load()

    embedder = LocalSentenceTransformerProvider()

    query = "funny stand-up comedy special"
    qvec = embedder.embed_query(query)

    results = store.search(qvec, top_k=5)

    print("Query:", query)
    print("Top 5 results:")
    for r in results:
        print("-", r.get("title"), "|", r.get("type"), "| score:", round(r["_score"], 4))


if __name__ == "__main__":
    main()