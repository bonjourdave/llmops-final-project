from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.core.local_embedding_provider import LocalSentenceTransformerProvider
from src.core.faiss_store import FAISSVectorStore


def main():
    # 1) Load data (adjust if your folder name is different)
    csv_path = Path("dataraw") / "netflix_titles.csv"
    df = pd.read_csv(csv_path)

    # 2) Create a text field to embed
    # We'll keep it simple: title + type + listed_in + description
    df["text_for_embedding"] = (
        df["title"].fillna("") + " | " +
        df["type"].fillna("") + " | " +
        df["listed_in"].fillna("") + " | " +
        df["description"].fillna("")
    )

    # 3) Convert to lists
    texts = df["text_for_embedding"].tolist()

    # 4) Embed
    embedder = LocalSentenceTransformerProvider()
    embeddings = embedder.embed_text(texts)

    # 5) Prepare metadata
    metadata = []
    for _, row in df.iterrows():
        metadata.append({
            "show_id": row.get("show_id"),
            "title": row.get("title"),
            "type": row.get("type"),
            "listed_in": row.get("listed_in"),
            "description": row.get("description"),
        })

    # 6) Store in FAISS
    store = FAISSVectorStore(
        index_path="releases/current/faiss.index",
        metadata_path="releases/current/faiss_meta.json",
    )
    store.add_embeddings(embeddings, metadata)

    print("✅ FAISS index built successfully!")
    print("Rows indexed:", len(df))
    print("Embedding dim:", embeddings.shape[1])


if __name__ == "__main__":
    main()