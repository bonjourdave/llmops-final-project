import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.core.local_embedding_provider import LocalSentenceTransformerProvider
from src.core.faiss_store import FAISSVectorStore
from src.core.retriever import Retriever


def main():
    store = FAISSVectorStore(
        index_path="releases/current/faiss.index",
        metadata_path="releases/current/faiss_meta.json",
    )
    store.load()

    embedder = LocalSentenceTransformerProvider()
    retriever = Retriever(embedder=embedder, store=store)

    query = "romantic korean drama series"
    result = retriever.retrieve(query, top_k=5)

    print("Query:", query)
    for item in result.items:
        print("-", item.get("title"), "|", item.get("type"), "| score:", round(item["_score"], 4))


if __name__ == "__main__":
    main()