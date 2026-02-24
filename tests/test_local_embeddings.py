import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.core.local_embedding_provider import LocalSentenceTransformerProvider


def main():
    provider = LocalSentenceTransformerProvider()

    text = "This is a test sentence for embeddings."
    vector = provider.embed_query(text)

    print("Embedding length:", len(vector))
    print("First 5 values:", vector[:5])


if __name__ == "__main__":
    main()