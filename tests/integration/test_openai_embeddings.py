import sys
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.core.openai_embedding_provider import OpenAIEmbeddingProvider


def main():
    provider = OpenAIEmbeddingProvider()

    text = "This is a test sentence for embeddings."
    vector = provider.embed_query(text)

    print("Embedding length:", len(vector))
    print("First 5 values:", vector[:5])


if __name__ == "__main__":
    main()