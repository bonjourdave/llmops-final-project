"""
Netflix ingestion Prefect flow.

Flow: load → chunk → embed → write → update versions.json

Run locally:
    python ingestion/dag.py

Or via Prefect CLI:
    prefect run -p ingestion/dag.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from prefect import flow, task

from ingestion.chunkers import TextChunker
from ingestion.embedders import OpenAIEmbedder
from ingestion.loaders import load_netflix_csv
from shared.config_loader import load_config

CONFIG_PATH = Path("config/pipeline.yaml")
VERSIONS_PATH = Path("versions.json")
DATA_PATH = Path("dataraw/netflix_titles.csv")
EMBED_BATCH_SIZE = 100

# Known embedding dimensions — passed to Zilliz at collection creation time.
_EMBEDDING_DIM: Dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------


@task(name="load-records")
def load_records() -> List[Dict]:
    records = load_netflix_csv(DATA_PATH)
    print(f"Loaded {len(records)} records from {DATA_PATH}")
    return records


@task(name="chunk-records")
def chunk_records(
    records: List[Dict], chunk_size: int, chunk_overlap: int
) -> List[Dict]:
    chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks: List[Dict] = []
    for rec in records:
        for i, text in enumerate(chunker.split(rec["text"])):
            chunks.append(
                {
                    "show_id": rec["show_id"],
                    "title": rec["title"],
                    "type": rec["type"],
                    "listed_in": rec["listed_in"],
                    "description": rec["description"],
                    "text": text,
                    "chunk_index": i,
                }
            )
    print(f"Produced {len(chunks)} chunks from {len(records)} records")
    return chunks


@task(name="embed-and-write")
def embed_and_write(
    chunks: List[Dict],
    model: str,
    collection_name: str,
    persist_dir: str,
    backend: str = "chroma",
    connection_uri: str = "",
    token: str = "",
) -> int:
    embedder = OpenAIEmbedder(model=model)

    if backend == "zilliz":
        if not connection_uri or not token:
            raise ValueError(
                "backend=zilliz but ZILLIZ_URI or ZILLIZ_TOKEN is not set. "
                "Check your .env file."
            )
        from shared.backends.zilliz_store import ZillizVectorStore
        store = ZillizVectorStore(uri=connection_uri, token=token)
    else:
        from shared.backends.chroma_store import ChromaVectorStore
        store = ChromaVectorStore(persist_directory=persist_dir)

    print(f"embed_and_write: backend={backend} collection={collection_name}")
    dim = _EMBEDDING_DIM.get(model, 1536)
    store.create_collection(collection_name, dimension=dim)

    for i in range(0, len(chunks), EMBED_BATCH_SIZE):
        batch = chunks[i : i + EMBED_BATCH_SIZE]
        texts = [c["text"] for c in batch]
        embeddings = embedder.embed_text(texts)
        store.add_embeddings(embeddings, batch)
        print(f"  Written batch {i // EMBED_BATCH_SIZE + 1}: {len(batch)} chunks")

    return len(chunks)


@task(name="update-versions")
def update_versions(
    version: str,
    model: str,
    chunk_size: int,
    chunk_overlap: int,
    doc_count: int,
) -> None:
    versions: List[Dict] = []
    if VERSIONS_PATH.exists():
        versions = json.loads(VERSIONS_PATH.read_text(encoding="utf-8"))

    versions.append(
        {
            "version": version,
            "model": model,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "doc_count": doc_count,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "notes": "",
        }
    )

    VERSIONS_PATH.write_text(json.dumps(versions, indent=2), encoding="utf-8")
    print(f"versions.json updated — version {version}, {doc_count} chunks")


# ---------------------------------------------------------------------------
# Flow
# ---------------------------------------------------------------------------


@flow(name="netflix-ingestion")
def ingest() -> None:
    cfg = load_config(CONFIG_PATH)
    pipeline_cfg = cfg["pipeline"]
    vs_cfg = cfg["vector_store"]

    model: str = pipeline_cfg["embedding_model"]
    chunk_size: int = pipeline_cfg["chunk_size"]
    chunk_overlap: int = pipeline_cfg["chunk_overlap"]
    version: str = pipeline_cfg["vector_store_version"]
    collection_name: str = f"{vs_cfg['collection_prefix']}_{version}"
    backend: str = vs_cfg.get("backend", "chroma")
    persist_dir: str = ".chroma"

    records = load_records()
    chunks = chunk_records(records, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    doc_count = embed_and_write(
        chunks,
        model=model,
        collection_name=collection_name,
        persist_dir=persist_dir,
        backend=backend,
        connection_uri=vs_cfg.get("connection_uri", ""),
        token=vs_cfg.get("token", ""),
    )
    update_versions(
        version=version,
        model=model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        doc_count=doc_count,
    )


if __name__ == "__main__":
    ingest()
