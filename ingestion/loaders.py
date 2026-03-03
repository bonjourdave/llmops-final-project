from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List


def load_netflix_csv(path: str | Path) -> List[Dict]:
    """
    Load the Netflix titles CSV and return one record per row.

    Each record contains the raw fields plus a pre-built 'text' field using
    the canonical composite format:
        title | type | listed_in | description

    This is the same field construction that the original build_faiss_index.py
    used and carries forward unchanged into the new pipeline.
    """
    records: List[Dict] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = (
                (row.get("title") or "") + " | "
                + (row.get("type") or "") + " | "
                + (row.get("listed_in") or "") + " | "
                + (row.get("description") or "")
            )
            records.append(
                {
                    "show_id": row.get("show_id", ""),
                    "title": row.get("title", ""),
                    "type": row.get("type", ""),
                    "listed_in": row.get("listed_in", ""),
                    "description": row.get("description", ""),
                    "text": text,
                }
            )
    return records
