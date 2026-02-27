import os
import time
import statistics
from typing import Dict

import requests
from dotenv import load_dotenv

API_URL = "http://127.0.0.1:8000/search"

TEST_QUERIES = [
    "crime documentary series",
    "romantic comedy movie",
    "space exploration documentary",
    "spanish language thriller",
    "true crime limited series",
]


def init_langfuse():
    """
    Initialize Langfuse AFTER .env has been loaded.
    Returns a Langfuse client or None (never raises).
    """
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

    if not (public_key and secret_key):
        return None

    try:
        from langfuse import Langfuse

        return Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
        )
    except Exception:
        return None


def run_query(query: str, variant: str) -> Dict:
    payload = {"query": query, "variant": variant}

    start = time.perf_counter()
    response = requests.post(API_URL, json=payload)
    latency = (time.perf_counter() - start) * 1000.0

    if response.status_code != 200:
        raise RuntimeError(f"API error ({response.status_code}): {response.text}")

    data = response.json()

    return {
        "latency_ms": latency,
        "score_avg": data.get("score_stats", {}).get("avg"),
        "variant": variant,
        "query": query,
    }


def evaluate_variant(variant: str) -> Dict:
    latencies = []
    scores = []

    for q in TEST_QUERIES:
        result = run_query(q, variant)
        latencies.append(result["latency_ms"])
        if result["score_avg"] is not None:
            scores.append(result["score_avg"])

    return {
        "variant": variant,
        "avg_latency_ms": statistics.mean(latencies),
        "avg_score": statistics.mean(scores) if scores else None,
        "min_latency_ms": min(latencies),
        "max_latency_ms": max(latencies),
        "num_queries": len(TEST_QUERIES),
    }


def main():
    load_dotenv()

    langfuse = init_langfuse()

    print("Running A/B evaluation...")

    v1_results = evaluate_variant("v1")
    v2_results = evaluate_variant("v2")

    print("\nResults:")
    print(v1_results)
    print(v2_results)

    if langfuse is not None:
        try:
            langfuse.create_event(
                name="ab_evaluation",
                metadata={
                    "v1": v1_results,
                    "v2": v2_results,
                },
            )
            print("\nEvaluation logged to Langfuse.")
        except Exception as e:
            print(f"\nLangfuse logging failed: {e}")
    else:
        print("\nLangfuse not configured (missing keys) - evaluation not logged.")


if __name__ == "__main__":
    main()