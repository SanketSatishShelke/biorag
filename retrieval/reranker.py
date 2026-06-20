import os
import httpx
from dotenv import load_dotenv

load_dotenv()

RERANKER_URL = "https://ai.api.nvidia.com/v1/retrieval/nvidia/reranking"
RERANKER_MODEL = "nv-rerank-qa-mistral-4b:1"
RERANKER_TIMEOUT = 30.0


def rerank(query: str, candidates: list[dict]) -> list[dict]:
    """
    Rerank candidate chunks using NVIDIA's nv-rerank-qa-mistral-4b cross-encoder.

    Takes the RRF-fused candidate list and rescores each (query, chunk) pair
    using full cross-attention — capturing relevance signals that vector
    similarity and BM25 cannot (negation, specificity, conditional relationships).

    Args:
        query:      the user's original question
        candidates: RRF-fused chunks from retrieve(), each must have 'text' key

    Returns:
        candidates reordered by cross-encoder relevance score, descending.
        Each dict gets a 'rerank_score' key added.
    """
    if not candidates:
        return []

    api_key = os.getenv("NIM_API_KEY")

    response = httpx.post(
        RERANKER_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
        },
        json={
            "model": RERANKER_MODEL,
            "query": {"text": query},
            "passages": [{"text": c["text"]} for c in candidates],
        },
        timeout=RERANKER_TIMEOUT,
    )

    response.raise_for_status()
    rankings = response.json()["rankings"]

    # rankings is a list of {"index": int, "logit": float}
    score_map = {r["index"]: r["logit"] for r in rankings}

    for i, chunk in enumerate(candidates):
        chunk["rerank_score"] = score_map.get(i, float("-inf"))

    return sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
