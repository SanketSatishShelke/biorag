"""
Local cross-encoder reranker.

Runs on Saraswati's CPU with no external API dependency. Avoids the
deprecation/rate-limit/quota risk hit twice in one session with hosted
rerank endpoints (NVIDIA NIM direct, then OpenRouter's free-tier limits).

Model: cross-encoder/ms-marco-MiniLM-L-6-v2 -- general-purpose, not
biomedical-tuned. No widely-adopted biomedical cross-encoder exists as a
drop-in replacement the way PubMedBERT was for embeddings. Evaluating a
domain-tuned reranker is a candidate Phase 5/6 item once retrieval
metrics can measure whether it actually helps.
"""
from sentence_transformers import CrossEncoder

_model = None  # lazy-loaded, same pattern as the embedder


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        _model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _model


def rerank(query: str, candidates: list[dict]) -> list[dict]:
    """
    Rerank candidate chunks using a local cross-encoder.

    Takes the RRF-fused candidate list and rescores each (query, chunk) pair
    using full cross-attention -- capturing relevance signals that vector
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

    model = _get_model()
    pairs = [(query, c["text"]) for c in candidates]
    scores = model.predict(pairs)

    for chunk, score in zip(candidates, scores):
        chunk["rerank_score"] = float(score)

    return sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)