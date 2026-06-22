# Default threshold for rerank score below which we refuse to answer.
# Cross-encoder logits for clearly relevant pairs: > 0
# Marginally relevant: -3 to -5
# Irrelevant: < -5
# Tune this in Phase 5 using eval dataset.
DEFAULT_CONFIDENCE_THRESHOLD = -5.0


class LowConfidenceError(Exception):
    """Raised when retrieved chunks don't meet the confidence threshold."""
    def __init__(self, top_score: float, threshold: float):
        self.top_score = top_score
        self.threshold = threshold
        super().__init__(
            f"Top rerank score {top_score:.3f} below threshold {threshold:.3f}"
        )


def check_confidence(
    chunks: list[dict],
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> None:
    """
    Check whether retrieved chunks meet the minimum confidence threshold.

    Uses the top chunk's rerank_score as the confidence signal — the
    cross-encoder score is the most reliable relevance indicator in the
    pipeline since it scores (query, chunk) jointly.

    Args:
        chunks:    reranked chunks from retrieve(), sorted by rerank_score desc
        threshold: minimum acceptable rerank_score for the top chunk

    Raises:
        LowConfidenceError: if top chunk's rerank_score is below threshold
    """
    if not chunks:
        raise LowConfidenceError(top_score=float("-inf"), threshold=threshold)

    top_score = chunks[0].get("rerank_score", float("-inf"))

    if top_score < threshold:
        raise LowConfidenceError(top_score=top_score, threshold=threshold)
