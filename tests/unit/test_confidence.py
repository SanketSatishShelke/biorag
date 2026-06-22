import pytest
from guardrails.confidence import check_confidence, LowConfidenceError, DEFAULT_CONFIDENCE_THRESHOLD


def _chunk(rerank_score: float) -> dict:
    return {
        "chunk_id": 1,
        "text": "Some biomedical text.",
        "filename": "paper.pdf",
        "page_number": 1,
        "score": 0.02,
        "semantic_score": 0.75,
        "rerank_score": rerank_score,
    }


def test_high_confidence_passes():
    """Chunks with rerank_score above threshold must not raise."""
    check_confidence([_chunk(rerank_score=2.0)])


def test_low_confidence_raises():
    """Chunks with rerank_score below threshold must raise LowConfidenceError."""
    with pytest.raises(LowConfidenceError):
        check_confidence([_chunk(rerank_score=-10.0)])


def test_empty_chunks_raises():
    """Empty chunk list must raise LowConfidenceError."""
    with pytest.raises(LowConfidenceError):
        check_confidence([])


def test_error_contains_scores():
    """LowConfidenceError must expose top_score and threshold."""
    with pytest.raises(LowConfidenceError) as exc_info:
        check_confidence([_chunk(rerank_score=-8.0)], threshold=-5.0)
    assert exc_info.value.top_score == -8.0
    assert exc_info.value.threshold == -5.0


def test_score_exactly_at_threshold_passes():
    """Score exactly at threshold must pass — threshold is a lower bound."""
    check_confidence([_chunk(rerank_score=DEFAULT_CONFIDENCE_THRESHOLD)],
                     threshold=DEFAULT_CONFIDENCE_THRESHOLD)


def test_only_top_chunk_score_is_checked():
    """Confidence is determined by the first chunk only — list is already sorted desc."""
    # first chunk passes, second would fail — must not raise
    check_confidence([
        _chunk(rerank_score=1.0),
        _chunk(rerank_score=-10.0),
    ])


def test_custom_threshold_respected():
    """Custom threshold must override the default."""
    # passes default threshold but fails strict custom threshold
    with pytest.raises(LowConfidenceError):
        check_confidence([_chunk(rerank_score=-3.0)], threshold=0.0)