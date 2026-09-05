import pytest
from unittest.mock import patch, MagicMock


def _mock_model(scores):
    """Fake CrossEncoder — .predict() returns the given scores in order."""
    model = MagicMock()
    model.predict.return_value = scores
    return model


def test_rerank_returns_list():
    from retrieval.reranker import rerank
    candidates = [
        {"chunk_id": 1, "text": "CD8+ T cells mediate antitumor immunity."},
        {"chunk_id": 2, "text": "mTOR signaling regulates cell growth."},
    ]
    with patch("retrieval.reranker._get_model", return_value=_mock_model([0.9, 0.5])):
        results = rerank("immune response", candidates)
    assert isinstance(results, list)


def test_rerank_adds_rerank_score():
    from retrieval.reranker import rerank
    candidates = [
        {"chunk_id": 1, "text": "CD8+ T cells mediate antitumor immunity."},
        {"chunk_id": 2, "text": "mTOR signaling regulates cell growth."},
    ]
    with patch("retrieval.reranker._get_model", return_value=_mock_model([0.9, 0.5])):
        results = rerank("immune response", candidates)
    for r in results:
        assert "rerank_score" in r
        assert isinstance(r["rerank_score"], float)


def test_rerank_sorted_by_rerank_score_descending():
    from retrieval.reranker import rerank
    candidates = [
        {"chunk_id": 1, "text": "CD8+ T cells mediate antitumor immunity."},
        {"chunk_id": 2, "text": "mTOR signaling regulates cell growth."},
        {"chunk_id": 3, "text": "Daratumumab targets CD38 on myeloma cells."},
    ]
    # deliberately out-of-order scores to prove rerank() does the sorting, not the mock
    with patch("retrieval.reranker._get_model", return_value=_mock_model([0.2, 0.9, 0.5])):
        results = rerank("immune response", candidates)
    scores = [r["rerank_score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_rerank_empty_candidates_returns_empty():
    from retrieval.reranker import rerank
    with patch("retrieval.reranker._get_model") as mock_get_model:
        results = rerank("immune response", [])
    mock_get_model.assert_not_called()
    assert results == []


def test_rerank_preserves_all_candidate_fields():
    from retrieval.reranker import rerank
    candidates = [
        {
            "chunk_id": 1,
            "text": "CD8+ T cells mediate antitumor immunity.",
            "filename": "paper.pdf",
            "page_number": 3,
            "score": 0.031,
            "semantic_score": 0.85,
        }
    ]
    with patch("retrieval.reranker._get_model", return_value=_mock_model([0.7])):
        results = rerank("immune response", candidates)
    assert results[0]["filename"] == "paper.pdf"
    assert results[0]["semantic_score"] == 0.85
    assert results[0]["score"] == 0.031
