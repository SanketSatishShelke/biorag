import pytest
from unittest.mock import patch, MagicMock


def _make_nim_response(n_passages):
    """Fake NIM ranking response — returns passages in reverse order as a simple test."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "rankings": [
            {"index": i, "logit": float(n_passages - i)}
            for i in range(n_passages)
        ]
    }
    mock_response.raise_for_status.return_value = None
    return mock_response


def test_rerank_returns_list():
    from retrieval.reranker import rerank
    candidates = [
        {"chunk_id": 1, "text": "CD8+ T cells mediate antitumor immunity."},
        {"chunk_id": 2, "text": "mTOR signaling regulates cell growth."},
    ]
    with patch("retrieval.reranker.httpx.post", return_value=_make_nim_response(2)):
        results = rerank("immune response", candidates)
    assert isinstance(results, list)


def test_rerank_adds_rerank_score():
    from retrieval.reranker import rerank
    candidates = [
        {"chunk_id": 1, "text": "CD8+ T cells mediate antitumor immunity."},
        {"chunk_id": 2, "text": "mTOR signaling regulates cell growth."},
    ]
    with patch("retrieval.reranker.httpx.post", return_value=_make_nim_response(2)):
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
    with patch("retrieval.reranker.httpx.post", return_value=_make_nim_response(3)):
        results = rerank("immune response", candidates)
    scores = [r["rerank_score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_rerank_empty_candidates_returns_empty():
    from retrieval.reranker import rerank
    with patch("retrieval.reranker.httpx.post") as mock_post:
        results = rerank("immune response", [])
    mock_post.assert_not_called()
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
    with patch("retrieval.reranker.httpx.post", return_value=_make_nim_response(1)):
        results = rerank("immune response", candidates)
    assert results[0]["filename"] == "paper.pdf"
    assert results[0]["semantic_score"] == 0.85
    assert results[0]["score"] == 0.031
