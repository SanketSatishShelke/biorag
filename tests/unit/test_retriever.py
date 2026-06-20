import pytest
from unittest.mock import patch, MagicMock
from retrieval.retriever import retrieve


def _make_mock_session(rows):
    session = MagicMock()
    session.execute.return_value.mappings.return_value.all.return_value = rows
    return session


def _candidate_rows():
    return [
        {
            "chunk_id": 1,
            "text": "CD8+ T cells mediate antitumor immunity through perforin.",
            "page_number": 3,
            "chunk_index": 5,
            "document_id": 1,
            "filename": "paper.pdf",
            "namespace": "default",
            "score": 0.85,
        },
        {
            "chunk_id": 2,
            "text": "Regulatory T cells suppress immune responses in tumors.",
            "page_number": 4,
            "chunk_index": 6,
            "document_id": 1,
            "filename": "paper.pdf",
            "namespace": "default",
            "score": 0.72,
        },
    ]


def _mock_rerank(query, candidates):
    """Fake reranker — adds rerank_score and returns candidates in same order."""
    for i, c in enumerate(candidates):
        c["rerank_score"] = float(len(candidates) - i)
    return sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)


def test_retrieve_returns_list():
    session = _make_mock_session(_candidate_rows())
    with patch("retrieval.retriever.embed_query", return_value=[0.1] * 768), \
         patch("retrieval.retriever.rerank", side_effect=_mock_rerank):
        results = retrieve("CD8+ T cells", session=session)
    assert isinstance(results, list)


def test_retrieve_returns_correct_number_of_results():
    session = _make_mock_session(_candidate_rows())
    with patch("retrieval.retriever.embed_query", return_value=[0.1] * 768), \
         patch("retrieval.retriever.rerank", side_effect=_mock_rerank):
        results = retrieve("CD8+ T cells", session=session, top_k=1)
    assert len(results) <= 1


def test_retrieve_result_has_required_keys():
    required_keys = {
        "chunk_id", "text", "score", "semantic_score", "rerank_score",
        "document_id", "filename", "page_number", "chunk_index", "namespace",
    }
    session = _make_mock_session(_candidate_rows())
    with patch("retrieval.retriever.embed_query", return_value=[0.1] * 768), \
         patch("retrieval.retriever.rerank", side_effect=_mock_rerank):
        results = retrieve("CD8+ T cells", session=session)
    for result in results:
        assert required_keys.issubset(result.keys())


def test_retrieve_calls_embed_query_once():
    session = _make_mock_session(_candidate_rows())
    with patch("retrieval.retriever.embed_query", return_value=[0.1] * 768) as mock_embed, \
         patch("retrieval.retriever.rerank", side_effect=_mock_rerank):
        retrieve("CD8+ T cells", session=session)
    mock_embed.assert_called_once_with("CD8+ T cells")


def test_retrieve_passes_query_to_embedder():
    query = "what are types of galectins?"
    session = _make_mock_session(_candidate_rows())
    with patch("retrieval.retriever.embed_query", return_value=[0.1] * 768) as mock_embed, \
         patch("retrieval.retriever.rerank", side_effect=_mock_rerank):
        retrieve(query, session=session)
    mock_embed.assert_called_once_with(query)


def test_retrieve_rrf_scores_are_positive():
    session = _make_mock_session(_candidate_rows())
    with patch("retrieval.retriever.embed_query", return_value=[0.1] * 768), \
         patch("retrieval.retriever.rerank", side_effect=_mock_rerank):
        results = retrieve("CD8+ T cells", session=session)
    for r in results:
        assert r["score"] > 0.0


def test_retrieve_results_ordered_by_rerank_score():
    """Final ordering is by rerank_score, not RRF score."""
    session = _make_mock_session(_candidate_rows())
    with patch("retrieval.retriever.embed_query", return_value=[0.1] * 768), \
         patch("retrieval.retriever.rerank", side_effect=_mock_rerank):
        results = retrieve("CD8+ T cells", session=session, top_k=2)
    scores = [r["rerank_score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_retrieve_empty_candidates_returns_empty_list():
    session = _make_mock_session([])
    with patch("retrieval.retriever.embed_query", return_value=[0.1] * 768), \
         patch("retrieval.retriever.rerank", side_effect=_mock_rerank):
        results = retrieve("CD8+ T cells", session=session)
    assert results == []
