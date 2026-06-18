import pytest
from unittest.mock import patch, MagicMock
from retrieval.retriever import retrieve


@pytest.fixture
def mock_session():
    """Mock SQLAlchemy session that returns fake query results."""
    session = MagicMock()

    # Simulate two result rows returned by session.execute()
    row1 = {
        "chunk_id": 1,
        "text": "CD8+ T cells mediate antitumor immunity through perforin.",
        "page_number": 3,
        "chunk_index": 5,
        "document_id": 1,
        "filename": "paper.pdf",
        "namespace": "default",
        "score": 0.85,
    }
    row2 = {
        "chunk_id": 2,
        "text": "Regulatory T cells suppress immune responses.",
        "page_number": 4,
        "chunk_index": 6,
        "document_id": 1,
        "filename": "paper.pdf",
        "namespace": "default",
        "score": 0.72,
    }

    # Chain the mock: session.execute().mappings().all() returns our fake rows
    session.execute.return_value.mappings.return_value.all.return_value = [
        row1, row2
    ]
    return session


def test_retrieve_returns_list(mock_session):
    """retrieve() should return a list."""
    with patch("retrieval.retriever.embed_query", return_value=[0.1] * 768):
        results = retrieve("CD8+ T cells", session=mock_session)
    assert isinstance(results, list)


def test_retrieve_returns_correct_number_of_results(mock_session):
    """retrieve() should return at most top_k results."""
    with patch("retrieval.retriever.embed_query", return_value=[0.1] * 768):
        results = retrieve("CD8+ T cells", session=mock_session, top_k=2)
    assert len(results) <= 2


def test_retrieve_result_has_required_keys(mock_session):
    """Each result dict must contain all required keys."""
    required_keys = {
        "chunk_id", "text", "score", "document_id",
        "filename", "page_number", "chunk_index", "namespace"
    }
    with patch("retrieval.retriever.embed_query", return_value=[0.1] * 768):
        results = retrieve("CD8+ T cells", session=mock_session)
    for result in results:
        assert required_keys.issubset(result.keys())


def test_retrieve_calls_embed_query_once(mock_session):
    """embed_query should be called exactly once per retrieve() call."""
    with patch("retrieval.retriever.embed_query", return_value=[0.1] * 768) as mock_embed:
        retrieve("CD8+ T cells", session=mock_session)
    mock_embed.assert_called_once_with("CD8+ T cells")


def test_retrieve_passes_query_to_embedder(mock_session):
    """The exact query string should be passed to embed_query."""
    query = "what are types of galectins?"
    with patch("retrieval.retriever.embed_query", return_value=[0.1] * 768) as mock_embed:
        retrieve(query, session=mock_session)
    mock_embed.assert_called_once_with(query)
