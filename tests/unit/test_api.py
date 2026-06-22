import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from api.main import app, get_db


# --- dependency override ---

def make_mock_db():
    mock_session = MagicMock()
    mock_session.execute.return_value.scalar.return_value = 5
    return mock_session


@pytest.fixture
def client():
    mock_db = make_mock_db()

    def override_get_db():
        yield mock_db

    app.dependency_overrides[get_db] = override_get_db
    yield TestClient(app), mock_db
    app.dependency_overrides.clear()


def _mock_chunk():
    return {
        "chunk_id": 1,
        "text": "Galectins regulate tumor immunity.",
        "page_number": 3,
        "chunk_index": 0,
        "document_id": 1,
        "filename": "paper.pdf",
        "namespace": "default",
        "score": 0.85,
    }


# --- health check ---

def test_health_check():
    with TestClient(app) as c:
        response = c.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# --- ingest endpoint ---

def test_ingest_rejects_non_pdf(client):
    test_client, _ = client
    response = test_client.post(
        "/ingest",
        files={"file": ("document.txt", b"some text content", "text/plain")},
        data={"namespace": "default"},
    )
    assert response.status_code == 400
    assert "PDF" in response.json()["detail"]


def test_ingest_uses_injected_db_session(client):
    test_client, mock_db = client

    with patch("api.main.ingest_pdf") as mock_ingest:
        mock_doc = MagicMock()
        mock_doc.id = 42
        mock_doc.filename = "test.pdf"
        mock_doc.namespace = "default"
        mock_ingest.return_value = mock_doc

        test_client.post(
            "/ingest",
            files={"file": ("test.pdf", b"%PDF-1.4 fake pdf content", "application/pdf")},
            data={"namespace": "default"},
        )

    call_kwargs = mock_ingest.call_args.kwargs
    assert call_kwargs["session"] is mock_db


def test_ingest_returns_correct_shape(client):
    test_client, _ = client

    with patch("api.main.ingest_pdf") as mock_ingest:
        mock_doc = MagicMock()
        mock_doc.id = 42
        mock_doc.filename = "test.pdf"
        mock_doc.namespace = "default"
        mock_ingest.return_value = mock_doc

        response = test_client.post(
            "/ingest",
            files={"file": ("test.pdf", b"%PDF-1.4 fake pdf content", "application/pdf")},
            data={"namespace": "default"},
        )

    assert response.status_code == 200
    body = response.json()
    assert "document_id" in body
    assert "filename" in body
    assert "namespace" in body
    assert "chunk_count" in body
    assert "message" in body


# --- query endpoint ---

def test_query_rejects_empty_question(client):
    test_client, _ = client
    response = test_client.post(
        "/query",
        data={"question": "   ", "namespace": "default", "top_k": 3},
    )
    assert response.status_code == 400


def test_query_uses_injected_db_session(client):
    test_client, mock_db = client

    with patch("api.main.rewrite_query", return_value="rewritten query"), \
         patch("api.main.retrieve") as mock_retrieve, \
         patch("api.main.generate") as mock_generate:

        mock_retrieve.return_value = [_mock_chunk()]
        mock_generate.return_value = {
            "answer": "Galectins suppress T cells.",
            "sources": mock_retrieve.return_value,
        }

        test_client.post(
            "/query",
            data={"question": "What do galectins do?", "namespace": "default", "top_k": 3},
        )

    call_kwargs = mock_retrieve.call_args.kwargs
    assert call_kwargs["session"] is mock_db


def test_query_returns_correct_shape(client):
    """Query response must contain answer, sources, chunks_retrieved, retrieval_query."""
    test_client, _ = client

    with patch("api.main.rewrite_query", return_value="galectin tumor immunity mechanism"), \
         patch("api.main.retrieve") as mock_retrieve, \
         patch("api.main.generate") as mock_generate:

        mock_retrieve.return_value = [_mock_chunk()]
        mock_generate.return_value = {
            "answer": "Galectins suppress T cells.",
            "sources": mock_retrieve.return_value,
        }

        response = test_client.post(
            "/query",
            data={"question": "What do galectins do?", "namespace": "default", "top_k": 3},
        )

    assert response.status_code == 200
    body = response.json()
    assert "answer" in body
    assert "sources" in body
    assert "chunks_retrieved" in body
    assert "retrieval_query" in body


def test_query_retrieval_uses_rewritten_query(client):
    """
    retrieve() must be called with the rewritten query, not the original question.
    The generator must receive the original question.
    """
    test_client, _ = client

    with patch("api.main.rewrite_query", return_value="galectin tumor immunity T cell suppression") as mock_rewrite, \
         patch("api.main.retrieve") as mock_retrieve, \
         patch("api.main.generate") as mock_generate:

        mock_retrieve.return_value = [_mock_chunk()]
        mock_generate.return_value = {
            "answer": "Galectins suppress T cells.",
            "sources": mock_retrieve.return_value,
        }

        test_client.post(
            "/query",
            data={"question": "What do galectins do?", "namespace": "default", "top_k": 3},
        )

    # rewriter called with original question
    mock_rewrite.assert_called_once_with("What do galectins do?")
    # retriever called with rewritten query
    assert mock_retrieve.call_args.args[0] == "galectin tumor immunity T cell suppression"
    # generator called with original question
    assert mock_generate.call_args.args[0] == "What do galectins do?"


def test_query_returns_cannot_find_when_no_chunks(client):
    test_client, _ = client

    with patch("api.main.rewrite_query", return_value="dark matter composition physics"), \
         patch("api.main.retrieve") as mock_retrieve:

        mock_retrieve.return_value = []

        response = test_client.post(
            "/query",
            data={"question": "What is dark matter?", "namespace": "default", "top_k": 3},
        )

    assert response.status_code == 200
    assert "cannot find" in response.json()["answer"].lower()
    assert response.json()["chunks_retrieved"] == 0


def test_query_exposes_retrieval_query_in_response(client):
    """retrieval_query in response should be the rewritten form, not the original."""
    test_client, _ = client

    rewritten = "galectin lectin tumor microenvironment immunosuppression"

    with patch("api.main.rewrite_query", return_value=rewritten), \
         patch("api.main.retrieve") as mock_retrieve, \
         patch("api.main.generate") as mock_generate:

        mock_retrieve.return_value = [_mock_chunk()]
        mock_generate.return_value = {
            "answer": "Galectins suppress T cells.",
            "sources": mock_retrieve.return_value,
        }

        response = test_client.post(
            "/query",
            data={"question": "What do galectins do?", "namespace": "default", "top_k": 3},
        )

    assert response.json()["retrieval_query"] == rewritten
