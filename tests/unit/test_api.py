import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
import io

from api.main import app, get_db


# --- dependency override ---

def make_mock_db():
    """
    Create a mock database session.
    Returns a mock that satisfies SQLAlchemy session interface.
    """
    mock_session = MagicMock()
    # mock chunk count query for /ingest endpoint
    mock_session.execute.return_value.scalar.return_value = 5
    return mock_session


@pytest.fixture
def client():
    """
    TestClient with get_db dependency overridden.
    This is the key fixture — proves dependency injection works:
    if Depends(get_db) is wired correctly, our mock session
    is what the endpoint receives, not a real DB connection.
    """
    mock_db = make_mock_db()

    def override_get_db():
        yield mock_db

    app.dependency_overrides[get_db] = override_get_db
    yield TestClient(app), mock_db
    app.dependency_overrides.clear()  # reset after test — don't leak overrides


# --- health check ---

def test_health_check():
    """Health endpoint should return 200 and status ok."""
    with TestClient(app) as c:
        response = c.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# --- ingest endpoint ---

def test_ingest_rejects_non_pdf(client):
    """Non-PDF files should return 400."""
    test_client, _ = client
    response = test_client.post(
        "/ingest",
        files={"file": ("document.txt", b"some text content", "text/plain")},
        data={"namespace": "default"},
    )
    assert response.status_code == 400
    assert "PDF" in response.json()["detail"]


def test_ingest_uses_injected_db_session(client):
    """
    Core dependency injection test:
    verify the endpoint uses the injected mock session,
    not a real database connection.
    """
    test_client, mock_db = client

    with patch("api.main.ingest_pdf") as mock_ingest:
        # mock ingest_pdf to return a fake document
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

    # verify ingest_pdf was called with the mock session, not a real one
    call_kwargs = mock_ingest.call_args.kwargs
    assert call_kwargs["session"] is mock_db


def test_ingest_returns_correct_shape(client):
    """Successful ingest should return document_id, filename, namespace, chunk_count, message."""
    test_client, mock_db = client

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
    """Empty question should return 400."""
    test_client, _ = client
    response = test_client.post(
        "/query",
        data={"question": "   ", "namespace": "default", "top_k": 3},
    )
    assert response.status_code == 400


def test_query_uses_injected_db_session(client):
    """
    Core dependency injection test for /query:
    verify retrieve() receives the injected mock session.
    """
    test_client, mock_db = client

    with patch("api.main.retrieve") as mock_retrieve, \
         patch("api.main.generate") as mock_generate:

        mock_retrieve.return_value = [
            {
                "chunk_id": 1,
                "text": "Galectins regulate tumor immunity.",
                "page_number": 3,
                "chunk_index": 0,
                "document_id": 1,
                "filename": "paper.pdf",
                "namespace": "default",
                "score": 0.85,
            }
        ]
        mock_generate.return_value = {
            "answer": "Galectins suppress T cells [paper.pdf, p.3].",
            "sources": mock_retrieve.return_value,
        }

        response = test_client.post(
            "/query",
            data={"question": "What do galectins do?", "namespace": "default", "top_k": 3},
        )

    # verify retrieve was called with the mock session
    call_kwargs = mock_retrieve.call_args.kwargs
    assert call_kwargs["session"] is mock_db


def test_query_returns_correct_shape(client):
    """Query response must contain answer, sources, chunks_retrieved."""
    test_client, mock_db = client

    with patch("api.main.retrieve") as mock_retrieve, \
         patch("api.main.generate") as mock_generate:

        mock_retrieve.return_value = [
            {
                "chunk_id": 1,
                "text": "Galectins regulate tumor immunity.",
                "page_number": 3,
                "chunk_index": 0,
                "document_id": 1,
                "filename": "paper.pdf",
                "namespace": "default",
                "score": 0.85,
            }
        ]
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


def test_query_returns_cannot_find_when_no_chunks(client):
    """When retrieval returns no chunks, response should indicate no results found."""
    test_client, _ = client

    with patch("api.main.retrieve") as mock_retrieve:
        mock_retrieve.return_value = []

        response = test_client.post(
            "/query",
            data={"question": "What is dark matter?", "namespace": "default", "top_k": 3},
        )

    assert response.status_code == 200
    assert "cannot find" in response.json()["answer"].lower()
    assert response.json()["chunks_retrieved"] == 0
