import pytest
import os
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

from api.main import app, get_db
from db.models import Document, Chunk

load_dotenv()


@pytest.fixture(scope="module")
def db_engine():
    DATABASE_URL = (
        f"postgresql+psycopg://"
        f"{os.getenv('POSTGRES_USER', 'biorag')}:"
        f"{os.getenv('POSTGRES_PASSWORD', 'biorag_dev_password')}@"
        f"{os.getenv('POSTGRES_HOST', 'localhost')}:"
        f"{os.getenv('POSTGRES_PORT', '5432')}/"
        f"{os.getenv('POSTGRES_DB', 'biorag')}"
    )
    engine = create_engine(DATABASE_URL)
    yield engine
    engine.dispose()


@pytest.fixture(scope="function")
def db_session(db_engine):
    connection = db_engine.connect()
    transaction = connection.begin()
    Session = sessionmaker(bind=connection)
    session = Session()
    yield session
    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture(scope="function")
def client(db_session):
    """
    TestClient with get_db overridden to use the test session.
    scope="function" matches db_session scope — fresh session per test.
    """
    def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture(scope="module")
def sample_pdf_path():
    data_root = os.getenv("DATA_ROOT", "/mnt/data/biorag")
    raw_dir = os.path.join(data_root, "raw/tests")

    if not os.path.exists(raw_dir):
        pytest.skip(f"Test data directory not found: {raw_dir}")

    pdfs = [f for f in os.listdir(raw_dir) if f.endswith(".pdf")]
    if not pdfs:
        pytest.skip(f"No PDFs found in {raw_dir}")

    path = os.path.join(raw_dir, pdfs[0])
    if os.path.getsize(path) < 50000:
        pytest.skip(f"PDF too small — likely invalid: {path}")

    return path


def test_ingest_via_api_creates_document(client, db_session, sample_pdf_path):
    """
    Full integration: upload PDF via HTTP → verify Document row in DB.
    Tests the complete request lifecycle through FastAPI + dependency injection.
    """
    with open(sample_pdf_path, "rb") as f:
        response = client.post(
            "/ingest",
            files={"file": (os.path.basename(sample_pdf_path), f, "application/pdf")},
            data={"namespace": "api_test"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["document_id"] is not None

    # verify DB actually has the document
    doc = db_session.query(Document).filter_by(id=body["document_id"]).first()
    assert doc is not None
    assert doc.namespace == "api_test"


def test_ingest_via_api_creates_chunks(client, db_session, sample_pdf_path):
    """Verify chunks are created in DB after API ingest."""
    with open(sample_pdf_path, "rb") as f:
        response = client.post(
            "/ingest",
            files={"file": (os.path.basename(sample_pdf_path), f, "application/pdf")},
            data={"namespace": "api_test"},
        )

    assert response.status_code == 200
    doc_id = response.json()["document_id"]

    chunks = db_session.query(Chunk).filter_by(document_id=doc_id).all()
    assert len(chunks) > 0


def test_ingest_via_api_chunk_count_matches_response(client, db_session, sample_pdf_path):
    """chunk_count in response should match actual chunks in DB."""
    with open(sample_pdf_path, "rb") as f:
        response = client.post(
            "/ingest",
            files={"file": (os.path.basename(sample_pdf_path), f, "application/pdf")},
            data={"namespace": "api_test"},
        )

    body = response.json()
    db_count = db_session.query(Chunk).filter_by(
        document_id=body["document_id"]
    ).count()
    assert body["chunk_count"] == db_count


def test_ingest_rejects_non_pdf_integration(client):
    """Non-PDF upload should return 400 — no DB writes."""
    response = client.post(
        "/ingest",
        files={"file": ("document.txt", b"some text", "text/plain")},
        data={"namespace": "default"},
    )
    assert response.status_code == 400


def test_health_check_integration(client):
    """Health endpoint should return 200 in real environment."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
