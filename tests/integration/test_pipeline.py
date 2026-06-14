import pytest
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

from db.base import Base
from db.models import Document, Chunk
from ingestion.pipeline import ingest_pdf

load_dotenv()

# --- fixtures ---

@pytest.fixture(scope="module")
def db_engine():
    """
    Create a real engine pointing at the test database.
    scope="module" means this fixture is created once per test file,
    not once per test — engine creation is expensive, reuse it.
    """
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
    engine.dispose()  # close all connections in pool when module tests finish


@pytest.fixture(scope="function")
def db_session(db_engine):
    """
    Provide a session that rolls back after each test.
    scope="function" means a fresh session per test — guarantees isolation.
    Uses a transaction rollback pattern so no test data persists.
    """
    connection = db_engine.connect()
    transaction = connection.begin()
    Session = sessionmaker(bind=connection)
    session = Session()

    yield session

    session.close()
    transaction.rollback()  # wipe everything the test wrote
    connection.close()


@pytest.fixture(scope="module")
def sample_pdf_path():
    """
    Path to a real PDF for integration testing.
    Reads from DATA_ROOT env var — same place ingestion looks.
    Skips all integration tests if no PDF is found.
    """
    data_root = os.getenv("DATA_ROOT", "/mnt/data/biorag")
    raw_dir = os.path.join(data_root, "raw/tests")

    if not os.path.exists(raw_dir):
        pytest.skip(f"Raw data directory not found: {raw_dir}")

    pdfs = [f for f in os.listdir(raw_dir) if f.endswith(".pdf")]
    if not pdfs:
        pytest.skip("No PDF files found in raw directory")

    return os.path.join(raw_dir, pdfs[0])


# --- tests ---

def test_ingest_pdf_creates_document(db_session, sample_pdf_path):
    """Ingesting a PDF should create exactly one Document row."""
    doc = ingest_pdf(sample_pdf_path, session=db_session)

    result = db_session.query(Document).filter_by(id=doc.id).first()
    assert result is not None
    assert result.filename == os.path.basename(sample_pdf_path)


def test_ingest_pdf_creates_chunks(db_session, sample_pdf_path):
    """Ingesting a PDF should create at least one Chunk row."""
    doc = ingest_pdf(sample_pdf_path, session=db_session)

    chunks = db_session.query(Chunk).filter_by(document_id=doc.id).all()
    assert len(chunks) > 0


def test_ingest_pdf_chunks_have_embeddings(db_session, sample_pdf_path):
    """Every chunk should have a non-null embedding of correct dimension."""
    doc = ingest_pdf(sample_pdf_path, session=db_session)

    chunks = db_session.query(Chunk).filter_by(document_id=doc.id).all()
    for chunk in chunks:
        assert chunk.embedding is not None
        assert len(chunk.embedding) == 768


def test_ingest_pdf_chunk_indices_sequential(db_session, sample_pdf_path):
    """Chunk indices should be sequential starting from 0."""
    doc = ingest_pdf(sample_pdf_path, session=db_session)

    chunks = (
        db_session.query(Chunk)
        .filter_by(document_id=doc.id)
        .order_by(Chunk.chunk_index)
        .all()
    )
    indices = [c.chunk_index for c in chunks]
    assert indices == list(range(len(chunks)))


def test_ingest_pdf_default_namespace(db_session, sample_pdf_path):
    """Document should have namespace='default' when not specified."""
    doc = ingest_pdf(sample_pdf_path, session=db_session)

    result = db_session.query(Document).filter_by(id=doc.id).first()
    assert result.namespace == "default"


def test_ingest_pdf_custom_namespace(db_session, sample_pdf_path):
    """Document should store the namespace passed at ingestion time."""
    doc = ingest_pdf(sample_pdf_path, session=db_session, namespace="oncology")

    result = db_session.query(Document).filter_by(id=doc.id).first()
    assert result.namespace == "oncology"


def test_ingest_pdf_rollback_on_failure(db_session):
    """Ingesting a non-existent file should raise and leave DB clean."""
    with pytest.raises((FileNotFoundError, Exception)):
        ingest_pdf("/non/existent/file.pdf", session=db_session)

    # no documents should have been committed
    count = db_session.query(Document).count()
    assert count == 0