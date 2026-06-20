import pytest
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

from db.models import Document, Chunk
from ingestion.pipeline import ingest_pdf
from retrieval.retriever import retrieve

load_dotenv()

pytestmark = pytest.mark.skipif(
    os.getenv("CI") == "true",
    reason="Skipping live NIM API call in CI — vcrpy cassettes coming in Phase 3"
)


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


@pytest.fixture(scope="module")
def ingested_document(db_engine):
    """
    Ingest a real PDF once for the entire module.
    Cleans up after all tests in the module complete.
    """
    data_root = os.getenv("DATA_ROOT", "/mnt/data/biorag")
    raw_dir = os.path.join(data_root, "raw/tests")

    if not os.path.exists(raw_dir):
        pytest.skip(f"Test data directory not found: {raw_dir}")

    pdfs = [f for f in os.listdir(raw_dir) if f.endswith(".pdf")]
    if not pdfs:
        pytest.skip(f"No PDFs found in {raw_dir}")

    pdf_path = os.path.join(raw_dir, pdfs[0])

    Session = sessionmaker(bind=db_engine)
    session = Session()

    try:
        doc = ingest_pdf(pdf_path, session=session, namespace="test_retrieval")
        doc_id = doc.id
        yield doc_id
    finally:
        session.query(Chunk).filter_by(document_id=doc_id).delete()
        session.query(Document).filter_by(id=doc_id).delete()
        session.commit()
        session.close()


@pytest.fixture(scope="function")
def db_session(db_engine):
    Session = sessionmaker(bind=db_engine)
    session = Session()
    yield session
    session.close()


def test_retrieve_returns_results(db_session, ingested_document):
    """A biomedical query should return at least one result."""
    results = retrieve(
        "immune response tumor microenvironment",
        session=db_session,
        top_k=3,
        namespace="test_retrieval",
    )
    assert len(results) > 0


def test_retrieve_scores_are_positive(db_session, ingested_document):
    """
    RRF scores are small positive floats — not bounded to 0-1.
    Max theoretical RRF score with two signals is 2/(k+0) = ~0.033 at k=60.
    """
    results = retrieve(
        "immune response tumor microenvironment",
        session=db_session,
        top_k=5,
        namespace="test_retrieval",
    )
    for r in results:
        assert r["score"] > 0.0


def test_retrieve_semantic_scores_between_zero_and_one(db_session, ingested_document):
    """Raw cosine similarity scores (semantic_score) must be bounded 0-1."""
    results = retrieve(
        "immune response tumor microenvironment",
        session=db_session,
        top_k=5,
        namespace="test_retrieval",
    )
    for r in results:
        assert 0.0 <= r["semantic_score"] <= 1.0


def test_retrieve_results_ordered_by_score(db_session, ingested_document):
    """Results should be returned in descending rerank score order."""
    results = retrieve(
        "immune response tumor microenvironment",
        session=db_session,
        top_k=5,
        namespace="test_retrieval",
    )
    scores = [r["rerank_score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_retrieve_namespace_isolation(db_session, ingested_document):
    """Querying a different namespace should return no results from test data."""
    results = retrieve(
        "immune response tumor microenvironment",
        session=db_session,
        top_k=5,
        namespace="completely_different_namespace",
    )
    doc_ids = [r["document_id"] for r in results]
    assert ingested_document not in doc_ids

def test_retrieve_result_shape(db_session, ingested_document):
    """Each result should have all required fields with correct types."""
    results = retrieve(
        "immune response",
        session=db_session,
        top_k=1,
        namespace="test_retrieval",
    )
    assert len(results) == 1
    r = results[0]
    assert isinstance(r["chunk_id"], int)
    assert isinstance(r["text"], str)
    assert isinstance(r["score"], float)
    assert isinstance(r["semantic_score"], float)
    assert isinstance(r["rerank_score"], float)
    assert isinstance(r["document_id"], int)
    assert isinstance(r["filename"], str)
    assert isinstance(r["page_number"], int)
    assert isinstance(r["chunk_index"], int)
    assert isinstance(r["namespace"], str)

def test_retrieve_top_k_respected(db_session, ingested_document):
    """retrieve() should return at most top_k results."""
    results = retrieve(
        "immune response",
        session=db_session,
        top_k=3,
        namespace="test_retrieval",
    )
    assert len(results) <= 3
