import os
import shutil
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, Form
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from dotenv import load_dotenv

from db.base import SessionLocal
from ingestion.pipeline import ingest_pdf
from retrieval.retriever import retrieve
from generation.generator import generate

load_dotenv()

app = FastAPI(
    title="BioRAG",
    description="RAG system for biomedical and pharmaceutical literature",
    version="0.1.0"
)

DATA_ROOT = os.getenv("DATA_ROOT", "/mnt/data/biorag")
UPLOAD_DIR = Path(DATA_ROOT) / "raw" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# --- dependency ---

def get_db():
    """
    FastAPI dependency that provides a database session per request.
    Guarantees session is closed after request completes, even if an
    exception is raised — the finally block always runs.
    """
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


# --- endpoints ---

@app.get("/health")
def health():
    """Liveness check — confirms API is running."""
    return {"status": "ok"}


@app.post("/ingest")
def ingest_document(
    file: UploadFile = File(...),
    namespace: str = Form(default="default"),
    db: Session = Depends(get_db),
):
    """
    Upload and ingest a PDF into BioRAG.

    - Saves the uploaded PDF to DATA_ROOT/raw/uploads/
    - Parses, chunks, embeds, and stores it in pgvector
    - Returns document ID and chunk count

    Args:
        file:      the PDF file to ingest
        namespace: ACL namespace for this document (default: "default")
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    save_path = UPLOAD_DIR / file.filename

    # save uploaded file to disk
    with save_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        doc = ingest_pdf(str(save_path), session=db, namespace=namespace)
    except ValueError as e:
        # ingest_pdf raises ValueError for empty/unparseable PDFs
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

    chunk_count = db.execute(
        __import__("sqlalchemy").text(
            "SELECT COUNT(*) FROM chunks WHERE document_id = :doc_id"
        ),
        {"doc_id": doc.id}
    ).scalar()

    return {
        "document_id": doc.id,
        "filename": doc.filename,
        "namespace": doc.namespace,
        "chunk_count": chunk_count,
        "message": f"Successfully ingested {doc.filename}"
    }


@app.post("/query")
def query_documents(
    question: str = Form(...),
    namespace: str = Form(default="default"),
    top_k: int = Form(default=5),
    db: Session = Depends(get_db),
):
    """
    Query the BioRAG system with a natural language question.

    - Embeds the question using PubMedBERT
    - Retrieves top_k most similar chunks from pgvector
    - Generates a cited answer using the NIM LLM

    Args:
        question:  natural language question
        namespace: ACL namespace to search within
        top_k:     number of chunks to retrieve (default: 5)
    """
    if not question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    chunks = retrieve(question, session=db, top_k=top_k, namespace=namespace)

    if not chunks:
        return {
            "answer": "I cannot find information about this in the provided literature.",
            "sources": [],
            "chunks_retrieved": 0,
        }

    result = generate(question, chunks)

    return {
        "answer": result["answer"],
        "sources": [
            {
                "filename": c["filename"],
                "page_number": c["page_number"],
                "score": round(c["score"], 4),
                "text_preview": c["text"][:200],
            }
            for c in result["sources"]
        ],
        "chunks_retrieved": len(chunks),
    }
