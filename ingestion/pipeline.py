import os
from sqlalchemy.orm import Session

from ingestion.pdf_parser import extract_pages
from ingestion.chunker import chunk_pages
from ingestion.embedder import embed_texts
from db.models import Document, Chunk


def ingest_pdf(
    pdf_path: str,
    session: Session,
    namespace: str = "default",
) -> Document:
    """
    Full ingestion pipeline for a single PDF:
    parse → chunk → embed → write to DB.

    Args:
        pdf_path:  absolute path to the PDF file
        session:   SQLAlchemy session (injected by caller)
        namespace: ACL namespace this document belongs to

    Returns:
        The created Document ORM object (with id populated after commit)
    """
    filename = os.path.basename(pdf_path)

    # --- 1. Parse ---
    pages = extract_pages(pdf_path)

    # --- 2. Chunk ---
    chunks = chunk_pages(pages)

    if not chunks:
        raise ValueError(f"No chunks produced from {filename} — file may be empty or image-only")

    # --- 3. Embed ---
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)

    # --- 4. Write to DB ---
    document = Document(filename=filename, namespace=namespace)
    session.add(document)
    session.flush()  # assigns document.id without committing the transaction

    chunk_objects = [
        Chunk(
            document_id=document.id,
            text=c["text"],
            page_number=c["page_number"],
            chunk_index=c["chunk_index"],
            embedding=embeddings[i],
        )
        for i, c in enumerate(chunks)
    ]
    session.add_all(chunk_objects)
    session.commit()

    return document
