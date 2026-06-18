from sqlalchemy.orm import Session
from pgvector.sqlalchemy import Vector
from sqlalchemy import select, text

from db.models import Chunk, Document
from ingestion.embedder import embed_query


def retrieve(
    query: str,
    session: Session,
    top_k: int = 5,
    namespace: str | None = None,
) -> list[dict]:
    """
    Retrieve the top-k most semantically similar chunks for a query.

    Args:
        query:     the user's natural language question
        session:   SQLAlchemy session
        top_k:     number of chunks to return
        namespace: if provided, restrict retrieval to this namespace only

    Returns:
        list of dicts with keys:
        {
            "chunk_id": int,
            "text": str,
            "score": float,        # cosine similarity (0-1, higher = more similar)
            "document_id": int,
            "filename": str,
            "page_number": int,
            "chunk_index": int,
            "namespace": str,
        }
    """
    # 1. Embed the query using the same model used at ingestion
    query_vector = embed_query(query)

    # 2. Build the similarity search query
    # cosine_distance = 1 - cosine_similarity, so we subtract from 1 for the score
    similarity = (1 - Chunk.embedding.cosine_distance(query_vector)).label("score")

    stmt = (
        select(
            Chunk.id.label("chunk_id"),
            Chunk.text,
            Chunk.page_number,
            Chunk.chunk_index,
            Chunk.document_id,
            Document.filename,
            Document.namespace,
            similarity,
        )
        .join(Document, Chunk.document_id == Document.id)
        .order_by(Chunk.embedding.cosine_distance(query_vector))
        .limit(top_k)
    )

    # 3. Filter by namespace if provided (ACL enforcement)
    if namespace:
        stmt = stmt.where(Document.namespace == namespace)

    results = session.execute(stmt).mappings().all()

    return [dict(row) for row in results]
