from sqlalchemy.orm import Session
from pgvector.sqlalchemy import Vector
from sqlalchemy import select
from rank_bm25 import BM25Okapi

from db.models import Chunk, Document
from ingestion.embedder import embed_query


def _tokenize(text: str) -> list[str]:
    """
    Lowercase whitespace tokenizer for BM25.
    Deliberately simple — no stemming, no stopwords.
    BM25 on biomedical text benefits from keeping full terms intact
    (e.g. 'daratumumab' must not be stemmed to 'daratumuma').
    """
    return text.lower().split()


def retrieve(
    query: str,
    session: Session,
    top_k: int = 5,
    namespace: str | None = None,
    candidate_pool: int = 20,
) -> list[dict]:
    """
    Hybrid retrieval: semantic search (pgvector) + BM25 reranking via RRF fusion.

    Args:
        query:          the user's natural language question
        session:        SQLAlchemy session
        top_k:          number of chunks to return after fusion
        namespace:      if provided, restrict retrieval to this namespace only
        candidate_pool: size of the semantic candidate pool before BM25 reranking
                        (must be >= top_k; typical values: 20-50)

    Returns:
        list of dicts with keys:
        {
            "chunk_id": int,
            "text": str,
            "score": float,        # RRF fusion score (higher = more relevant)
            "semantic_score": float,  # raw cosine similarity for debugging
            "document_id": int,
            "filename": str,
            "page_number": int,
            "chunk_index": int,
            "namespace": str,
        }
    """
    # ── 1. Embed the query ────────────────────────────────────────────────────

    query_vector = embed_query(query)

    # ── 2. Semantic search — fetch candidate pool (larger than final top_k) ──

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
        .limit(candidate_pool)
    )

    if namespace:
        stmt = stmt.where(Document.namespace == namespace)

    candidates = [dict(row) for row in session.execute(stmt).mappings().all()]

    if not candidates:
        return []

    # ── 3. BM25 over the candidate pool ──────────────────────────────────────

    tokenized_corpus = [_tokenize(c["text"]) for c in candidates]
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores = bm25.get_scores(_tokenize(query))

    # ── 4. RRF fusion ─────────────────────────────────────────────────────────
    # Each chunk gets a rank from semantic search (position in candidates list)
    # and a rank from BM25 (argsort of bm25_scores descending).
    # RRF score = 1/(k + rank_semantic) + 1/(k + rank_bm25)
    # k=60 is standard — dampens the weight of top-ranked results slightly,
    # making the fusion robust to one signal dominating.

    K = 60

    # semantic rank is simply the index in candidates (already sorted by cosine)
    semantic_ranks = {c["chunk_id"]: i for i, c in enumerate(candidates)}

    # bm25 rank: argsort descending → position of each candidate
    bm25_order = sorted(range(len(candidates)), key=lambda i: bm25_scores[i], reverse=True)
    bm25_ranks = {candidates[i]["chunk_id"]: rank for rank, i in enumerate(bm25_order)}

    for chunk in candidates:
        cid = chunk["chunk_id"]
        chunk["semantic_score"] = chunk.pop("score")  # rename raw score for clarity
        chunk["score"] = (
            1 / (K + semantic_ranks[cid]) +
            1 / (K + bm25_ranks[cid])
        )

    # ── 5. Sort by RRF score, return top_k ───────────────────────────────────

    ranked = sorted(candidates, key=lambda c: c["score"], reverse=True)
    return ranked[:top_k]