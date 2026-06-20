from sqlalchemy.orm import Session
from pgvector.sqlalchemy import Vector
from sqlalchemy import select
from rank_bm25 import BM25Okapi

from db.models import Chunk, Document
from ingestion.embedder import embed_query
from retrieval.reranker import rerank


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
    Hybrid retrieval: semantic search + BM25 + RRF fusion + cross-encoder reranking.

    Pipeline:
        1. Embed query with PubMedBERT
        2. Fetch top candidate_pool chunks by cosine similarity (pgvector)
        3. Score same candidates with BM25 (in-memory, rank-bm25)
        4. Fuse rankings with RRF (k=60)
        5. Rerank fused candidates with cross-encoder (NIM rerank-qa-mistral-4b)
        6. Return top_k

    Args:
        query:          the user's natural language question
        session:        SQLAlchemy session
        top_k:          number of chunks to return after reranking
        namespace:      if provided, restrict retrieval to this namespace only
        candidate_pool: size of the semantic candidate pool (must be >= top_k)

    Returns:
        list of dicts with keys:
        {
            "chunk_id": int,
            "text": str,
            "score": float,         # RRF fusion score
            "semantic_score": float, # raw cosine similarity
            "rerank_score": float,   # cross-encoder logit (higher = more relevant)
            "document_id": int,
            "filename": str,
            "page_number": int,
            "chunk_index": int,
            "namespace": str,
        }
    """
    # ── 1. Embed the query ────────────────────────────────────────────────────

    query_vector = embed_query(query)

    # ── 2. Semantic search — fetch candidate pool ─────────────────────────────

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

    K = 60

    semantic_ranks = {c["chunk_id"]: i for i, c in enumerate(candidates)}

    bm25_order = sorted(range(len(candidates)), key=lambda i: bm25_scores[i], reverse=True)
    bm25_ranks = {candidates[i]["chunk_id"]: rank for rank, i in enumerate(bm25_order)}

    for chunk in candidates:
        cid = chunk["chunk_id"]
        chunk["semantic_score"] = chunk.pop("score")
        chunk["score"] = (
            1 / (K + semantic_ranks[cid]) +
            1 / (K + bm25_ranks[cid])
        )

    # ── 5. Cross-encoder reranking ────────────────────────────────────────────

    reranked = rerank(query, candidates)

    # ── 6. Return top_k ───────────────────────────────────────────────────────

    return reranked[:top_k]
