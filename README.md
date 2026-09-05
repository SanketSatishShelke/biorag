# BioRAG

An enterprise-grade Retrieval-Augmented Generation (RAG) system for biomedical and pharmaceutical scientific literature, built as a demonstration of production-credible RAG architecture — retrieval quality, evaluation, guardrails, and monitoring — with tradeoff reasoning documented at every architectural decision point.

## Pipeline

```
input guardrail → query rewriter → hybrid search (pgvector + BM25 + RRF) → cross-encoder reranker → confidence check → generator
```

1. **Input guardrail** — rejects prompt injection and out-of-scope queries before they reach retrieval
2. **Query rewriter** — converts natural-language questions into precise biomedical retrieval queries
3. **Hybrid search** — dense retrieval (pgvector cosine similarity) fused with sparse lexical retrieval (BM25) via Reciprocal Rank Fusion (RRF)
4. **Cross-encoder reranker** — rescores fused candidates using full query-passage cross-attention, capturing relevance signals (negation, specificity, conditional relationships) that fusion scores alone cannot
5. **Confidence check** — refuses to answer when the top reranked result falls below a relevance threshold, rather than generating from weak evidence
6. **Generator** — produces a cited answer grounded in the retrieved context

## Tech stack

| Layer | Tool |
|---|---|
| Language / deps | Python 3.11, `uv` + `pyproject.toml` |
| API layer | FastAPI + uvicorn |
| Vector DB | Postgres 16 + pgvector (Docker) |
| Embeddings | `NeuML/pubmedbert-base-embeddings` (local CPU, 768-dim) — domain-tuned, not a generic embedding model |
| LLM inference | NVIDIA NIM (OpenAI-compatible API) |
| Reranker | Local cross-encoder (`sentence-transformers`, `cross-encoder/ms-marco-MiniLM-L-6-v2`) — self-hosted, no external API dependency |
| ORM / migrations | SQLAlchemy 2.0 + Alembic |
| PDF parsing | PyMuPDF |
| Lexical retrieval | `rank-bm25` |
| Testing | pytest, unit + integration coverage across all modules |
| CI | GitHub Actions (pgvector service container, HuggingFace model cache) |

## Project status

**Phase 1 — end-to-end skeleton:** ✅ Complete. Docker Compose + pgvector, SQLAlchemy models, Alembic migrations, PDF ingestion pipeline, PubMedBERT embedder, FastAPI (`/health`, `/ingest`, `/query`), and a terminal CLI (`ui/cli.py`) for query-only interaction — a REPL client that holds conversation history in memory and folds recent turns into follow-up questions, giving the stateless `/query` endpoint multi-turn context without backend changes.

**Phase 2 — retrieval quality:** ✅ Complete. BM25 + pgvector hybrid search via RRF fusion, cross-encoder reranking, query rewriting.

**Phase 3 — production safety:** ✅ Complete. Input guardrails (prompt injection + out-of-scope detection), confidence scoring, structured refusal responses.

**Phase 4 — cost/reliability:** Not started. Planned: Redis semantic cache, model router with fallback, async ingestion via Celery, prompt registry, and agentic additions (multi-hop query decomposition, ClinicalTrials.gov live API routing, self-correction on low-confidence retrievals).

**Phase 5 — evaluation:** Not started. Planned: retrieval metrics (precision/recall/NDCG), LLM-as-judge faithfulness scoring, cost dashboard, drift detection.

**Phase 6 — architectural upgrade:** Not started. Planned migration from pgvector to self-hosted Qdrant.

## Known deviations from original spec (deliberate)

- PyMuPDF only for PDF parsing — no dedicated table extraction yet (flattening tables to linear text destroys row/column relational meaning; Docling flagged as the leading candidate for structure-aware parsing)
- `NeuML/pubmedbert-base-embeddings` instead of originally scoped BiomedBERT variants
- NVIDIA NIM instead of OpenAI API for LLM inference
- Local cross-encoder reranker instead of a hosted NIM/OpenRouter rerank endpoint — adopted after repeated hosted free-tier reranking endpoints were deprecated with no notice; removes an external dependency from the retrieval hot path entirely
- Synchronous ingestion instead of async/Celery (planned for Phase 4)
- Query-only terminal CLI instead of a web UI — enterprise ingestion is a backend batch operation, not something end users trigger through a chat interface

## Known limitations

- Input guardrail classification (LLM-based, temperature=0.0) is not perfectly reliable — occasional false negatives observed during manual testing where out-of-scope queries were not rejected. Flagged as a Phase 5 evaluation target (measure guardrail precision/recall against a labeled test set) rather than patched ad hoc.

## Development

```bash
uv sync --extra dev
docker compose up -d          # starts pgvector
uvicorn api.main:app --reload # starts the API
python ui/cli.py              # terminal query client
```
