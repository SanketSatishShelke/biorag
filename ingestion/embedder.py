import os
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "NeuML/pubmedbert-base-embeddings")
_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """
    Lazy-load [load only when first needed, not at import time] the embedding
    model. Keeps startup fast when other modules are imported without needing
    embeddings.
    """
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of strings using PubMedBERT.
    Returns a list of vectors (one per input text).
    """
    model = get_model()
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,  # L2-normalize for cosine similarity via dot product
    )
    return embeddings.tolist()


def embed_query(query: str) -> list[float]:
    """
    Embed a single query string.
    Thin wrapper around embed_texts for the single-string case.
    """
    return embed_texts([query])[0]
