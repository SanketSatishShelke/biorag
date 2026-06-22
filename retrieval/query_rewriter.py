import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

_client: OpenAI | None = None

REWRITE_PROMPT = """You are a biomedical search query optimizer.
Rewrite the user's question as a precise retrieval query for searching scientific literature.

Rules:
- Use specific biomedical terminology
- Expand abbreviations (AML → acute myeloid leukemia)
- Replace conversational phrasing with technical vocabulary
- Keep it concise — one to two sentences maximum
- Return ONLY the rewritten query, no explanation, no preamble

Examples:
User: "Does chemo work better with immunotherapy for blood cancer?"
Rewritten: "combination chemotherapy immunotherapy efficacy hematologic malignancies clinical outcomes"

User: "What does MYC do in AML?"
Rewritten: "MYC oncogene function mechanism acute myeloid leukemia transcriptional regulation"
"""


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=os.getenv("NIM_API_KEY"),
            base_url=os.getenv("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1"),
        )
    return _client


def rewrite_query(question: str) -> str:
    """
    Rewrite a conversational question as a precise biomedical retrieval query.

    Uses the same NIM LLM as generation. Called before retrieval — the
    rewritten query goes to pgvector + BM25, while the original question
    is preserved for the generator so the answer addresses what the user asked.

    Args:
        question: the user's original natural language question

    Returns:
        rewritten query string optimized for biomedical retrieval
    """
    client = _get_client()

    response = client.chat.completions.create(
        model=os.getenv("NIM_MODEL", "meta/llama-3.3-70b-instruct"),
        messages=[
            {"role": "system", "content": REWRITE_PROMPT},
            {"role": "user", "content": question},
        ],
        temperature=0.0,  # deterministic — query rewriting is not creative
        max_tokens=128,   # rewritten query should be short
    )

    rewritten = response.choices[0].message.content.strip()

    # fallback: if rewriter returns empty string, use original question
    return rewritten if rewritten else question
