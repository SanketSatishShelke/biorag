import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

_client: OpenAI | None = None

SYSTEM_PROMPT_PATH = os.path.join(
    os.path.dirname(__file__), "prompts", "system_prompt.txt"
)


def _get_client() -> OpenAI:
    """
    Lazy-load the OpenAI-compatible NIM client.
    Uses same lazy pattern as embedder — don't initialize at import time.
    """
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=os.getenv("NIM_API_KEY"),
            base_url=os.getenv("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1"),
        )
    return _client


def _load_system_prompt() -> str:
    with open(SYSTEM_PROMPT_PATH, "r") as f:
        return f.read().strip()


def _build_context_block(chunks: list[dict]) -> str:
    """
    Format retrieved chunks into a numbered context block for the prompt.
    Each chunk is labelled with its source so the model can cite it.
    """
    lines = []
    for i, chunk in enumerate(chunks, start=1):
        lines.append(
            f"[{i}] Source: {chunk['filename']}, Page {chunk['page_number']}\n"
            f"{chunk['text']}"
        )
    return "\n\n".join(lines)


def generate(
    query: str,
    chunks: list[dict],
    model: str | None = None,
) -> dict:
    """
    Generate a cited answer from retrieved chunks.

    Args:
        query:  the user's question
        chunks: retrieved chunks from retriever.retrieve()
        model:  NIM model string, defaults to NIM_MODEL env var

    Returns:
        {
            "answer": str,       # the generated answer with inline citations
            "sources": list[dict] # the chunks used as context
        }
    """
    if not chunks:
        return {
            "answer": "I cannot find information about this in the provided literature.",
            "sources": []
        }

    client = _get_client()
    system_prompt = _load_system_prompt()
    context_block = _build_context_block(chunks)
    model_name = model or os.getenv("NIM_MODEL", "meta/llama-3.3-70b-instruct")

    user_message = (
        f"Answer the following question using only the provided research excerpts.\n\n"
        f"Research excerpts:\n{context_block}\n\n"
        f"Question: {query}"
    )

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0.1,  # low temperature for factual, deterministic responses
        max_tokens=1024,
    )

    answer = response.choices[0].message.content

    return {
        "answer": answer,
        "sources": chunks,
    }
