import pytest
from unittest.mock import patch, MagicMock
from generation.generator import generate, _build_context_block


# --- helpers ---

def make_chunks(n: int = 2) -> list[dict]:
    """Build fake retrieval results as retriever.retrieve() would return."""
    return [
        {
            "chunk_id": i,
            "text": f"Galectin-{i} regulates T cell apoptosis in tumor microenvironment.",
            "page_number": i + 1,
            "chunk_index": i,
            "document_id": 1,
            "filename": "galectin_paper.pdf",
            "namespace": "default",
            "score": 0.9 - (i * 0.1),
        }
        for i in range(1, n + 1)
    ]


def make_mock_response(content: str) -> MagicMock:
    """
    Build a mock that mimics the OpenAI API response structure:
    response.choices[0].message.content
    """
    mock_response = MagicMock()
    mock_response.choices[0].message.content = content
    return mock_response


# --- unit tests ---

def test_generate_returns_answer_and_sources():
    """generate() must return dict with 'answer' and 'sources' keys."""
    chunks = make_chunks()
    mock_response = make_mock_response("Galectins suppress antitumor immunity [galectin_paper.pdf, p.2].")

    with patch("generation.generator._get_client") as mock_client:
        mock_client.return_value.chat.completions.create.return_value = mock_response
        result = generate("What do galectins do?", chunks)

    assert "answer" in result
    assert "sources" in result


def test_generate_answer_is_string():
    """answer field must be a string."""
    chunks = make_chunks()
    mock_response = make_mock_response("Galectins regulate immune responses.")

    with patch("generation.generator._get_client") as mock_client:
        mock_client.return_value.chat.completions.create.return_value = mock_response
        result = generate("What do galectins do?", chunks)

    assert isinstance(result["answer"], str)


def test_generate_sources_match_input_chunks():
    """sources in response should be the same chunks passed in."""
    chunks = make_chunks(3)
    mock_response = make_mock_response("Some answer.")

    with patch("generation.generator._get_client") as mock_client:
        mock_client.return_value.chat.completions.create.return_value = mock_response
        result = generate("What do galectins do?", chunks)

    assert result["sources"] == chunks


def test_generate_empty_chunks_returns_cannot_find():
    """Empty chunks should return 'cannot find' response without API call."""
    with patch("generation.generator._get_client") as mock_client:
        result = generate("What do galectins do?", chunks=[])

    # API should never be called for empty chunks
    mock_client.assert_not_called()
    assert "cannot find" in result["answer"].lower()
    assert result["sources"] == []


def test_generate_calls_api_with_correct_model():
    """generate() should call the API with the configured model."""
    chunks = make_chunks()
    mock_response = make_mock_response("Some answer.")

    with patch("generation.generator._get_client") as mock_client, \
         patch.dict("os.environ", {"NIM_MODEL": "meta/llama-3.3-70b-instruct"}):
        mock_client.return_value.chat.completions.create.return_value = mock_response
        generate("What do galectins do?", chunks)

    call_kwargs = mock_client.return_value.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "meta/llama-3.3-70b-instruct"


def test_generate_passes_system_and_user_messages():
    """API call must include both system and user messages."""
    chunks = make_chunks()
    mock_response = make_mock_response("Some answer.")

    with patch("generation.generator._get_client") as mock_client:
        mock_client.return_value.chat.completions.create.return_value = mock_response
        generate("What do galectins do?", chunks)

    messages = mock_client.return_value.chat.completions.create.call_args.kwargs["messages"]
    roles = [m["role"] for m in messages]
    assert "system" in roles
    assert "user" in roles


def test_generate_user_message_contains_query():
    """The user message must contain the query string."""
    chunks = make_chunks()
    query = "What is the role of Gal-3 in immune evasion?"
    mock_response = make_mock_response("Some answer.")

    with patch("generation.generator._get_client") as mock_client:
        mock_client.return_value.chat.completions.create.return_value = mock_response
        generate(query, chunks)

    messages = mock_client.return_value.chat.completions.create.call_args.kwargs["messages"]
    user_message = next(m["content"] for m in messages if m["role"] == "user")
    assert query in user_message


def test_generate_user_message_contains_chunk_text():
    """The user message must contain chunk text so the LLM has context."""
    chunks = make_chunks(1)
    mock_response = make_mock_response("Some answer.")

    with patch("generation.generator._get_client") as mock_client:
        mock_client.return_value.chat.completions.create.return_value = mock_response
        generate("What do galectins do?", chunks)

    messages = mock_client.return_value.chat.completions.create.call_args.kwargs["messages"]
    user_message = next(m["content"] for m in messages if m["role"] == "user")
    assert chunks[0]["text"] in user_message


# --- context block tests ---

def test_build_context_block_contains_filename():
    """Context block must include filename for citation purposes."""
    chunks = make_chunks(1)
    block = _build_context_block(chunks)
    assert "galectin_paper.pdf" in block


def test_build_context_block_contains_page_number():
    """Context block must include page number for citation purposes."""
    chunks = make_chunks(1)
    block = _build_context_block(chunks)
    assert str(chunks[0]["page_number"]) in block


def test_build_context_block_contains_chunk_text():
    """Context block must include the actual chunk text."""
    chunks = make_chunks(1)
    block = _build_context_block(chunks)
    assert chunks[0]["text"] in block


def test_build_context_block_numbers_chunks():
    """Each chunk should be numbered starting from 1."""
    chunks = make_chunks(3)
    block = _build_context_block(chunks)
    assert "[1]" in block
    assert "[2]" in block
    assert "[3]" in block
