from unittest.mock import patch, MagicMock
import retrieval.query_rewriter as qr_module


def _mock_nim_response(text):
    mock_response = MagicMock()
    mock_response.choices[0].message.content = text
    return mock_response


def _make_mock_client(response_text):
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_nim_response(response_text)
    return mock_client


def test_rewrite_query_returns_string():
    mock_client = _make_mock_client("acute myeloid leukemia MYC oncogene mechanism")
    with patch.object(qr_module, "_client", None), \
         patch("retrieval.query_rewriter.OpenAI", return_value=mock_client):
        from retrieval.query_rewriter import rewrite_query
        result = rewrite_query("What does MYC do in AML?")
    assert isinstance(result, str)
    assert len(result) > 0


def test_rewrite_query_calls_llm_once():
    mock_client = _make_mock_client("acute myeloid leukemia MYC oncogene")
    with patch.object(qr_module, "_client", None), \
         patch("retrieval.query_rewriter.OpenAI", return_value=mock_client):
        from retrieval.query_rewriter import rewrite_query
        rewrite_query("What does MYC do in AML?")
    mock_client.chat.completions.create.assert_called_once()


def test_rewrite_query_falls_back_to_original_on_empty_response():
    """If LLM returns empty string, original question must be returned."""
    mock_client = _make_mock_client("   ")
    with patch.object(qr_module, "_client", None), \
         patch("retrieval.query_rewriter.OpenAI", return_value=mock_client):
        from retrieval.query_rewriter import rewrite_query
        result = rewrite_query("What does MYC do in AML?")
    assert result == "What does MYC do in AML?"


def test_rewrite_query_uses_temperature_zero():
    """Query rewriting must be deterministic — temperature must be 0.0."""
    mock_client = _make_mock_client("some rewritten query")
    with patch.object(qr_module, "_client", None), \
         patch("retrieval.query_rewriter.OpenAI", return_value=mock_client):
        from retrieval.query_rewriter import rewrite_query
        rewrite_query("What does MYC do in AML?")
    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["temperature"] == 0.0
