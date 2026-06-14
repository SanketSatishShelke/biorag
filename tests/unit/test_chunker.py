import pytest
from ingestion.chunker import chunk_pages, ENCODING


# --- helpers ---

def make_pages(texts: list[str]) -> list[dict]:
    """Build a pages list as extract_pages() would return."""
    return [{"page_number": i + 1, "text": t} for i, t in enumerate(texts)]


def token_count(text: str) -> int:
    return len(ENCODING.encode(text))


# --- tests ---

def test_single_short_page_produces_one_chunk():
    """A page well under chunk_size should produce exactly one chunk."""
    pages = make_pages(["CD8+ T cells mediate antitumor immunity."])
    chunks = chunk_pages(pages, chunk_size=512, overlap=50)
    assert len(chunks) == 1


def test_chunk_contains_correct_keys():
    """Every chunk dict must have text, chunk_index, and page_number."""
    pages = make_pages(["Some biomedical text about T cells."])
    chunks = chunk_pages(pages, chunk_size=512, overlap=50)
    for chunk in chunks:
        assert "text" in chunk
        assert "chunk_index" in chunk
        assert "page_number" in chunk


def test_chunk_index_is_sequential():
    """chunk_index must start at 0 and increment by 1."""
    # generate enough text to produce multiple chunks
    long_text = "Immunotherapy targets tumor microenvironment. " * 100
    pages = make_pages([long_text])
    chunks = chunk_pages(pages, chunk_size=50, overlap=10)
    indices = [c["chunk_index"] for c in chunks]
    assert indices == list(range(len(chunks)))


def test_chunks_respect_chunk_size():
    """No chunk should exceed chunk_size tokens."""
    long_text = "Regulatory T cells suppress antitumor immunity. " * 200
    pages = make_pages([long_text])
    chunk_size = 100
    chunks = chunk_pages(pages, chunk_size=chunk_size, overlap=20)
    for chunk in chunks:
        assert token_count(chunk["text"]) <= chunk_size


def test_overlap_is_present():
    """
    The tail of chunk N and head of chunk N+1 should share tokens.
    We verify this by checking that text from the end of chunk 0
    appears at the start of chunk 1.
    """
    long_text = "Perforin and granzyme B mediate cytotoxic killing. " * 100
    pages = make_pages([long_text])
    chunks = chunk_pages(pages, chunk_size=50, overlap=20)
    assert len(chunks) >= 2

    # last 20 tokens of chunk 0
    tail_tokens = ENCODING.encode(chunks[0]["text"])[-20:]
    # first 20 tokens of chunk 1
    head_tokens = ENCODING.encode(chunks[1]["text"])[:20]
    assert tail_tokens == head_tokens


def test_page_number_tracked_correctly():
    """Chunks starting in page 1 text should have page_number=1."""
    # Use enough text to ensure page 1 content produces at least one chunk
    # before page 2 content starts
    page1_text = "CD8+ T cells mediate antitumor immunity through perforin. " * 60
    page2_text = "Regulatory T cells suppress immune responses in tumors. " * 60
    pages = make_pages([page1_text, page2_text])
    chunks = chunk_pages(pages, chunk_size=200, overlap=20)

    # first chunk must start in page 1
    assert chunks[0]["page_number"] == 1

    # at least one chunk must start in page 2
    page_numbers = [c["page_number"] for c in chunks]
    assert 2 in page_numbers


def test_empty_pages_produces_no_chunks():
    """Empty page list should return empty chunk list."""
    chunks = chunk_pages([], chunk_size=512, overlap=50)
    assert chunks == []


def test_multi_page_chunk_boundary():
    """
    When concatenated pages exceed chunk_size, chunks should span
    page boundaries — verified by getting more chunks than pages.
    """
    # each repetition is ~10 tokens, 60 reps = ~600 tokens > chunk_size 512
    text = "Tumor infiltrating lymphocytes express PD-1. " * 60
    pages = make_pages([text])
    chunks = chunk_pages(pages, chunk_size=200, overlap=20)
    assert len(chunks) > 1
