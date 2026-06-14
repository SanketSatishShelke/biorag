import pytest
from ingestion.pdf_parser import _normalize_whitespace


def test_non_breaking_space_replaced():
    """\\xa0 should be converted to regular space."""
    text = "et\xa0al. and Nishioka\xa0T."
    result = _normalize_whitespace(text)
    assert "\xa0" not in result
    assert "et al." in result


def test_multiple_spaces_collapsed():
    """Runs of spaces/tabs should collapse to single space."""
    text = "CD8+   T    cells"
    result = _normalize_whitespace(text)
    assert result == "CD8+ T cells"


def test_excessive_newlines_collapsed():
    """3+ consecutive newlines should collapse to exactly 2."""
    text = "Paragraph one.\n\n\n\n\nParagraph two."
    result = _normalize_whitespace(text)
    assert "\n\n\n" not in result
    assert "Paragraph one.\n\nParagraph two." in result


def test_paragraph_breaks_preserved():
    """Double newlines (paragraph breaks) should be preserved."""
    text = "First paragraph.\n\nSecond paragraph."
    result = _normalize_whitespace(text)
    assert "\n\n" in result

def test_empty_string_returns_empty():
    """Empty input should return empty string."""
    assert _normalize_whitespace("") == ""


def test_whitespace_only_returns_empty():
    """String of only whitespace should return empty after strip."""
    assert _normalize_whitespace("   \n\n\t  ") == ""


def test_clean_text_unchanged():
    """Text with no artifacts should pass through without change."""
    text = "Perforin and granzyme B mediate cytotoxic killing."
    result = _normalize_whitespace(text)
    assert result == text