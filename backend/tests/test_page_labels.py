"""Unit tests for printed page-label extraction (no PDF / no DB)."""
from __future__ import annotations

from app.ingest import extract_page_label


def test_extracts_plain_number_and_strips_it():
    text = "Surprise, Surprise\nBerkshire paid $3.3 billion.\n4"
    body, label = extract_page_label(text)
    assert label == "4"
    # The label must not stay in the indexed content, or it becomes searchable text.
    assert body == "Surprise, Surprise\nBerkshire paid $3.3 billion."


def test_extracts_prefixed_10k_label():
    text = "Notes to Consolidated Financial Statements\nFair value 16,434\nK-83"
    body, label = extract_page_label(text)
    assert label == "K-83"
    assert body.endswith("Fair value 16,434")


def test_ignores_trailing_line_that_is_not_a_label():
    text = "Apple Inc. fair value was $161,155 million"
    body, label = extract_page_label(text)
    assert label is None
    assert body == text


def test_ignores_long_number_that_is_not_a_page_label():
    # A 4+ digit figure is data, not a page number.
    text = "Total revenues\n302089"
    body, label = extract_page_label(text)
    assert label is None
    assert body == text


def test_skips_trailing_blank_lines():
    text = "Some prose here\nK-45\n\n  \n"
    body, label = extract_page_label(text)
    assert label == "K-45"
    assert "K-45" not in body


def test_handles_empty_text():
    assert extract_page_label("") == ("", None)
