"""Unit tests for table serialization (no PDF / no DB)."""
from __future__ import annotations

from app.ingest import serialize_tables


def test_serialize_drops_separator_cells_and_keeps_label_with_values():
    # A pdfplumber financial-statement row: label, empty separators, '$' and the
    # numbers in their own cells. Only the alphanumeric survivors should remain.
    table = [["Insurance premiums earned", "", "$", "83,403", "", "$", "74,576"]]
    assert serialize_tables([table]) == "Insurance premiums earned | 83,403 | 74,576"


def test_serialize_skips_rows_with_fewer_than_two_tokens():
    # A lone header/label with no value is not a fact worth its own line.
    table = [["Revenues:", "", "", None], ["Apple Inc.", "915,560,382", "174,347"]]
    assert serialize_tables([table]) == "Apple Inc. | 915,560,382 | 174,347"


def test_serialize_ignores_placeholder_and_punctuation_only_cells():
    # '�' (dash placeholder) and ')' carry no alphanumeric token -> dropped;
    # "(21,998" keeps its digits.
    table = [["Investment losses", "", "(21,998", ")", "�"]]
    assert serialize_tables([table]) == "Investment losses | (21,998"


def test_serialize_handles_none_and_empty():
    assert serialize_tables(None) == ""
    assert serialize_tables([]) == ""
    assert serialize_tables([[["", None, "$"]]]) == ""
