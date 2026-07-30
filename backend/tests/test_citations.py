"""Unit tests for citation-marker parsing (no LLM / no DB).

The contract: /query reports the passages the ANSWER cited, not everything that was
retrieved — so a claim with no marker is reported as uncited rather than dressed up
with the whole candidate pool.
"""
from __future__ import annotations

from app.llm import _citations, used_markers
from app.retrieval import RetrievedChunk


def _chunk(cid: str, page: int, label: str | None = None) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid,
        document_id="doc",
        page_number=page,
        page_label=label,
        content=cid,
        score=0.5,
    )


def test_extracts_markers_in_order_deduplicated():
    text = "Float grew [2]. Taxes were $3.3bn [1]. Also float [2] again."
    assert used_markers(text, 5) == [2, 1]


def test_drops_markers_outside_the_passage_range():
    # A model that invents [9] against 5 passages has cited nothing real.
    assert used_markers("Backed by [9] and [3].", 5) == [3]
    assert used_markers("All from [7].", 5) == []


def test_no_markers_means_nothing_cited():
    assert used_markers("Berkshire paid $3.3 billion in federal income taxes.", 5) == []


def test_long_bracketed_number_is_data_not_a_marker():
    # "[164]" style figures shouldn't be mistaken for a marker beyond 2 digits.
    assert used_markers("The value was [1234] million.", 5) == []


def test_citations_cover_only_the_used_markers():
    chunks = [_chunk("a", 8, "6"), _chunk("b", 99, "K-84"), _chunk("c", 12)]
    cites = _citations(chunks, [2])

    assert len(cites) == 1
    assert cites[0]["marker"] == 2
    assert cites[0]["chunk_id"] == "b"
    # The human-facing form prefers the printed label, keeping the PDF index too.
    assert cites[0]["page_citation"] == "page K-84 (PDF page 99)"


def test_citation_falls_back_to_pdf_page_without_a_label():
    cites = _citations([_chunk("c", 12)], [1])
    assert cites[0]["page_label"] is None
    assert cites[0]["page_citation"] == "PDF page 12"


def test_no_markers_yields_no_citations():
    assert _citations([_chunk("a", 1)], []) == []
