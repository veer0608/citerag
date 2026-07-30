"""Unit tests for citation-marker parsing (no LLM / no DB).

The contract: /query reports the passages the ANSWER cited, not everything that was
retrieved — so a claim with no marker is reported as uncited rather than dressed up
with the whole candidate pool.
"""
from __future__ import annotations

from app.llm import MAX_INFERRED, _citations, infer_citations, used_markers
from app.retrieval import RetrievedChunk


def _chunk(
    cid: str, page: int, label: str | None = None, content: str | None = None
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid,
        document_id="doc",
        page_number=page,
        page_label=label,
        content=content if content is not None else cid,
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


def test_citations_record_declared_vs_inferred():
    chunks = [_chunk("a", 1)]
    declared = _citations(chunks, [1])[0]
    assert declared["inferred"] is False
    assert declared["matched_figures"] == []

    guessed = _citations(chunks, [1], inferred=True, basis={1: ["41.4%"]})[0]
    assert guessed["inferred"] is True
    assert guessed["matched_figures"] == ["41.4%"]


# --- post-hoc grounding attribution -----------------------------------------


def test_infers_the_passage_holding_the_answer_figure():
    chunks = [
        _chunk("a", 1, content="Berkshire owns a 38.6% interest in Pilot."),
        _chunk("b", 2, content="Acquired an additional 41.4% of Pilot in 2023."),
        _chunk("c", 3, content="Unrelated prose about insurance float."),
    ]
    assert infer_citations("Berkshire acquired an additional 41.4%.", chunks) == [
        (2, ["41.4%"])
    ]


def test_ranks_by_how_much_of_the_answer_a_passage_accounts_for():
    chunks = [
        _chunk("a", 1, content="Fair value 174,347 reported."),
        _chunk("b", 2, content="Cost 31,089 and fair value 174,347 together."),
    ]
    got = infer_citations("Cost was 31,089 and fair value 174,347.", chunks)
    assert got[0][0] == 2  # matches both figures, so it ranks first
    assert sorted(got[0][1]) == ["174,347", "31,089"]


def test_ignores_figures_present_in_most_passages():
    # "2023" is in every passage, so it cannot discriminate between them.
    chunks = [_chunk(c, i, content=f"Report for 2023, item {c}") for i, c in enumerate("abcd", 1)]
    assert infer_citations("This was reported in 2023.", chunks) == []


def test_ignores_single_digits():
    chunks = [_chunk("a", 1, content="There were 8 investees listed.")]
    assert infer_citations("There were 8 investees.", chunks) == []


def test_ignores_short_integers_like_date_components():
    # "31" comes from "January 31" — not a claim, and with separators stripped it
    # would also substring-match inside unrelated numbers such as "31,089".
    chunks = [_chunk("a", 1, content="Costs of 31,089 were recorded.")]
    assert infer_citations("The deal closed on January 31, 2023.", chunks) == []


def test_keeps_decimals_and_percentages_even_when_short():
    chunks = [_chunk("a", 1, content="float grew by 3.3 billion")]
    assert infer_citations("Float grew 3.3 billion.", chunks) == [(1, ["3.3"])]


def test_matches_across_comma_and_space_noise():
    # pdfplumber output loses/adds separators; "174,347" must still match "174347".
    chunks = [_chunk("a", 1, content="fairvalue174347atyearend")]
    assert infer_citations("The fair value was 174,347.", chunks) == [(1, ["174,347"])]


def test_caps_inferred_citations_so_the_whole_pool_is_never_reattached():
    # A figure echoed by every passage must not turn all of them back into
    # "citations" — that's the behaviour this feature exists to remove.
    chunks = [_chunk(c, i, content="fair value 174,347") for i, c in enumerate("abcde", 1)]
    got = infer_citations("Fair value was 174,347.", chunks)
    assert len(got) == MAX_INFERRED < len(chunks)


def test_no_figures_or_no_chunks_infers_nothing():
    assert infer_citations("Berkshire is a holding company.", [_chunk("a", 1)]) == []
    assert infer_citations("Value was 41.4%.", []) == []
