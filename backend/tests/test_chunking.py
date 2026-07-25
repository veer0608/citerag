"""Pure unit tests for the chunking strategy — no DB, runs anywhere."""
from __future__ import annotations

from app.chunking import (
    Page,
    count_tokens,
    fixed_size_chunks,
    structure_aware_chunks,
)


def _long_page(n_words: int, page_number: int = 1) -> Page:
    return Page(page_number=page_number, text=" ".join(f"word{i}" for i in range(n_words)))


def test_empty_input_yields_no_chunks():
    assert fixed_size_chunks([]) == []
    assert fixed_size_chunks([Page(page_number=1, text="   ")]) == []


def test_chunks_respect_token_budget():
    page = _long_page(2000)
    chunks = fixed_size_chunks([page], chunk_tokens=100, overlap_tokens=10)
    assert len(chunks) > 1
    for c in chunks:
        assert c.token_count <= 100
        assert count_tokens(c.content) <= 100


def test_overlap_produces_shared_context():
    page = _long_page(2000)
    no_overlap = fixed_size_chunks([page], chunk_tokens=100, overlap_tokens=0)
    with_overlap = fixed_size_chunks([page], chunk_tokens=100, overlap_tokens=50)
    # Larger overlap -> smaller step -> more chunks over the same text.
    assert len(with_overlap) > len(no_overlap)


def test_chunk_indices_are_sequential():
    page = _long_page(1000)
    chunks = fixed_size_chunks([page], chunk_tokens=100, overlap_tokens=10)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_page_number_tracked_across_pages():
    pages = [_long_page(300, page_number=1), _long_page(300, page_number=2)]
    chunks = fixed_size_chunks(pages, chunk_tokens=100, overlap_tokens=0)
    seen_pages = {c.page_number for c in chunks}
    assert seen_pages == {1, 2}


def test_invalid_params_raise():
    page = _long_page(100)
    for bad in [dict(chunk_tokens=0), dict(chunk_tokens=50, overlap_tokens=50)]:
        try:
            fixed_size_chunks([page], **bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad}")


# --- structure-aware strategy ---

def _lined_page(n_lines: int, words_per_line: int, page_number: int = 1) -> Page:
    text = "\n".join(
        " ".join(f"w{p}" for p in range(words_per_line)) for _ in range(n_lines)
    )
    return Page(page_number=page_number, text=text)


def test_structure_chunks_never_cross_pages():
    pages = [_lined_page(20, 10, 1), _lined_page(20, 10, 2)]
    chunks = structure_aware_chunks(pages, max_tokens=60, overlap_lines=1)
    assert len(chunks) > 2
    # Every chunk belongs to exactly one page (no cross-page blending).
    assert {c.page_number for c in chunks} == {1, 2}


def test_structure_chunks_preserve_whole_lines():
    # Distinct lines so we can check none are split mid-line.
    text = "\n".join(f"line-{i} alpha beta gamma delta" for i in range(30))
    chunks = structure_aware_chunks([Page(1, text)], max_tokens=40, overlap_lines=0)
    for c in chunks:
        for ln in c.content.splitlines():
            assert ln.startswith("line-")  # each line intact, not cut mid-way


def test_structure_long_single_line_becomes_its_own_chunk():
    long_line = " ".join(f"tok{i}" for i in range(500))
    chunks = structure_aware_chunks([Page(1, long_line)], max_tokens=50)
    assert len(chunks) == 1
    assert chunks[0].content == long_line


def test_structure_invalid_params_raise():
    page = _lined_page(5, 5)
    for bad in [dict(max_tokens=0), dict(overlap_lines=-1)]:
        try:
            structure_aware_chunks([page], **bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad}")
