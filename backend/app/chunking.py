"""Chunking strategy — deliberately isolated so it can be swapped and unit-tested.

Phase 1 is naive fixed-size token chunking with overlap. Phase 3 will add a
structure-aware strategy (keep table rows / sections intact) and the eval harness
will prove whether it actually helped.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import tiktoken


@lru_cache
def _encoder():
    # cl100k_base is a reasonable general-purpose tokenizer and matches OpenAI
    # models, so token_count stays meaningful whichever embedding backend is used.
    # Loaded lazily so importing this module doesn't require a network fetch.
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_encoder().encode(text))


@dataclass
class Chunk:
    """A single chunk before it has an id / embedding."""

    chunk_index: int
    page_number: int | None
    content: str
    token_count: int


@dataclass
class Page:
    """One parsed PDF page."""

    page_number: int
    text: str


def fixed_size_chunks(
    pages: list[Page],
    *,
    chunk_tokens: int = 500,
    overlap_tokens: int = 50,
) -> list[Chunk]:
    """Concatenate page text and slice into overlapping token windows.

    Each chunk is tagged with the page number where it STARTS, so citations can
    point at a page even though a chunk may span a page boundary.
    """
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be positive")
    if overlap_tokens >= chunk_tokens:
        raise ValueError("overlap_tokens must be smaller than chunk_tokens")

    # Build a flat token stream, remembering which page each token came from.
    tokens: list[int] = []
    token_pages: list[int] = []
    for page in pages:
        page_tokens = _encoder().encode(page.text)
        tokens.extend(page_tokens)
        token_pages.extend([page.page_number] * len(page_tokens))

    if not tokens:
        return []

    step = chunk_tokens - overlap_tokens
    chunks: list[Chunk] = []
    index = 0
    for start in range(0, len(tokens), step):
        window = tokens[start : start + chunk_tokens]
        if not window:
            break
        content = _encoder().decode(window).strip()
        if not content:
            continue
        chunks.append(
            Chunk(
                chunk_index=index,
                page_number=token_pages[start],
                content=content,
                token_count=len(window),
            )
        )
        index += 1
        if start + chunk_tokens >= len(tokens):
            break

    return chunks


def structure_aware_chunks(
    pages: list[Page],
    *,
    max_tokens: int = 220,
    overlap_lines: int = 1,
) -> list[Chunk]:
    """Page-bounded, line-preserving chunking.

    Two structural rules the naive strategy violates:
    1. Chunks never cross a page boundary, so page citations are exact and a chunk
       stays on one topic instead of blending the end of one page with the start of
       the next.
    2. Chunks are packed from whole lines and never split a line mid-way, so a table
       row (or a sentence) stays intact — the diagnosis showed specific facts were
       getting diluted inside large multi-topic windows.

    Smaller windows also concentrate each fact, making its embedding more specific.
    A line longer than `max_tokens` on its own is emitted as its own chunk rather
    than dropped.
    """
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if overlap_lines < 0:
        raise ValueError("overlap_lines must be non-negative")

    enc = _encoder()
    chunks: list[Chunk] = []
    index = 0

    for page in pages:
        lines = [ln.strip() for ln in page.text.splitlines() if ln.strip()]
        if not lines:
            continue

        cur_lines: list[str] = []
        cur_tokens = 0

        def flush() -> None:
            nonlocal index, cur_lines, cur_tokens
            if not cur_lines:
                return
            content = "\n".join(cur_lines).strip()
            if content:
                chunks.append(
                    Chunk(
                        chunk_index=index,
                        page_number=page.page_number,
                        content=content,
                        token_count=len(enc.encode(content)),
                    )
                )
                index += 1

        for line in lines:
            n = len(enc.encode(line))
            # Adding this line would overflow the budget -> close the current chunk
            # and start the next one carrying the last `overlap_lines` for context.
            if cur_lines and cur_tokens + n > max_tokens:
                flush()
                carry = cur_lines[-overlap_lines:] if overlap_lines else []
                cur_lines = list(carry)
                cur_tokens = sum(len(enc.encode(x)) for x in cur_lines)
            cur_lines.append(line)
            cur_tokens += n

        flush()

    return chunks
