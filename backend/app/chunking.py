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
