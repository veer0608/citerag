"""PDF -> parsed pages -> chunks -> embeddings -> DB.

Kept free of FastAPI so it can be driven from a script (seed_corpus.py), a test,
or the /ingest route identically. Chunk metadata goes to the `chunks` table;
embeddings go to the sqlite-vec `vec_chunks` table, joined by rowid.
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from pathlib import Path

import pdfplumber
from sqlalchemy import delete
from sqlalchemy.orm import Session

from app import vectorstore
from app.chunking import Page, fixed_size_chunks, structure_aware_chunks
from app.config import settings
from app.embeddings import embed_documents
from app.models import Chunk, Document

# Stable namespace so re-ingesting the same file with the same chunking config
# yields the same document/chunk ids — a golden set keyed on chunk id survives.
_NS = uuid.UUID("6f6d0b7e-1f2a-4d3c-9b1e-0c1a2b3d4e5f")


def _document_id(source_path: str) -> str:
    return str(uuid.uuid5(_NS, f"doc:{source_path}"))


def _chunk_id(source_path: str, chunk_index: int, content: str) -> str:
    return str(uuid.uuid5(_NS, f"chunk:{source_path}:{chunk_index}:{content}"))


@dataclass
class IngestResult:
    document_id: str
    title: str
    n_pages: int
    n_chunks: int
    n_empty_pages: int  # pages that yielded no text (likely scanned -> would need OCR)


# pdfplumber frequently emits words welded together — "MitsubishiCorporation",
# "AppleInc", "Berkshirepaid$27.1billionin2021". That is fatal for the lexical half
# of hybrid retrieval: FTS5 tokenizes on non-alphanumerics, so the whole run becomes
# ONE token and a query for "mitsubishi" cannot match it (verified against the index).
# Splitting at case boundaries restores the entity as a searchable token.
#
# Two boundaries are safe to split on:
#   lower -> Upper   "MitsubishiCorporation" -> "Mitsubishi Corporation"
#   ACRONYM -> Word  "BYDCo"                 -> "BYD Co"
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z])(?=[A-Z])")
_ACRONYM_BOUNDARY = re.compile(r"(?<=[A-Z])(?=[A-Z][a-z])")
# Only tokens at least this long are touched. Ordinary prose words are shorter, so
# this leaves normal text — and correctly-spaced sentences — completely alone.
_MIN_WELDED_LEN = 8


def split_run_together(text: str) -> str:
    """Re-space words that PDF extraction welded together.

    Conservative by design: a token is only split when it is long enough to be
    suspicious AND contains an internal case boundary. Real prose is untouched,
    because ordinary words are short and have no interior capitals.

    Imperfect splits are acceptable — the goal is that the entity becomes a token a
    query can match, not perfect typography. "BankofAmericaCorp" yields
    "Bankof America Corp": "america" is now searchable even though "bank" is not.
    """
    out = []
    for token in text.split(" "):
        if len(token) >= _MIN_WELDED_LEN:
            token = _ACRONYM_BOUNDARY.sub(" ", _CAMEL_BOUNDARY.sub(" ", token))
        out.append(token)
    return " ".join(out)


# A printed page label sitting alone on the final line of a page: either a plain
# number from the shareholder letter ("7") or the 10-K's prefixed form ("K-83").
# Anchored and length-capped so a stray figure in the text isn't mistaken for one.
_PAGE_LABEL_RE = re.compile(r"^([A-Z]{1,2}-)?\d{1,3}$")


def extract_page_label(text: str) -> tuple[str, str | None]:
    """Split a trailing printed page label off a page's text.

    Returns (text_without_label, label_or_None). These reports print the page
    number as the last line of the page, and it uses two different schemes — plain
    integers in the shareholder letter, "K-" prefixed in the 10-K — so it can't be
    derived from the physical index by any fixed offset. It has to be read off the
    page.

    The label is REMOVED from the text it labels: left in, it would be indexed as
    content and could match a query's own numbers.
    """
    lines = text.splitlines()
    # Walk back over trailing blank lines to find the last non-empty one.
    for i in range(len(lines) - 1, -1, -1):
        stripped = lines[i].strip()
        if not stripped:
            continue
        if _PAGE_LABEL_RE.match(stripped):
            return "\n".join(lines[:i] + lines[i + 1 :]), stripped
        break
    return text, None


def serialize_tables(tables: list[list[list[str | None]]] | None) -> str:
    """Flatten pdfplumber tables into one compact fact per row.

    pdfplumber emits financial tables with empty separator columns and with '$',
    ')' and the '�' placeholder split into their own cells. Keep only cells
    that carry an alphanumeric token, join a row's survivors with ' | ', and drop
    rows with fewer than two — so a label stays attached to its value(s) on one
    line ("Apple Inc. | 915,560,382 | 174,347") instead of being scattered.
    """
    lines: list[str] = []
    for table in tables or []:
        for row in table:
            cells = [
                cell.strip()
                for cell in row
                if cell and any(ch.isalnum() for ch in cell)
            ]
            if len(cells) >= 2:
                lines.append(" | ".join(cells))
    return "\n".join(lines)


def parse_pdf(path: Path, *, extract_tables: bool = False) -> tuple[list[Page], int]:
    """Extract text per page. Returns (pages, empty_page_count).

    Empty pages are surfaced rather than silently dropped: a page with no
    extractable text is almost certainly a scanned image that would need OCR, and
    hiding that would quietly corrupt recall numbers later.

    When `extract_tables` is set, each page's detected tables are serialized into
    row-level fact lines and appended to that page's text, so table rows survive
    chunking intact. A page counts as empty only when neither text nor tables yield
    anything. It defaults to False to match `settings.table_extraction_enabled` —
    that experiment measured *worse* (see the README), so a direct caller must opt
    in deliberately rather than inherit the rejected behaviour.

    Each page's printed label (the number shown on the page itself) is read off the
    end of its text and carried separately, so citations can quote what a reader
    sees rather than only the physical PDF index.
    """
    pages: list[Page] = []
    empty = 0
    with pdfplumber.open(str(path)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            # Read the label off the raw page text, before tables are appended.
            text, label = extract_page_label(text)
            if settings.split_run_together_enabled:
                text = split_run_together(text)
            if extract_tables:
                table_text = serialize_tables(page.extract_tables())
                if table_text:
                    text = (text + "\n" + table_text).strip() if text.strip() else table_text
            if not text.strip():
                empty += 1
            pages.append(Page(page_number=i, text=text, page_label=label))
    return pages, empty


def ingest_pdf(session: Session, path: Path, *, title: str | None = None) -> IngestResult:
    # Resolve to a canonical absolute path so the document/chunk ids are stable
    # regardless of whether the caller passed a relative or absolute path (and from
    # which working directory). Without this, re-ingesting the same file as e.g.
    # "data/corpus/x.pdf" then "/abs/data/corpus/x.pdf" hashes to different ids and
    # duplicates the document instead of replacing it.
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    pages, empty = parse_pdf(path, extract_tables=settings.table_extraction_enabled)
    if settings.chunk_strategy == "structure":
        chunks = structure_aware_chunks(
            pages,
            max_tokens=settings.structure_max_tokens,
            overlap_lines=settings.structure_overlap_lines,
        )
    else:
        chunks = fixed_size_chunks(
            pages,
            chunk_tokens=settings.chunk_tokens,
            overlap_tokens=settings.chunk_overlap_tokens,
        )
    if not chunks:
        raise ValueError(
            f"{path.name}: no extractable text in any page "
            f"({empty} empty pages) — likely a scanned PDF needing OCR."
        )

    embeddings = embed_documents([c.content for c in chunks])

    source_path = str(path)
    doc_id = _document_id(source_path)

    # Idempotent: re-ingesting the same file replaces its prior rows. Vec and FTS
    # rows have no FK cascade, so drop them explicitly before the document (which
    # cascades to its chunk rows).
    vectorstore.delete_document_vectors(session, doc_id)
    vectorstore.delete_document_fts(session, doc_id)
    session.execute(delete(Document).where(Document.id == doc_id))
    session.flush()

    document = Document(id=doc_id, title=title or path.stem, source_path=source_path)
    session.add(document)
    session.flush()

    id_by_index: dict[int, str] = {}
    for chunk in chunks:
        cid = _chunk_id(source_path, chunk.chunk_index, chunk.content)
        id_by_index[chunk.chunk_index] = cid
        session.add(
            Chunk(
                id=cid,
                document_id=document.id,
                chunk_index=chunk.chunk_index,
                page_number=chunk.page_number,
                page_label=chunk.page_label,
                content=chunk.content,
                token_count=chunk.token_count,
            )
        )
    session.flush()  # assigns rowids to the new chunk rows

    # Map chunk id -> rowid, then store each embedding against its chunk's rowid.
    rowid_by_id = vectorstore.chunk_rowids(session, document.id)
    vec_rows = [
        (rowid_by_id[id_by_index[chunk.chunk_index]], vector)
        for chunk, vector in zip(chunks, embeddings, strict=True)
    ]
    vectorstore.upsert_chunk_vectors(session, vec_rows)

    # Same rowids feed the FTS5 keyword index used by hybrid retrieval.
    fts_rows = [
        (rowid_by_id[id_by_index[chunk.chunk_index]], chunk.content)
        for chunk in chunks
    ]
    vectorstore.upsert_chunk_fts(session, fts_rows)

    session.commit()

    return IngestResult(
        document_id=str(document.id),
        title=document.title,
        n_pages=len(pages),
        n_chunks=len(chunks),
        n_empty_pages=empty,
    )
