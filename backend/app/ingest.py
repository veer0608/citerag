"""PDF -> parsed pages -> chunks -> embeddings -> DB.

Kept free of FastAPI so it can be driven from a script (seed_corpus.py), a test,
or the /ingest route identically. Chunk metadata goes to the `chunks` table;
embeddings go to the sqlite-vec `vec_chunks` table, joined by rowid.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path

import pdfplumber
from sqlalchemy import delete
from sqlalchemy.orm import Session

from app import vectorstore
from app.chunking import Page, fixed_size_chunks
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


def parse_pdf(path: Path) -> tuple[list[Page], int]:
    """Extract text per page. Returns (pages, empty_page_count).

    Empty pages are surfaced rather than silently dropped: a page with no
    extractable text is almost certainly a scanned image that would need OCR, and
    hiding that would quietly corrupt recall numbers later.
    """
    pages: list[Page] = []
    empty = 0
    with pdfplumber.open(str(path)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            if not text.strip():
                empty += 1
            pages.append(Page(page_number=i, text=text))
    return pages, empty


def ingest_pdf(session: Session, path: Path, *, title: str | None = None) -> IngestResult:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    pages, empty = parse_pdf(path)
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

    # Idempotent: re-ingesting the same file replaces its prior rows. Vec rows have
    # no FK cascade, so drop them explicitly before the document (which cascades to
    # its chunk rows).
    vectorstore.delete_document_vectors(session, doc_id)
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

    session.commit()

    return IngestResult(
        document_id=str(document.id),
        title=document.title,
        n_pages=len(pages),
        n_chunks=len(chunks),
        n_empty_pages=empty,
    )
