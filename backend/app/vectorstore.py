"""sqlite-vec vector store.

Embeddings live in the `vec_chunks` virtual table (created by the migration),
keyed by the same integer rowid SQLite assigns to each row in `chunks`. That rowid
is the join key between the relational chunk metadata and its vector.

This module is the single seam that would be swapped to use pgvector later — the
rest of the app talks to `knn` / `upsert_chunk_vectors` and never to SQL directly.
"""
from __future__ import annotations

import re
import struct
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.orm import Session


def serialize(vector: list[float]) -> bytes:
    """Pack a float list into the compact float32 blob sqlite-vec expects."""
    return struct.pack(f"{len(vector)}f", *vector)


@dataclass
class VecHit:
    chunk_id: str
    document_id: str
    page_number: int | None
    content: str
    distance: float  # cosine distance: 0 = identical, 2 = opposite


def upsert_chunk_vectors(session: Session, rows: list[tuple[int, list[float]]]) -> None:
    """Insert (chunk_rowid, embedding) pairs into the vec table."""
    if not rows:
        return
    session.execute(
        text("INSERT INTO vec_chunks(rowid, embedding) VALUES (:rowid, :emb)"),
        [{"rowid": rowid, "emb": serialize(vec)} for rowid, vec in rows],
    )


def delete_document_vectors(session: Session, document_id: str) -> None:
    """Remove vec rows for a document's chunks (virtual tables have no FK cascade)."""
    session.execute(
        text(
            "DELETE FROM vec_chunks WHERE rowid IN "
            "(SELECT rowid FROM chunks WHERE document_id = :doc)"
        ),
        {"doc": document_id},
    )


def chunk_rowids(session: Session, document_id: str) -> dict[str, int]:
    """Map chunk id -> SQLite rowid for a freshly inserted document."""
    result = session.execute(
        text("SELECT id, rowid FROM chunks WHERE document_id = :doc"),
        {"doc": document_id},
    )
    return {row.id: row.rowid for row in result}


def upsert_chunk_fts(session: Session, rows: list[tuple[int, str]]) -> None:
    """Insert (chunk_rowid, content) pairs into the FTS5 keyword index."""
    if not rows:
        return
    session.execute(
        text("INSERT INTO fts_chunks(rowid, content) VALUES (:rowid, :content)"),
        [{"rowid": rowid, "content": content} for rowid, content in rows],
    )


def delete_document_fts(session: Session, document_id: str) -> None:
    """Remove FTS rows for a document's chunks (virtual tables have no FK cascade)."""
    session.execute(
        text(
            "DELETE FROM fts_chunks WHERE rowid IN "
            "(SELECT rowid FROM chunks WHERE document_id = :doc)"
        ),
        {"doc": document_id},
    )


def _fts_match_query(query: str) -> str:
    """Turn free text into a safe FTS5 MATCH expression.

    FTS5 treats characters like " * : ( ) - as operators, so a raw question would
    be a syntax error. Extract word/number tokens, quote each as a literal, and OR
    them together — OR (not the default AND) keeps lexical recall high; precision is
    recovered by the re-ranker downstream. Returns "" when there's nothing to match.
    """
    tokens = re.findall(r"\w+", query.lower())
    return " OR ".join(f'"{t}"' for t in tokens)


def keyword_search(session: Session, query: str, k: int) -> list[VecHit]:
    """Return the k best chunks for a query by BM25 over the FTS5 index.

    `distance` carries the raw BM25 score (lower = better in SQLite's bm25()), only
    so the list stays ordered; hybrid fusion uses rank position, not the raw score.
    """
    match = _fts_match_query(query)
    if not match:
        return []
    sql = text(
        """
        SELECT c.id, c.document_id, c.page_number, c.content, f.score
        FROM (
            SELECT rowid, bm25(fts_chunks) AS score
            FROM fts_chunks
            WHERE fts_chunks MATCH :q
            ORDER BY score
            LIMIT :k
        ) AS f
        JOIN chunks c ON c.rowid = f.rowid
        ORDER BY f.score
        """
    )
    result = session.execute(sql, {"q": match, "k": k})
    return [
        VecHit(
            chunk_id=row.id,
            document_id=row.document_id,
            page_number=row.page_number,
            content=row.content,
            distance=float(row.score),
        )
        for row in result
    ]


def knn(session: Session, query_embedding: list[float], k: int) -> list[VecHit]:
    """Return the k nearest chunks to the query embedding, closest first."""
    sql = text(
        """
        SELECT c.id, c.document_id, c.page_number, c.content, v.distance
        FROM (
            SELECT rowid, distance
            FROM vec_chunks
            WHERE embedding MATCH :q AND k = :k
            ORDER BY distance
        ) AS v
        JOIN chunks c ON c.rowid = v.rowid
        ORDER BY v.distance
        """
    )
    result = session.execute(sql, {"q": serialize(query_embedding), "k": k})
    return [
        VecHit(
            chunk_id=row.id,
            document_id=row.document_id,
            page_number=row.page_number,
            content=row.content,
            distance=float(row.distance),
        )
        for row in result
    ]
