"""sqlite-vec vector store.

Embeddings live in the `vec_chunks` virtual table (created by the migration),
keyed by the same integer rowid SQLite assigns to each row in `chunks`. That rowid
is the join key between the relational chunk metadata and its vector.

This module is the single seam that would be swapped to use pgvector later — the
rest of the app talks to `knn` / `upsert_chunk_vectors` and never to SQL directly.
"""
from __future__ import annotations

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
