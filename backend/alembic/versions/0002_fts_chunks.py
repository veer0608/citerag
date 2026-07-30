"""add fts_chunks: FTS5 keyword index for hybrid (BM25 + dense) retrieval

Mirrors chunk content into an FTS5 virtual table keyed by the same rowid used to
join chunks <-> vec_chunks, so lexical and dense hits refer to the same chunk. Any
chunks already present are backfilled here; new chunks are indexed by app/ingest.py.

Revision ID: 0002
Revises: 0001
Create Date: 2026-07-30
"""
from __future__ import annotations

from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # FTS5 is compiled into SQLite (no extension load needed, unlike sqlite-vec).
    op.execute("CREATE VIRTUAL TABLE fts_chunks USING fts5(content)")
    # Backfill existing chunks so an already-seeded DB is searchable immediately.
    op.execute(
        "INSERT INTO fts_chunks(rowid, content) SELECT rowid, content FROM chunks"
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS fts_chunks")
