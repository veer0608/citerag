"""initial schema: documents, chunks, eval_questions, eval_runs, vec_chunks

Revision ID: 0001
Revises:
Create Date: 2026-07-24
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

from app.config import settings

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

EMBEDDING_DIM = settings.embedding_dim


def upgrade() -> None:
    op.create_table(
        "documents",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("source_path", sa.Text(), nullable=False),
        sa.Column(
            "ingested_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
    )

    op.create_table(
        "chunks",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "document_id",
            sa.String(36),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("page_number", sa.Integer(), nullable=True),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=False),
    )
    op.create_index("ix_chunks_document_id", "chunks", ["document_id"])

    # sqlite-vec virtual table. Its rowid is joined back to chunks.rowid. The
    # sqlite-vec extension is loaded on every connection (see app/db.py), so this
    # runs under Alembic too. cosine matches how embeddings are normalised.
    op.execute(
        f"CREATE VIRTUAL TABLE vec_chunks USING vec0("
        f"embedding float[{EMBEDDING_DIM}] distance_metric=cosine)"
    )

    op.create_table(
        "eval_questions",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("expected_answer", sa.Text(), nullable=True),
        sa.Column("expected_chunk_ids", sa.JSON(), nullable=False),
    )

    op.create_table(
        "eval_runs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "run_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column("config", sa.JSON(), nullable=False),
        sa.Column("metrics", sa.JSON(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("eval_runs")
    op.drop_table("eval_questions")
    op.execute("DROP TABLE IF EXISTS vec_chunks")
    op.drop_index("ix_chunks_document_id", table_name="chunks")
    op.drop_table("chunks")
    op.drop_table("documents")
