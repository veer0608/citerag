"""add chunks.page_label: the page number printed on the page

chunks.page_number is the 1-based PHYSICAL index within the PDF. The number printed
on the page is a different thing entirely in these reports — the shareholder letter
uses plain integers while the 10-K uses a "K-" prefix — so it can't be derived from
the physical index and has to be stored.

Nullable: ~6-10% of pages (covers, dividers, back matter) print no number at all.

Revision ID: 0003
Revises: 0002
Create Date: 2026-07-30
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("chunks", sa.Column("page_label", sa.String(16), nullable=True))


def downgrade() -> None:
    op.drop_column("chunks", "page_label")
