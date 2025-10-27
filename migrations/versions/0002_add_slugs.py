"""add slug fields for search terms and processed claim sets

Revision ID: 0002_add_slugs
Revises: 0001_initial_schema
Create Date: 2025-10-27 00:00:00.000000

"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.engine.reflection import Inspector
from sqlalchemy.orm import Session

from app.models import ProcessedClaimSet, SearchTerm
from app.utils.slugs import build_claim_set_slug, build_search_term_slug

# revision identifiers, used by Alembic.
revision = "0002_add_slugs"
down_revision = "0001_initial_schema"
branch_labels = None
depends_on = None


def _table_has_column(inspector: Inspector, table: str, column: str) -> bool:
    return any(col["name"] == column for col in inspector.get_columns(table))


def _table_has_index(inspector: Inspector, table: str, index_name: str) -> bool:
    return any(idx["name"] == index_name for idx in inspector.get_indexes(table))


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    is_sqlite = bind.dialect.name == "sqlite"

    if not _table_has_column(inspector, "search_terms", "slug"):
        if is_sqlite:
            with op.batch_alter_table("search_terms") as batch:
                batch.add_column(sa.Column("slug", sa.String(length=256), nullable=True))
        else:
            op.add_column("search_terms", sa.Column("slug", sa.String(length=256), nullable=True))
    if not _table_has_index(inspector, "search_terms", "ix_search_terms_slug"):
        op.create_index("ix_search_terms_slug", "search_terms", ["slug"], unique=True)

    if not _table_has_column(inspector, "processed_claim_sets", "slug"):
        if is_sqlite:
            with op.batch_alter_table("processed_claim_sets") as batch:
                batch.add_column(sa.Column("slug", sa.String(length=256), nullable=True))
        else:
            op.add_column("processed_claim_sets", sa.Column("slug", sa.String(length=256), nullable=True))
    if not _table_has_index(inspector, "processed_claim_sets", "ix_processed_claim_sets_slug"):
        op.create_index("ix_processed_claim_sets_slug", "processed_claim_sets", ["slug"], unique=True)

    session = Session(bind=bind)
    try:
        for term in session.query(SearchTerm).filter(SearchTerm.slug.is_(None)):
            term.slug = build_search_term_slug(term.canonical)

        for claim_set in session.query(ProcessedClaimSet).filter(ProcessedClaimSet.slug.is_(None)):
            claim_set.slug = build_claim_set_slug(claim_set.condition_label, claim_set.mesh_signature)

        session.commit()
    finally:
        session.close()

    if is_sqlite:
        with op.batch_alter_table("search_terms") as batch:
            batch.alter_column("slug", existing_type=sa.String(length=256), nullable=False)
        with op.batch_alter_table("processed_claim_sets") as batch:
            batch.alter_column("slug", existing_type=sa.String(length=256), nullable=False)
    else:
        op.alter_column("search_terms", "slug", nullable=False)
        op.alter_column("processed_claim_sets", "slug", nullable=False)


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if _table_has_index(inspector, "processed_claim_sets", "ix_processed_claim_sets_slug"):
        op.drop_index("ix_processed_claim_sets_slug", table_name="processed_claim_sets")
    if _table_has_column(inspector, "processed_claim_sets", "slug"):
        op.drop_column("processed_claim_sets", "slug")

    if _table_has_index(inspector, "search_terms", "ix_search_terms_slug"):
        op.drop_index("ix_search_terms_slug", table_name="search_terms")
    if _table_has_column(inspector, "search_terms", "slug"):
        op.drop_column("search_terms", "slug")
