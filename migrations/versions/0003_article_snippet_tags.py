"""add tags column to article snippets

Revision ID: 0003_article_snippet_tags
Revises: 0002_add_slugs
Create Date: 2025-10-27 00:00:00.000001

"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.engine.reflection import Inspector
from sqlalchemy.orm import Session

from app.models import ArticleSnippet

# revision identifiers, used by Alembic.
revision = "0003_article_snippet_tags"
down_revision = "0002_add_slugs"
branch_labels = None
depends_on = None


def _table_has_column(inspector: Inspector, table: str, column: str) -> bool:
    return any(col["name"] == column for col in inspector.get_columns(table))


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if _table_has_column(inspector, "article_snippets", "tags"):
        return

    is_sqlite = bind.dialect.name == "sqlite"
    server_default = sa.text("'[]'")

    if is_sqlite:
        with op.batch_alter_table("article_snippets") as batch:
            batch.add_column(sa.Column("tags", sa.JSON(), nullable=False, server_default=server_default))
    else:
        op.add_column(
            "article_snippets",
            sa.Column("tags", sa.JSON(), nullable=False, server_default=server_default),
        )

    session = Session(bind=bind)
    try:
        session.query(ArticleSnippet).update({ArticleSnippet.tags: []})
        session.commit()
    finally:
        session.close()

    if is_sqlite:
        with op.batch_alter_table("article_snippets") as batch:
            batch.alter_column(
                "tags",
                existing_type=sa.JSON(),
                server_default=None,
                nullable=False,
            )
    else:
        op.alter_column(
            "article_snippets",
            "tags",
            server_default=None,
            existing_type=sa.JSON(),
            nullable=False,
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if not _table_has_column(inspector, "article_snippets", "tags"):
        return

    is_sqlite = bind.dialect.name == "sqlite"

    if is_sqlite:
        with op.batch_alter_table("article_snippets") as batch:
            batch.drop_column("tags")
    else:
        op.drop_column("article_snippets", "tags")
