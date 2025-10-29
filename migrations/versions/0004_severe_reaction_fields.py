"""Add severe reaction fields to processed claims

Revision ID: 0004_severe_reaction_fields
Revises: 0003_article_snippet_tags
Create Date: 2024-05-27 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa


revision = "0004_severe_reaction_fields"
down_revision = "0003_article_snippet_tags"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing_columns = {column["name"] for column in inspector.get_columns("processed_claims")}

    if "severe_reaction_flag" not in existing_columns:
        op.add_column(
            "processed_claims",
            sa.Column("severe_reaction_flag", sa.Boolean(), nullable=False, server_default=sa.false()),
        )

    if "severe_reaction_terms" not in existing_columns:
        op.add_column(
            "processed_claims",
            sa.Column("severe_reaction_terms", sa.JSON(), nullable=False, server_default=sa.text("'[]'")),
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing_columns = {column["name"] for column in inspector.get_columns("processed_claims")}

    if "severe_reaction_terms" in existing_columns:
        op.drop_column("processed_claims", "severe_reaction_terms")

    if "severe_reaction_flag" in existing_columns:
        op.drop_column("processed_claims", "severe_reaction_flag")
