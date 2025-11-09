"""Introduce claim versioning and feedback tables

Revision ID: 0005_claim_versioning
Revises: 0004_severe_reaction_fields
Create Date: 2024-06-06 00:00:00.000000
"""

from __future__ import annotations

from datetime import datetime, timezone

from alembic import op
import sqlalchemy as sa


revision = "0005_claim_versioning"
down_revision = "0004_severe_reaction_fields"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    dialect_name = bind.dialect.name
    inspector = sa.inspect(bind)

    version_table_name = "processed_claim_set_versions"
    if not inspector.has_table(version_table_name):
        op.create_table(
            version_table_name,
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column(
                "claim_set_id",
                sa.Integer(),
                sa.ForeignKey("processed_claim_sets.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column("version_number", sa.Integer(), nullable=False),
            sa.Column("status", sa.String(length=32), nullable=False, server_default=sa.text("'draft'")),
            sa.Column("pipeline_metadata", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
            sa.Column("activated_at", sa.DateTime(timezone=True), nullable=True),
            sa.UniqueConstraint("claim_set_id", "version_number", name="uq_claimset_version_number"),
        )

    inspector = sa.inspect(bind)
    version_indexes = set()
    if inspector.has_table(version_table_name):
        version_indexes = {idx["name"] for idx in inspector.get_indexes(version_table_name)}
    if "ix_processed_claim_set_versions_claim_set_id" not in version_indexes:
        op.create_index(
            "ix_processed_claim_set_versions_claim_set_id",
            version_table_name,
            ["claim_set_id"],
        )

    processed_claims_columns = {
        column["name"]: column for column in inspector.get_columns("processed_claims")
    }
    if "claim_set_version_id" not in processed_claims_columns:
        op.add_column(
            "processed_claims",
            sa.Column("claim_set_version_id", sa.Integer(), nullable=True),
        )
    if "canonical_hash" not in processed_claims_columns:
        op.add_column(
            "processed_claims",
            sa.Column("canonical_hash", sa.String(length=128), nullable=True),
        )
    if "claim_group_id" not in processed_claims_columns:
        op.add_column(
            "processed_claims",
            sa.Column("claim_group_id", sa.String(length=128), nullable=True),
        )
    if "up_votes" not in processed_claims_columns:
        op.add_column(
            "processed_claims",
            sa.Column("up_votes", sa.Integer(), nullable=False, server_default=sa.text("0")),
        )
    if "down_votes" not in processed_claims_columns:
        op.add_column(
            "processed_claims",
            sa.Column("down_votes", sa.Integer(), nullable=False, server_default=sa.text("0")),
        )

    inspector = sa.inspect(bind)
    processed_claims_columns = {
        column["name"]: column for column in inspector.get_columns("processed_claims")
    }
    processed_claims_indexes = {idx["name"] for idx in inspector.get_indexes("processed_claims")}
    if "ix_processed_claims_claim_set_version_id" not in processed_claims_indexes:
        op.create_index(
            "ix_processed_claims_claim_set_version_id",
            "processed_claims",
            ["claim_set_version_id"],
        )

    processed_claims_fk_defs = inspector.get_foreign_keys("processed_claims")
    has_version_fk = any(
        set(fk.get("constrained_columns", [])) == {"claim_set_version_id"}
        and fk.get("referred_table") == version_table_name
        for fk in processed_claims_fk_defs
    )
    if not has_version_fk and dialect_name != "sqlite":
        op.create_foreign_key(
            "fk_processed_claims_claim_set_version_id",
            "processed_claims",
            version_table_name,
            ["claim_set_version_id"],
            ["id"],
            ondelete="CASCADE",
        )

    feedback_table_name = "processed_claim_feedback"
    if not inspector.has_table(feedback_table_name):
        op.create_table(
            feedback_table_name,
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column(
                "claim_id",
                sa.Integer(),
                sa.ForeignKey("processed_claims.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column("client_token", sa.String(length=128), nullable=False),
            sa.Column("vote", sa.String(length=8), nullable=False),
            sa.Column("comment", sa.Text(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
            sa.UniqueConstraint("claim_id", "client_token", name="uq_feedback_claim_token"),
        )

    inspector = sa.inspect(bind)
    feedback_indexes = set()
    if inspector.has_table(feedback_table_name):
        feedback_indexes = {idx["name"] for idx in inspector.get_indexes(feedback_table_name)}
    if "ix_processed_claim_feedback_claim_id" not in feedback_indexes:
        op.create_index(
            "ix_processed_claim_feedback_claim_id",
            feedback_table_name,
            ["claim_id"],
        )

    metadata = sa.MetaData()
    metadata.reflect(
        bind=bind,
        only=["processed_claim_set_versions", "processed_claims", "processed_claim_sets"],
    )

    versions_table = metadata.tables["processed_claim_set_versions"]
    claims_table = metadata.tables["processed_claims"]
    claim_sets_table = metadata.tables["processed_claim_sets"]

    utc_now = datetime.now(timezone.utc)

    claim_set_rows = list(bind.execute(sa.select(claim_sets_table.c.id)))
    for claim_set_row in claim_set_rows:
        existing_version = bind.execute(
            sa.select(versions_table.c.id)
            .where(versions_table.c.claim_set_id == claim_set_row.id)
            .where(versions_table.c.version_number == 1)
        ).fetchone()

        if existing_version:
            version_id = existing_version.id
        else:
            result = bind.execute(
                versions_table.insert().values(
                    claim_set_id=claim_set_row.id,
                    version_number=1,
                    status="active",
                    pipeline_metadata={},
                    created_at=utc_now,
                    activated_at=utc_now,
                )
            )
            version_id = result.inserted_primary_key[0]

        bind.execute(
            claims_table.update()
            .where(claims_table.c.claim_set_id == claim_set_row.id)
            .where(claims_table.c.claim_set_version_id.is_(None))
            .values(claim_set_version_id=version_id)
        )

    if "canonical_hash" in processed_claims_columns:
        op.execute(
            sa.text(
                "UPDATE processed_claims SET canonical_hash = claim_id WHERE canonical_hash IS NULL"
            )
        )
    if "claim_group_id" in processed_claims_columns:
        op.execute(
            sa.text(
                "UPDATE processed_claims SET claim_group_id = claim_id WHERE claim_group_id IS NULL"
            )
        )
    if "up_votes" in processed_claims_columns:
        op.execute(
            sa.text(
                "UPDATE processed_claims SET up_votes = 0 WHERE up_votes IS NULL"
            )
        )
    if "down_votes" in processed_claims_columns:
        op.execute(
            sa.text(
                "UPDATE processed_claims SET down_votes = 0 WHERE down_votes IS NULL"
            )
        )
    if "claim_set_version_id" in processed_claims_columns:
        op.execute(
            sa.text(
                """
                UPDATE processed_claims
                SET claim_set_version_id = (
                    SELECT MAX(id)
                    FROM processed_claim_set_versions
                    WHERE processed_claim_set_versions.claim_set_id = processed_claims.claim_set_id
                )
                WHERE claim_set_version_id IS NULL
                """
            )
        )

    inspector = sa.inspect(bind)
    processed_claims_columns = {
        column["name"]: column for column in inspector.get_columns("processed_claims")
    }
    if (
        "claim_set_version_id" in processed_claims_columns
        and processed_claims_columns["claim_set_version_id"].get("nullable", True)
    ):
        op.alter_column(
            "processed_claims",
            "claim_set_version_id",
            existing_type=sa.Integer(),
            nullable=False,
        )
    if "canonical_hash" in processed_claims_columns and processed_claims_columns["canonical_hash"].get("nullable", True):
        op.alter_column(
            "processed_claims",
            "canonical_hash",
            existing_type=sa.String(length=128),
            nullable=False,
        )
    if "claim_group_id" in processed_claims_columns and processed_claims_columns["claim_group_id"].get("nullable", True):
        op.alter_column(
            "processed_claims",
            "claim_group_id",
            existing_type=sa.String(length=128),
            nullable=False,
        )

    processed_claims_uniques = {
        constraint["name"] for constraint in inspector.get_unique_constraints("processed_claims")
    }
    if "uq_claim_version_hash" not in processed_claims_uniques:
        op.create_unique_constraint(
            "uq_claim_version_hash",
            "processed_claims",
            ["claim_set_version_id", "canonical_hash"],
        )


def downgrade() -> None:
    bind = op.get_bind()
    dialect_name = bind.dialect.name
    inspector = sa.inspect(bind)

    processed_claims_uniques = {
        constraint["name"] for constraint in inspector.get_unique_constraints("processed_claims")
    }
    processed_claims_columns = {
        column["name"] for column in inspector.get_columns("processed_claims")
    }
    processed_claims_indexes = {
        index["name"] for index in inspector.get_indexes("processed_claims")
    }
    processed_claims_foreign_keys = {
        fk["name"]
        for fk in inspector.get_foreign_keys("processed_claims")
        if fk.get("name")
    }

    if inspector.has_table("processed_claim_feedback"):
        feedback_indexes = {
            index["name"] for index in inspector.get_indexes("processed_claim_feedback")
        }
        if "ix_processed_claim_feedback_claim_id" in feedback_indexes:
            op.drop_index("ix_processed_claim_feedback_claim_id", table_name="processed_claim_feedback")
        op.drop_table("processed_claim_feedback")

    if "ix_processed_claims_claim_set_version_id" in processed_claims_indexes:
        op.drop_index("ix_processed_claims_claim_set_version_id", table_name="processed_claims")

    if dialect_name == "sqlite":
        with op.batch_alter_table("processed_claims", schema=None) as batch_op:
            if "uq_claim_version_hash" in processed_claims_uniques:
                batch_op.drop_constraint("uq_claim_version_hash", type_="unique")
            for column_name in [
                "down_votes",
                "up_votes",
                "claim_group_id",
                "canonical_hash",
                "claim_set_version_id",
            ]:
                if column_name in processed_claims_columns:
                    batch_op.drop_column(column_name)
    else:
        if "uq_claim_version_hash" in processed_claims_uniques:
            op.drop_constraint("uq_claim_version_hash", "processed_claims", type_="unique")
        if "fk_processed_claims_claim_set_version_id" in processed_claims_foreign_keys:
            op.drop_constraint(
                "fk_processed_claims_claim_set_version_id",
                "processed_claims",
                type_="foreignkey",
            )
        for column_name in [
            "down_votes",
            "up_votes",
            "claim_group_id",
            "canonical_hash",
            "claim_set_version_id",
        ]:
            if column_name in processed_claims_columns:
                op.drop_column("processed_claims", column_name)

    if inspector.has_table("processed_claim_set_versions"):
        version_indexes = {
            index["name"] for index in inspector.get_indexes("processed_claim_set_versions")
        }
        if "ix_processed_claim_set_versions_claim_set_id" in version_indexes:
            op.drop_index(
                "ix_processed_claim_set_versions_claim_set_id",
                table_name="processed_claim_set_versions",
            )
        op.drop_table("processed_claim_set_versions")
