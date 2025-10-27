from __future__ import annotations

import pathlib

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect


def _configured_alembic(db_url: str) -> Config:
    project_root = pathlib.Path(__file__).resolve().parents[1]
    cfg = Config(project_root / "alembic.ini")
    cfg.set_main_option("script_location", str(project_root / "migrations"))
    cfg.set_main_option("sqlalchemy.url", db_url)
    return cfg


def test_initial_migration_creates_core_tables(tmp_path):
    db_path = tmp_path / "alembic_test.db"
    db_url = f"sqlite:///{db_path}"

    cfg = _configured_alembic(db_url)
    command.upgrade(cfg, "head")

    engine = create_engine(db_url, future=True)
    inspector = inspect(engine)

    table_names = set(inspector.get_table_names())
    assert "claim_set_refreshes" in table_names
    assert "processed_claim_sets" in table_names
    assert "search_terms" in table_names

    command.downgrade(cfg, "base")
    inspector = inspect(engine)
    table_names_post = set(inspector.get_table_names())
    assert "claim_set_refreshes" not in table_names_post
