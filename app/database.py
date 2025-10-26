from __future__ import annotations

import os
from typing import Callable

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


class Base(DeclarativeBase):
    """Declarative base for SQLAlchemy models."""


def get_database_url() -> str:
    return os.getenv("DATABASE_URL", "sqlite+pysqlite:///:memory:")


def create_engine_for_url(url: str | None = None):
    return create_engine(url or get_database_url(), future=True)


def create_session_factory(url: str | None = None) -> sessionmaker[Session]:
    engine = create_engine_for_url(url)
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)


def get_sessionmaker() -> sessionmaker[Session]:
    return create_session_factory()
