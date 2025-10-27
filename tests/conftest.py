from __future__ import annotations

import pathlib
import sys

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import close_all_sessions, sessionmaker

project_root = pathlib.Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import app.models  # noqa: F401; ensure models are registered with Base.metadata
from app import create_app
from app.database import Base


@pytest.fixture()
def session_factory():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    try:
        yield SessionLocal
    finally:
        close_all_sessions()
        Base.metadata.drop_all(engine)
        engine.dispose()


@pytest.fixture()
def session(session_factory):
    db = session_factory()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture()
def app(session_factory):
    application = create_app(session_factory=session_factory, config={"TESTING": True})
    yield application


@pytest.fixture()
def client(app):
    return app.test_client()
