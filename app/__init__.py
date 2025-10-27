from __future__ import annotations

from typing import Callable, Mapping, Optional

from flask import Flask, jsonify, g
from sqlalchemy.orm import Session

from app.api import api_blueprint
from app.database import create_session_factory
from app.settings import DEFAULT_SEARCH_REFRESH_TTL_SECONDS, load_settings

SessionFactory = Callable[[], Session]


def create_app(
    *,
    session_factory: SessionFactory | None = None,
    config: Optional[Mapping[str, object]] = None,
) -> Flask:
    app = Flask(__name__)

    if config:
        app.config.from_mapping(config)  # type: ignore[arg-type]

    app.config.setdefault(
        "SEARCH_REFRESH_TTL_SECONDS",
        DEFAULT_SEARCH_REFRESH_TTL_SECONDS,
    )

    app.extensions["app_settings"] = load_settings(app.config)

    factory = session_factory or create_session_factory()
    app.extensions["session_factory"] = factory

    @app.before_request
    def _open_db_session() -> None:
        if "db_session" not in g:
            g.db_session = factory()

    @app.teardown_request
    def _cleanup_db_session(exc: BaseException | None) -> None:
        session = g.pop("db_session", None)
        if session is None:
            return
        try:
            if exc is None:
                session.commit()
            else:
                session.rollback()
        finally:
            session.close()

    @app.get("/health")
    def health() -> tuple[dict[str, str], int]:
        # Basic readiness probe for infrastructure tests
        return jsonify(status="ok"), 200

    app.register_blueprint(api_blueprint)

    return app
