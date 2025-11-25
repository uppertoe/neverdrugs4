from __future__ import annotations

import logging
import os
import secrets
from typing import Callable, Mapping, Optional

from flask import Flask, Response, jsonify, g, redirect, url_for
from sqlalchemy.orm import Session

from app.api import api_blueprint
from app.database import create_session_factory
from app.job_queue import configure_claim_refresh_enqueuer
from app.settings import DEFAULT_SEARCH_REFRESH_TTL_SECONDS, load_settings
from app.tasks import refresh_claims_for_condition_task
from app.ui import ui_blueprint

SessionFactory = Callable[[], Session]
logger = logging.getLogger(__name__)


def _enqueue_refresh_via_celery(
    *,
    session: Session,
    resolution,
    condition_label: str,
    mesh_signature: str | None,
) -> Mapping[str, object]:
    """Dispatch the refresh pipeline through Celery."""
    _ = session  # session is managed by request lifecycle; Celery job reopens its own.
    async_result = refresh_claims_for_condition_task.apply_async(
        kwargs={
            "resolution_id": resolution.search_term_id,
            "condition_label": condition_label,
            "normalized_condition": resolution.normalized_condition,
            "mesh_terms": list(resolution.mesh_terms),
            "mesh_signature": mesh_signature,
        }
    )
    return {"job_id": async_result.id, "status": "queued"}


def create_app(
    *,
    session_factory: SessionFactory | None = None,
    config: Optional[Mapping[str, object]] = None,
) -> Flask:
    app = Flask(__name__)

    if config:
        app.config.from_mapping(config)  # type: ignore[arg-type]

    env_name = str(app.config.get("ENV") or os.getenv("FLASK_ENV") or "").lower()
    debug_mode = bool(app.config.get("DEBUG"))
    testing_mode = bool(app.config.get("TESTING"))
    is_dev_environment = testing_mode or debug_mode or env_name in {"development", "debug"}

    secret_key = app.config.get("SECRET_KEY")
    if not secret_key:
        env_secret_key = os.getenv("SECRET_KEY")
        if env_secret_key:
            secret_key = env_secret_key.strip()
            app.config["SECRET_KEY"] = secret_key
    if not secret_key:
        if is_dev_environment:
            app.config["SECRET_KEY"] = secrets.token_urlsafe(32)
            logger.warning("SECRET_KEY not provided; generated ephemeral key for development/testing")
        else:
            raise RuntimeError("SECRET_KEY is required in production. Set it via environment or config.")
    else:
        insecure_values = {"change-me", "not-set", "your-secret"}
        if not is_dev_environment and (secret_key in insecure_values or len(str(secret_key)) < 16):
            raise RuntimeError("SECRET_KEY is not secure enough for production. Provide a stronger value.")

    if not is_dev_environment:
        app.config.setdefault("SESSION_COOKIE_SECURE", True)
        app.config.setdefault("REMEMBER_COOKIE_SECURE", True)
        app.config.setdefault("SESSION_COOKIE_HTTPONLY", True)
        app.config.setdefault("SESSION_COOKIE_SAMESITE", "Strict")
        app.config.setdefault("PREFERRED_URL_SCHEME", "https")
    else:
        # Even in development/testing, keep cookies HTTP-only by default.
        app.config.setdefault("SESSION_COOKIE_HTTPONLY", True)

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

    @app.get("/")
    def root_redirect() -> Response:
        return redirect(url_for("ui.index"))

    app.register_blueprint(api_blueprint)
    app.register_blueprint(ui_blueprint)
    configure_claim_refresh_enqueuer(_enqueue_refresh_via_celery)

    return app
