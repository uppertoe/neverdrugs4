from __future__ import annotations

from app import create_app
from app.settings import get_app_settings


def test_create_app_applies_refresh_ttl(session_factory) -> None:
    app = create_app(
        session_factory=session_factory,
        config={
            "TESTING": True,
            "SEARCH_REFRESH_TTL_SECONDS": 3600,
        },
    )

    with app.app_context():
        settings = get_app_settings()
        assert settings.search.refresh_ttl_seconds == 3600
        assert app.config["SEARCH_REFRESH_TTL_SECONDS"] == 3600
