from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping, Any

DEFAULT_SEARCH_REFRESH_TTL_SECONDS = 7 * 24 * 60 * 60


@dataclass(frozen=True)
class SearchSettings:
    refresh_ttl_seconds: int = DEFAULT_SEARCH_REFRESH_TTL_SECONDS


@dataclass(frozen=True)
class AppSettings:
    search: SearchSettings = SearchSettings()


def _coerce_positive_int(value: Any, default: int) -> int:
    try:
        coerced = int(value)
    except (TypeError, ValueError):
        return default
    if coerced <= 0:
        return default
    return coerced


def load_settings(config: Mapping[str, Any] | None = None) -> AppSettings:
    refresh_ttl = DEFAULT_SEARCH_REFRESH_TTL_SECONDS

    env_value = os.getenv("SEARCH_REFRESH_TTL_SECONDS")
    if env_value is not None:
        refresh_ttl = _coerce_positive_int(env_value, refresh_ttl)

    if config is not None and "SEARCH_REFRESH_TTL_SECONDS" in config:
        refresh_ttl = _coerce_positive_int(config["SEARCH_REFRESH_TTL_SECONDS"], refresh_ttl)

    return AppSettings(search=SearchSettings(refresh_ttl_seconds=refresh_ttl))


def get_app_settings(config: Mapping[str, Any] | None = None) -> AppSettings:
    try:
        from flask import current_app, has_app_context
    except ImportError:  # pragma: no cover - flask not installed in some contexts
        return load_settings(config)

    if has_app_context():
        app = current_app
        cached = app.extensions.get("app_settings")
        if cached is None:
            cached = load_settings(app.config)
            app.extensions["app_settings"] = cached
        return cached

    return load_settings(config)
