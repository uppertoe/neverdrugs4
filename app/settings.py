from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping, Any

DEFAULT_SEARCH_REFRESH_TTL_SECONDS = 7 * 24 * 60 * 60
DEFAULT_BASE_FULL_TEXT_ARTICLES = 50
DEFAULT_MAX_FULL_TEXT_ARTICLES = 100
DEFAULT_PUBMED_RETMAX = 200
DEFAULT_NIH_CONTACT_EMAIL = "neverdrugs@example.com"
DEFAULT_NIH_API_KEY = ""


@dataclass(frozen=True)
class SearchSettings:
    refresh_ttl_seconds: int = DEFAULT_SEARCH_REFRESH_TTL_SECONDS


@dataclass(frozen=True)
class ArticleSelectionSettings:
    base_full_text_articles: int = DEFAULT_BASE_FULL_TEXT_ARTICLES
    max_full_text_articles: int = DEFAULT_MAX_FULL_TEXT_ARTICLES
    pubmed_retmax: int = DEFAULT_PUBMED_RETMAX


@dataclass(frozen=True)
class AppSettings:
    search: SearchSettings = SearchSettings()
    article_selection: ArticleSelectionSettings = ArticleSelectionSettings()
    nih_contact_email: str = DEFAULT_NIH_CONTACT_EMAIL
    nih_api_key: str = DEFAULT_NIH_API_KEY


def _coerce_positive_int(value: Any, default: int) -> int:
    try:
        coerced = int(value)
    except (TypeError, ValueError):
        return default
    if coerced <= 0:
        return default
    return coerced


def load_settings(config: Mapping[str, Any] | None = None) -> AppSettings:
    def _resolve(name: str, default: int) -> int:
        env_val = os.getenv(name)
        if env_val is not None:
            return _coerce_positive_int(env_val, default)
        if config is not None and name in config:
            return _coerce_positive_int(config[name], default)
        return default

    def _resolve_str(name: str, default: str) -> str:
        env_val = os.getenv(name)
        if isinstance(env_val, str) and env_val.strip():
            return env_val.strip()
        if config is not None:
            value = config.get(name)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return default

    refresh_ttl = _resolve("SEARCH_REFRESH_TTL_SECONDS", DEFAULT_SEARCH_REFRESH_TTL_SECONDS)
    base_full_text = _resolve("FULL_TEXT_BASE_ARTICLES", DEFAULT_BASE_FULL_TEXT_ARTICLES)
    max_full_text = _resolve("FULL_TEXT_MAX_ARTICLES", DEFAULT_MAX_FULL_TEXT_ARTICLES)
    pubmed_retmax = _resolve("PUBMED_RETMAX", DEFAULT_PUBMED_RETMAX)
    contact_email = _resolve_str("NIH_CONTACT_EMAIL", DEFAULT_NIH_CONTACT_EMAIL)

    api_key_env = os.getenv("NIH_API_KEY") or os.getenv("NCBI_API_KEY")
    if config is not None and not api_key_env:
        config_key = config.get("NIH_API_KEY") or config.get("NCBI_API_KEY")
        api_key_env = config_key if isinstance(config_key, str) else None
    api_key = api_key_env.strip() if isinstance(api_key_env, str) and api_key_env.strip() else DEFAULT_NIH_API_KEY

    return AppSettings(
        search=SearchSettings(refresh_ttl_seconds=refresh_ttl),
        article_selection=ArticleSelectionSettings(
            base_full_text_articles=base_full_text,
            max_full_text_articles=max_full_text,
            pubmed_retmax=pubmed_retmax,
        ),
        nih_contact_email=contact_email,
        nih_api_key=api_key,
    )


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
