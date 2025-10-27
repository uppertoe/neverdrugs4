from __future__ import annotations

import hashlib
import re
import unicodedata

_DEFAULT_HASH_LENGTH = 8
_SLUG_CLEAN_RE = re.compile(r"[^a-z0-9]+")


def _normalise_text(value: str | None) -> str:
    if not value:
        return ""
    normalized = unicodedata.normalize("NFKD", value)
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    return ascii_value.lower().strip()


def slugify(value: str | None) -> str:
    lowered = _normalise_text(value)
    if not lowered:
        return ""
    cleaned = _SLUG_CLEAN_RE.sub("-", lowered)
    cleaned = cleaned.strip("-")
    cleaned = re.sub("-+", "-", cleaned)
    return cleaned


def short_hash(*values: str | None, length: int = _DEFAULT_HASH_LENGTH) -> str:
    joined = "::".join(value or "" for value in values)
    digest = hashlib.sha1(joined.encode("utf-8")).hexdigest()
    length = max(4, min(length, len(digest)))
    return digest[:length]


def build_search_term_slug(canonical_condition: str | None) -> str:
    base = slugify(canonical_condition)
    suffix = short_hash(canonical_condition, length=8)
    slug = f"{base}--{suffix}" if base else suffix
    return slug[:255]


def build_claim_set_slug(condition_label: str | None, mesh_signature: str | None) -> str:
    base = slugify(condition_label)
    suffix_source = mesh_signature or condition_label or base or "claim-set"
    suffix = short_hash(suffix_source, length=8)
    slug = f"{base}--{suffix}" if base else suffix
    return slug[:255]
