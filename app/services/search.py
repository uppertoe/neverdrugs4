from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import SearchArtefact, SearchTerm, SearchTermVariant


DEFAULT_TTL_SECONDS = 86_400


@dataclass(slots=True)
class MeshBuildResult:
    mesh_terms: list[str]
    query_payload: dict[str, Any]
    source: str
    ttl_policy_seconds: int = DEFAULT_TTL_SECONDS


@dataclass(slots=True)
class SearchResolution:
    normalized_condition: str
    mesh_terms: list[str]
    reused_cached: bool
    search_term_id: int


def normalize_condition(raw: str) -> str:
    cleaned = unicodedata.normalize("NFKD", raw or "")
    ascii_only = cleaned.encode("ascii", "ignore").decode("ascii")
    return " ".join(ascii_only.strip().lower().split())


def compute_mesh_signature(mesh_terms: list[str]) -> str:
    if not mesh_terms:
        return ""
    normalized_terms = [normalize_condition(term) for term in mesh_terms if term]
    return "|".join(sorted(normalized_terms))


def resolve_search_input(
    raw_condition: str,
    *,
    session: Session,
    mesh_builder: Callable[[str], MeshBuildResult],
    espell_fetcher: Callable[[str], Optional[str]],
) -> SearchResolution:
    normalized_input = normalize_condition(raw_condition)
    espell_suggestion = espell_fetcher(normalized_input)
    normalized_suggestion = normalize_condition(espell_suggestion) if espell_suggestion else None
    search_normalized = normalized_suggestion or normalized_input

    search_term = _lookup_search_term(session, search_normalized)
    reused_cached = False
    artefact: SearchArtefact | None = None

    if search_term is not None:
        _ensure_variant_record(session, search_term, raw_condition, search_normalized)
        artefact = _select_freshest_artefact(search_term)

        if artefact is None:
            build_result = mesh_builder(search_normalized)
            signature = compute_mesh_signature(build_result.mesh_terms)
            artefact = _upsert_artefact(search_term, session, build_result, signature)
        else:
            reused_cached = True
    else:
        build_result = mesh_builder(search_normalized)
        signature = compute_mesh_signature(build_result.mesh_terms)

        search_term = _lookup_search_term_by_signature(session, signature)
        created_new_term = False
        if search_term is None:
            search_term = SearchTerm(canonical=search_normalized)
            session.add(search_term)
            session.flush()
            created_new_term = True

        _ensure_variant_record(session, search_term, raw_condition, search_normalized)
        artefact = _upsert_artefact(search_term, session, build_result, signature)
        reused_cached = not created_new_term

    session.flush()

    if artefact is None:
        # This should not happen, but guard to avoid returning incomplete data
        raise RuntimeError("Failed to resolve or persist MeSH artefact")

    return SearchResolution(
        normalized_condition=search_normalized,
        mesh_terms=list(artefact.mesh_terms),
        reused_cached=reused_cached,
        search_term_id=search_term.id,
    )


def _lookup_search_term(session: Session, normalized: str) -> SearchTerm | None:
    stmt = select(SearchTerm).where(SearchTerm.canonical == normalized)
    return session.execute(stmt).scalar_one_or_none()


def _lookup_search_term_by_signature(session: Session, signature: str) -> SearchTerm | None:
    if not signature:
        return None
    stmt = (
        select(SearchTerm)
        .join(SearchArtefact)
        .where(SearchArtefact.mesh_signature == signature)
        .limit(1)
    )
    return session.execute(stmt).scalar_one_or_none()


def _ensure_variant_record(
    session: Session,
    search_term: SearchTerm,
    raw_value: str,
    normalized: str,
) -> None:
    existing = next(
        (variant for variant in search_term.variants if variant.value == raw_value),
        None,
    )
    if existing is not None:
        return

    variant = SearchTermVariant(
        value=raw_value,
        normalized_value=normalized,
    )
    search_term.variants.append(variant)
    session.flush()


def _select_freshest_artefact(search_term: SearchTerm) -> SearchArtefact | None:
    if not search_term.artefacts:
        return None
    return max(
        search_term.artefacts,
        key=lambda artefact: artefact.last_refreshed_at or datetime.fromtimestamp(0, tz=timezone.utc),
    )


def _persist_mesh_result(
    search_term: SearchTerm,
    session: Session,
    build_result: MeshBuildResult,
) -> SearchArtefact:
    artefact = SearchArtefact(
        query_payload=build_result.query_payload,
        mesh_terms=build_result.mesh_terms,
        ttl_policy_seconds=build_result.ttl_policy_seconds or DEFAULT_TTL_SECONDS,
        last_refreshed_at=datetime.now(timezone.utc),
        mesh_signature=compute_mesh_signature(build_result.mesh_terms),
    )
    search_term.artefacts.append(artefact)
    session.flush()
    return artefact


def _upsert_artefact(
    search_term: SearchTerm,
    session: Session,
    build_result: MeshBuildResult,
    signature: str,
) -> SearchArtefact:
    matching = next(
        (artefact for artefact in search_term.artefacts if artefact.mesh_signature == signature),
        None,
    )

    if matching is not None:
        matching.query_payload = build_result.query_payload
        matching.mesh_terms = build_result.mesh_terms
        matching.ttl_policy_seconds = build_result.ttl_policy_seconds or DEFAULT_TTL_SECONDS
        matching.last_refreshed_at = datetime.now(timezone.utc)
        session.flush()
        return matching

    return _persist_mesh_result(search_term, session, build_result)
