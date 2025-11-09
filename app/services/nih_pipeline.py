from __future__ import annotations

import hashlib
import logging
from collections import defaultdict
from time import perf_counter
from typing import Dict, Tuple

from sqlalchemy.orm import Session

from app.services.espell import NIHESpellClient
from app.services.mesh_builder import NIHMeshBuilder
from app.services.mesh_suggestions import NIHMeshSuggestionClient
from app.services.nih_pubmed import NIHPubMedSearcher
from app.services.search import SearchResolution, resolve_search_input
from app.settings import get_app_settings


class MeshTermsNotFoundError(RuntimeError):
    """Raised when no MeSH terms could be resolved for a condition."""

    def __init__(
        self,
        *,
        normalized_condition: str,
        search_term_id: int,
        suggestions: list[str],
    ) -> None:
        message = f"No MeSH terms found for '{normalized_condition}'"
        super().__init__(message)
        self.normalized_condition = normalized_condition
        self.search_term_id = search_term_id
        self.suggestions = suggestions


logger = logging.getLogger(__name__)


def resolve_condition_via_nih(
    raw_condition: str,
    *,
    session: Session,
    mesh_builder: NIHMeshBuilder | None = None,
    espell_client: NIHESpellClient | None = None,
    pubmed_searcher: NIHPubMedSearcher | None = None,
    mesh_suggestion_client: NIHMeshSuggestionClient | None = None,
    refresh_ttl_seconds: int | None = None,
) -> SearchResolution:
    overall_start = perf_counter()
    timings: Dict[str, float] = defaultdict(float)
    call_counts: Dict[str, int] = defaultdict(int)

    def _wrap_callable(label: str, func):
        def _wrapped(*args, **kwargs):
            start = perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed = perf_counter() - start
                timings[label] += elapsed
                call_counts[label] += 1

        return _wrapped

    builder = mesh_builder or NIHMeshBuilder()
    espell = espell_client or NIHESpellClient()
    searcher = pubmed_searcher or NIHPubMedSearcher()
    suggestion_client = mesh_suggestion_client or NIHMeshSuggestionClient()
    cached_signatures: Dict[Tuple[str, ...], str] = {}
    settings = get_app_settings()
    refresh_ttl = refresh_ttl_seconds or settings.search.refresh_ttl_seconds

    timed_builder = _wrap_callable("mesh_builder", builder)
    timed_espell = _wrap_callable("espell", espell)

    def _result_signature_provider(mesh_terms: list[str], _: str) -> str:
        if not mesh_terms:
            return ""
        key = tuple(mesh_terms)
        if key not in cached_signatures:
            start = perf_counter()
            result = searcher(mesh_terms)
            timings["pubmed_search"] += perf_counter() - start
            call_counts["pubmed_search"] += 1
            fingerprint = "|".join(result.pmids)
            cached_signatures[key] = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()
        return cached_signatures[key]

    resolve_start = perf_counter()
    resolution = resolve_search_input(
        raw_condition,
        session=session,
        mesh_builder=timed_builder,
        espell_fetcher=timed_espell,
        refresh_ttl_seconds=refresh_ttl,
        result_signature_provider=_result_signature_provider,
    )
    timings["resolve_search_input"] += perf_counter() - resolve_start
    call_counts["resolve_search_input"] += 1

    if not resolution.mesh_terms:
        suggestions: list[str] = []
        for candidate in {raw_condition, resolution.normalized_condition}:
            start = perf_counter()
            suggestions.extend(suggestion_client.suggest(candidate))
            timings["mesh_suggestion"] += perf_counter() - start
            call_counts["mesh_suggestion"] += 1
        # Deduplicate while preserving order
        seen: set[str] = set()
        unique_suggestions = []
        for entry in suggestions:
            if entry not in seen:
                unique_suggestions.append(entry)
                seen.add(entry)
        raise MeshTermsNotFoundError(
            normalized_condition=resolution.normalized_condition,
            search_term_id=resolution.search_term_id,
            suggestions=unique_suggestions,
        )

    total_elapsed = perf_counter() - overall_start
    timings["total"] += total_elapsed
    call_counts["resolve_condition_via_nih"] += 1

    if logger.isEnabledFor(logging.INFO):
        logger.info(
            "Resolved condition via NIH",
            extra={
                "condition": raw_condition,
                "normalized_condition": resolution.normalized_condition,
                "mesh_terms_count": len(resolution.mesh_terms),
                "reused_cached": resolution.reused_cached,
                "timings": {key: round(value, 4) for key, value in timings.items()},
                "call_counts": dict(call_counts),
            },
        )

    return resolution
