from __future__ import annotations

import hashlib
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
    builder = mesh_builder or NIHMeshBuilder()
    espell = espell_client or NIHESpellClient()
    searcher = pubmed_searcher or NIHPubMedSearcher()
    suggestion_client = mesh_suggestion_client or NIHMeshSuggestionClient()
    cached_signatures: Dict[Tuple[str, ...], str] = {}
    settings = get_app_settings()
    refresh_ttl = refresh_ttl_seconds or settings.search.refresh_ttl_seconds

    def _result_signature_provider(mesh_terms: list[str], _: str) -> str:
        if not mesh_terms:
            return ""
        key = tuple(mesh_terms)
        if key not in cached_signatures:
            result = searcher(mesh_terms)
            fingerprint = "|".join(result.pmids)
            cached_signatures[key] = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()
        return cached_signatures[key]

    resolution = resolve_search_input(
        raw_condition,
        session=session,
        mesh_builder=builder,
        espell_fetcher=espell,
        refresh_ttl_seconds=refresh_ttl,
        result_signature_provider=_result_signature_provider,
    )

    if not resolution.mesh_terms:
        suggestions: list[str] = []
        for candidate in {raw_condition, resolution.normalized_condition}:
            suggestions.extend(suggestion_client.suggest(candidate))
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

    return resolution
