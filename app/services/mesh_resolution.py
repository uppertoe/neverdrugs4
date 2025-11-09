from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

from app.services.espell import NIHESpellClient
from app.services.mesh_builder import NIHMeshBuilder
from app.services.mesh_suggestions import NIHMeshSuggestionClient
from app.services.search import MeshBuildResult, normalize_condition

MeshResolutionStatus = Literal["resolved", "needs_clarification", "not_found"]


@dataclass(slots=True)
class MeshResolutionPreview:
    status: MeshResolutionStatus
    raw_query: str
    normalized_query: str
    mesh_terms: list[str]
    ranked_options: list[str]
    suggestions: list[str]
    espell_correction: str | None = None


def preview_mesh_resolution(
    raw_query: str,
    *,
    mesh_builder: NIHMeshBuilder | None = None,
    espell_client: NIHESpellClient | None = None,
    suggestion_client: NIHMeshSuggestionClient | None = None,
) -> MeshResolutionPreview:
    builder = mesh_builder or NIHMeshBuilder()
    espell = espell_client or NIHESpellClient()
    suggestions_client = suggestion_client or NIHMeshSuggestionClient()

    normalized_input = normalize_condition(raw_query)
    espell_suggestion = espell(normalized_input)
    normalized_query = normalize_condition(espell_suggestion) if espell_suggestion else normalized_input

    build_result = builder(normalized_query)
    mesh_terms = list(build_result.mesh_terms)
    ranked_options = _extract_ranked_terms(build_result)

    if mesh_terms:
        status: MeshResolutionStatus = "resolved" if len(mesh_terms) == 1 else "needs_clarification"
        options = ranked_options or list(mesh_terms)
        return MeshResolutionPreview(
            status=status,
            raw_query=raw_query,
            normalized_query=normalized_query,
            mesh_terms=list(mesh_terms),
            ranked_options=options,
            suggestions=[],
            espell_correction=espell_suggestion,
        )

    suggestions = suggestions_client.suggest(normalized_query)
    return MeshResolutionPreview(
        status="not_found",
        raw_query=raw_query,
        normalized_query=normalized_query,
        mesh_terms=[],
        ranked_options=ranked_options,
        suggestions=suggestions,
        espell_correction=espell_suggestion,
    )


def _extract_ranked_terms(result: MeshBuildResult) -> list[str]:
    payload = result.query_payload or {}
    ranked_entries: Iterable[object] = payload.get("ranked_mesh_terms", []) if isinstance(payload, dict) else []
    terms: list[str] = []
    for entry in ranked_entries:
        term: str | None = None
        if isinstance(entry, dict):
            value = entry.get("term")
            term = str(value).strip() if value else None
        elif isinstance(entry, str):
            term = entry.strip() or None
        if term and term not in terms:
            terms.append(term)
    return terms
