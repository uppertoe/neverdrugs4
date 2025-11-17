from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

import pytest

from app.services.mesh_resolution import MeshResolutionPreview, preview_mesh_resolution
from app.services.search import MeshBuildResult


@dataclass
class _StubMeshBuilder:
    result: MeshBuildResult
    calls: List[str]

    def __call__(self, normalized_term: str) -> MeshBuildResult:
        self.calls.append(normalized_term)
        return self.result


@dataclass
class _StubESpell:
    suggestion: str | None
    calls: List[str]

    def __call__(self, term: str) -> str | None:
        self.calls.append(term)
        return self.suggestion


@dataclass
class _StubSuggestions:
    responses: list[str]
    calls: List[str]

    def suggest(self, label: str) -> list[str]:
        self.calls.append(label)
        return list(self.responses)


def _build_result(
    mesh_terms: list[str],
    ranked: list[str] | None = None,
    selected: list[str] | None = None,
) -> MeshBuildResult:
    ranked_entries = [{"term": term} for term in ranked or []]
    payload: dict[str, Any] = {"ranked_mesh_terms": ranked_entries}
    canonical = mesh_terms[0] if mesh_terms else None
    if canonical:
        payload["canonical_mesh_term"] = canonical
    if selected is not None:
        payload["selected_mesh_terms"] = list(selected)
    elif ranked:
        payload["selected_mesh_terms"] = list(ranked)
    return MeshBuildResult(mesh_terms=mesh_terms, query_payload=payload, source="test")


def test_preview_resolution_applies_espell_correction() -> None:
    builder = _StubMeshBuilder(result=_build_result(["Porphyria Variegata"], ["Porphyria Variegata"]), calls=[])
    espell = _StubESpell(suggestion="Porphyria Variegata", calls=[])
    suggestions = _StubSuggestions(responses=[], calls=[])

    preview = preview_mesh_resolution(
        "Porphyria",
        mesh_builder=builder,
        espell_client=espell,
        suggestion_client=suggestions,
    )

    assert preview.normalized_query == "porphyria variegata"
    assert preview.espell_correction == "Porphyria Variegata"
    assert builder.calls == ["porphyria variegata"]
    assert espell.calls == ["porphyria"]
    assert suggestions.calls == []
    assert preview.status == "resolved"


def test_preview_resolution_with_single_term_resolves_immediately() -> None:
    builder = _StubMeshBuilder(
        result=_build_result(["Porphyria Variegata"], ["Porphyria Variegata"]),
        calls=[],
    )
    espell = _StubESpell(suggestion=None, calls=[])
    suggestions = _StubSuggestions(responses=[], calls=[])

    preview = preview_mesh_resolution(
        "Porphyria Variegata",
        mesh_builder=builder,
        espell_client=espell,
        suggestion_client=suggestions,
    )

    assert preview.status == "resolved"
    assert preview.mesh_terms == ["Porphyria Variegata"]
    assert preview.ranked_options == ["Porphyria Variegata"]
    assert preview.suggestions == []
    assert suggestions.calls == []


def test_preview_resolution_with_aliases_resolves_using_canonical() -> None:
    builder = _StubMeshBuilder(
        result=_build_result(
            ["Porphyria, Variegate"],
            ["Porphyria, Variegate", "Variegate Porphyria", "Porphyria Variegata"],
        ),
        calls=[],
    )
    espell = _StubESpell(suggestion=None, calls=[])
    suggestions = _StubSuggestions(responses=["Unused"], calls=[])

    preview = preview_mesh_resolution(
        "Porphyria",
        mesh_builder=builder,
        espell_client=espell,
        suggestion_client=suggestions,
    )

    assert preview.status == "resolved"
    assert preview.mesh_terms == ["Porphyria, Variegate"]
    assert preview.ranked_options == [
        "Porphyria, Variegate",
        "Porphyria Variegata",
    ]
    assert preview.suggestions == []
    assert suggestions.calls == []


def test_preview_resolution_with_multiple_canonical_terms_requests_clarification() -> None:
    builder = _StubMeshBuilder(
        result=_build_result(
            ["Acute Intermittent Porphyria", "Porphyria, Variegate"],
            ["Acute Intermittent Porphyria", "Porphyria, Variegate"],
        ),
        calls=[],
    )
    espell = _StubESpell(suggestion=None, calls=[])
    suggestions = _StubSuggestions(responses=["Unused"], calls=[])

    preview = preview_mesh_resolution(
        "Porphyria",
        mesh_builder=builder,
        espell_client=espell,
        suggestion_client=suggestions,
    )

    assert preview.status == "needs_clarification"
    assert preview.mesh_terms == ["Acute Intermittent Porphyria", "Porphyria, Variegate"]
    assert preview.ranked_options == ["Acute Intermittent Porphyria", "Porphyria, Variegate"]
    assert preview.suggestions == []
    assert suggestions.calls == []


def test_preview_resolution_without_terms_returns_suggestions() -> None:
    builder = _StubMeshBuilder(result=_build_result([], ["Porphyria Variegata"]), calls=[])
    espell = _StubESpell(suggestion=None, calls=[])
    suggestions = _StubSuggestions(responses=["Porphyria Variegata", "Porphyria Cutanea Tarda"], calls=[])

    preview = preview_mesh_resolution(
        "Porphyria",
        mesh_builder=builder,
        espell_client=espell,
        suggestion_client=suggestions,
    )

    assert preview.status == "not_found"
    assert preview.mesh_terms == []
    assert preview.ranked_options == ["Porphyria Variegata"]
    assert preview.suggestions == ["Porphyria Variegata", "Porphyria Cutanea Tarda"]
    assert suggestions.calls == ["porphyria"]
