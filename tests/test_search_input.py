from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.models import SearchArtefact, SearchTerm, SearchTermVariant
from app.services.search import MeshBuildResult, compute_mesh_signature, resolve_search_input


class RecordingBuilder:
    def __init__(self, mesh_terms: list[str] | None = None) -> None:
        self.mesh_terms = mesh_terms or []
        self.calls: list[str] = []

    def __call__(self, normalized_condition: str) -> MeshBuildResult:
        self.calls.append(normalized_condition)
        return MeshBuildResult(
            mesh_terms=self.mesh_terms,
            query_payload={"term": normalized_condition},
            source="test-double",
        )


class RecordingESpell:
    def __init__(self, suggestion: str | None = None) -> None:
        self.suggestion = suggestion
        self.calls: list[str] = []

    def __call__(self, term: str) -> str | None:
        self.calls.append(term)
        return self.suggestion


def seed_cached_search(session: Session) -> SearchTerm:
    term = SearchTerm(
        canonical="duchenne",
        created_at=datetime.now(timezone.utc),
    )
    term.variants.append(
        SearchTermVariant(
            value="Duchenne",
            normalized_value="duchenne",
        )
    )
    artefact = SearchArtefact(
        query_payload={"term": "duchenne"},
        mesh_terms=["Duchenne Muscular Dystrophy"],
        last_refreshed_at=datetime.now(timezone.utc),
        ttl_policy_seconds=86_400,
        mesh_signature=compute_mesh_signature(["Duchenne Muscular Dystrophy"]),
    )
    term.artefacts.append(artefact)
    session.add(term)
    session.commit()
    return term


def test_resolve_search_reuses_cached_mesh_terms(session: Session) -> None:
    seed_cached_search(session)
    builder = RecordingBuilder()
    espell = RecordingESpell()

    result = resolve_search_input(
        "Duchenne",
        session=session,
        mesh_builder=builder,
        espell_fetcher=espell,
    )

    assert result.mesh_terms == ["Duchenne Muscular Dystrophy"]
    assert result.reused_cached is True
    assert builder.calls == []
    assert espell.calls == ["duchenne"]


def test_resolve_search_creates_mesh_query_when_cache_miss(session: Session) -> None:
    builder = RecordingBuilder(mesh_terms=["Novel Condition"])
    espell = RecordingESpell()

    result = resolve_search_input(
        " Novel Condition ",
        session=session,
        mesh_builder=builder,
        espell_fetcher=espell,
    )

    assert builder.calls == ["novel condition"]
    assert result.mesh_terms == ["Novel Condition"]
    assert result.reused_cached is False
    assert espell.calls == ["novel condition"]

    stored_term = session.query(SearchTerm).filter_by(canonical="novel condition").one()
    assert any(variant.value == " Novel Condition " for variant in stored_term.variants)
    artefact = stored_term.artefacts[0]
    assert artefact.mesh_terms == ["Novel Condition"]


def test_resolve_search_links_to_existing_term_via_mesh_similarity(session: Session) -> None:
    existing_term = seed_cached_search(session)
    builder = RecordingBuilder(mesh_terms=["Duchenne Muscular Dystrophy"])
    espell = RecordingESpell()

    result = resolve_search_input(
        "Duchenne muscular dystrophy",
        session=session,
        mesh_builder=builder,
        espell_fetcher=espell,
    )

    assert result.search_term_id == existing_term.id
    assert builder.calls == ["duchenne muscular dystrophy"]
    assert espell.calls == ["duchenne muscular dystrophy"]

    refreshed = session.query(SearchTerm).filter_by(id=existing_term.id).one()
    variant_values = {variant.value for variant in refreshed.variants}
    assert {"Duchenne", "Duchenne muscular dystrophy"}.issubset(variant_values)
    assert len(refreshed.artefacts) == 1


def test_resolve_search_applies_espell_correction(session: Session) -> None:
    builder = RecordingBuilder(mesh_terms=["Duchenne Muscular Dystrophy"])
    espell = RecordingESpell("Duchenne")

    result = resolve_search_input(
        "Duchene",
        session=session,
        mesh_builder=builder,
        espell_fetcher=espell,
    )

    assert espell.calls == ["duchene"]
    assert builder.calls == ["duchenne"]
    assert result.normalized_condition == "duchenne"

    stored_term = session.query(SearchTerm).filter_by(canonical="duchenne").one()
    variant_values = {variant.value for variant in stored_term.variants}
    assert "Duchene" in variant_values
    assert stored_term.canonical == "duchenne"