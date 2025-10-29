from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone

from sqlalchemy.orm import Session

from app.models import SearchArtefact, SearchTerm, SearchTermVariant
from app.services.nih_pipeline import resolve_condition_via_nih
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


class RecordingResultHasher:
    def __init__(self, responses: dict[str, str] | None = None, default: str | None = None) -> None:
        self.responses = responses or {}
        self.default = default
        self.calls: list[tuple[list[str], str]] = []

    def __call__(self, mesh_terms: list[str], normalized_condition: str) -> str:
        self.calls.append((list(mesh_terms), normalized_condition))
        key = "|".join(mesh_terms)
        if key in self.responses:
            return self.responses[key]
        if self.default is not None:
            return self.default
        raise KeyError(f"No signature configured for mesh terms '{key}'")


class StubPubMedResult:
    def __init__(self, pmids: list[str]) -> None:
        self.pmids = pmids


class RecordingPubMedSearcher:
    def __init__(self, responses: dict[tuple[str, ...], StubPubMedResult]) -> None:
        self.responses = responses
        self.calls: list[tuple[list[str], tuple[str, ...] | None]] = []

    def __call__(
        self,
        condition_mesh_terms: list[str],
        *,
        additional_text_terms: list[str] | None = None,
    ) -> StubPubMedResult:
        key = tuple(condition_mesh_terms)
        self.calls.append((list(condition_mesh_terms), tuple(additional_text_terms) if additional_text_terms else None))
        return self.responses[key]


def seed_cached_search(
    session: Session,
    *,
    last_refreshed_at: datetime | None = None,
    mesh_terms: list[str] | None = None,
    result_signature: str | None = None,
) -> SearchTerm:
    mesh_terms = mesh_terms or ["Duchenne Muscular Dystrophy"]
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
        mesh_terms=mesh_terms,
        last_refreshed_at=last_refreshed_at or datetime.now(timezone.utc),
        ttl_policy_seconds=86_400,
        mesh_signature=compute_mesh_signature(mesh_terms),
        result_signature=result_signature or compute_mesh_signature(mesh_terms),
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


def test_resolve_search_refreshes_stale_cache_without_changes(session: Session) -> None:
    stale_timestamp = datetime.now(timezone.utc) - timedelta(days=8)
    cached_term = seed_cached_search(
        session,
        last_refreshed_at=stale_timestamp,
        result_signature="sig:v1",
    )
    builder = RecordingBuilder(mesh_terms=["Duchenne Muscular Dystrophy"])
    espell = RecordingESpell()
    hasher = RecordingResultHasher({"Duchenne Muscular Dystrophy": "sig:v1"})

    result = resolve_search_input(
        "Duchenne",
        session=session,
        mesh_builder=builder,
        espell_fetcher=espell,
        refresh_ttl_seconds=7 * 24 * 60 * 60,
        result_signature_provider=hasher,
    )

    assert builder.calls == ["duchenne"]
    assert result.reused_cached is True
    assert hasher.calls == [(["Duchenne Muscular Dystrophy"], "duchenne")]

    refreshed = session.query(SearchTerm).filter_by(id=cached_term.id).one()
    artefact = refreshed.artefacts[0]
    assert artefact.mesh_terms == ["Duchenne Muscular Dystrophy"]
    assert artefact.last_refreshed_at > stale_timestamp
    assert artefact.result_signature == "sig:v1"


def test_resolve_search_refreshes_stale_cache_with_changes(session: Session) -> None:
    stale_timestamp = datetime.now(timezone.utc) - timedelta(days=8)
    cached_term = seed_cached_search(
        session,
        last_refreshed_at=stale_timestamp,
        result_signature="sig:v1",
    )
    builder = RecordingBuilder(mesh_terms=["Updated Condition"])
    espell = RecordingESpell()
    hasher = RecordingResultHasher({"Updated Condition": "sig:v2"})

    result = resolve_search_input(
        "Duchenne",
        session=session,
        mesh_builder=builder,
        espell_fetcher=espell,
        refresh_ttl_seconds=7 * 24 * 60 * 60,
        result_signature_provider=hasher,
    )

    assert builder.calls == ["duchenne"]
    assert result.reused_cached is False
    assert hasher.calls == [(["Updated Condition"], "duchenne")]

    refreshed = session.query(SearchTerm).filter_by(id=cached_term.id).one()
    updated = next(
        artefact for artefact in refreshed.artefacts if artefact.mesh_terms == ["Updated Condition"]
    )
    assert updated.last_refreshed_at > stale_timestamp
    assert updated.mesh_signature == compute_mesh_signature(["Updated Condition"])
    assert updated.result_signature == "sig:v2"


def test_resolve_search_refresh_handles_naive_timestamp(session: Session) -> None:
    stale_timestamp = (datetime.now(timezone.utc) - timedelta(days=8)).replace(tzinfo=None)
    cached_term = seed_cached_search(
        session,
        last_refreshed_at=stale_timestamp,
        result_signature="sig:v1",
    )
    builder = RecordingBuilder(mesh_terms=["Duchenne Muscular Dystrophy"])
    espell = RecordingESpell()
    hasher = RecordingResultHasher({"Duchenne Muscular Dystrophy": "sig:v1"})

    result = resolve_search_input(
        "Duchenne",
        session=session,
        mesh_builder=builder,
        espell_fetcher=espell,
        refresh_ttl_seconds=7 * 24 * 60 * 60,
        result_signature_provider=hasher,
    )

    assert result.reused_cached is True
    refreshed = session.query(SearchTerm).filter_by(id=cached_term.id).one()
    artefact = refreshed.artefacts[0]
    refreshed_at = artefact.last_refreshed_at
    comparison_baseline = stale_timestamp
    if refreshed_at.tzinfo is not None and comparison_baseline.tzinfo is None:
        comparison_baseline = comparison_baseline.replace(tzinfo=refreshed_at.tzinfo)
    assert refreshed_at > comparison_baseline


def test_resolve_condition_via_nih_reuses_cached_when_ranked_results_identical(
    session: Session,
) -> None:
    mesh_terms = ["Duchenne Muscular Dystrophy"]
    pmids = ["111", "222"]
    fingerprint = "|".join(pmids)
    signature = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()
    stale_timestamp = datetime.now(timezone.utc) - timedelta(days=8)
    search_term = seed_cached_search(
        session,
        mesh_terms=mesh_terms,
        last_refreshed_at=stale_timestamp,
        result_signature=signature,
    )

    builder = RecordingBuilder(mesh_terms=mesh_terms)
    espell = RecordingESpell()
    searcher = RecordingPubMedSearcher({tuple(mesh_terms): StubPubMedResult(pmids)})

    result = resolve_condition_via_nih(
        "Duchenne",
        session=session,
        mesh_builder=builder,
        espell_client=espell,
        pubmed_searcher=searcher,
    )

    assert result.reused_cached is True
    assert builder.calls == ["duchenne"]
    assert len(searcher.calls) == 1

    refreshed = session.get(SearchTerm, search_term.id)
    assert refreshed is not None
    artefact = refreshed.artefacts[0]
    assert artefact.result_signature == signature
    assert artefact.last_refreshed_at > stale_timestamp


def test_resolve_condition_via_nih_detects_changed_ranked_results(session: Session) -> None:
    mesh_terms = ["Duchenne Muscular Dystrophy"]
    pmids_original = ["111", "222"]
    original_signature = hashlib.sha256("|".join(pmids_original).encode("utf-8")).hexdigest()
    stale_timestamp = datetime.now(timezone.utc) - timedelta(days=8)
    search_term = seed_cached_search(
        session,
        mesh_terms=mesh_terms,
        last_refreshed_at=stale_timestamp,
        result_signature=original_signature,
    )

    pmids_updated = ["111", "333"]
    updated_signature = hashlib.sha256("|".join(pmids_updated).encode("utf-8")).hexdigest()

    builder = RecordingBuilder(mesh_terms=mesh_terms)
    espell = RecordingESpell()
    searcher = RecordingPubMedSearcher({tuple(mesh_terms): StubPubMedResult(pmids_updated)})

    result = resolve_condition_via_nih(
        "Duchenne",
        session=session,
        mesh_builder=builder,
        espell_client=espell,
        pubmed_searcher=searcher,
    )

    assert result.reused_cached is False
    assert builder.calls == ["duchenne"]
    assert len(searcher.calls) == 1

    refreshed = session.get(SearchTerm, search_term.id)
    assert refreshed is not None
    artefact = refreshed.artefacts[0]
    assert artefact.result_signature == updated_signature
    assert artefact.last_refreshed_at > stale_timestamp