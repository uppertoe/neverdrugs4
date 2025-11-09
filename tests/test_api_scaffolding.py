from __future__ import annotations

from datetime import datetime, timedelta, timezone
import pytest
from types import SimpleNamespace
from unittest.mock import patch
from urllib.parse import quote

from sqlalchemy import select

from app.models import (
    ArticleArtefact,
    ArticleSnippet,
    ClaimSetRefresh,
    ProcessedClaim,
    ProcessedClaimDrugLink,
    ProcessedClaimEvidence,
    ProcessedClaimSet,
    ProcessedClaimSetVersion,
    SearchArtefact,
    SearchTerm,
)
from app.services.mesh_resolution import MeshResolutionPreview
from app.services.nih_pipeline import MeshTermsNotFoundError
from app.services.search import SearchResolution, compute_mesh_signature


@pytest.fixture(autouse=True)
def stub_preview_mesh_resolution():
    preview = MeshResolutionPreview(
        status="resolved",
        raw_query="duchenne",
        normalized_query="duchenne",
        mesh_terms=["Duchenne Muscular Dystrophy"],
        ranked_options=["Duchenne Muscular Dystrophy"],
        suggestions=[],
        espell_correction=None,
    )
    with patch("app.api.routes.preview_mesh_resolution", return_value=preview) as mock:
        yield mock


def _seed_processed_claim_set(
    session,
    *,
    mesh_signature: str = "king-denborough|anesthesia",
    condition_label: str = "King Denborough syndrome",
) -> ProcessedClaimSet:
    claim_set = ProcessedClaimSet(
        mesh_signature=mesh_signature,
        condition_label=condition_label,
    )
    session.add(claim_set)
    session.flush()
    session.refresh(claim_set)

    version = ProcessedClaimSetVersion(
        claim_set=claim_set,
        version_number=1,
        status="active",
        pipeline_metadata={},
    )
    session.add(version)
    session.flush()

    claim = ProcessedClaim(
        claim_set_id=claim_set.id,
        claim_set_version=version,
        claim_id="risk:succinylcholine",
        classification="risk",
        summary="Succinylcholine precipitates malignant hyperthermia.",
        confidence="high",
        canonical_hash="hash-risk:succinylcholine",
        claim_group_id="group-risk:succinylcholine",
        drugs=["succinylcholine"],
        drug_classes=["depolarising neuromuscular blocker"],
        source_claim_ids=["risk:succinylcholine"],
        severe_reaction_flag=True,
        severe_reaction_terms=["cardiac arrest"],
        up_votes=0,
        down_votes=0,
    )
    session.add(claim)
    session.flush()

    evidence = ProcessedClaimEvidence(
        claim_id=claim.id,
        snippet_id="42",
        pmid="11111111",
        article_title="Safety considerations for neuromuscular blockade",
        citation_url="https://pubmed.ncbi.nlm.nih.gov/11111111/",
        key_points=["Case reports link succinylcholine to malignant hyperthermia"],
        notes="",
    )
    session.add(evidence)

    session.add_all(
        [
            ProcessedClaimDrugLink(claim_id=claim.id, term="succinylcholine", term_kind="drug"),
            ProcessedClaimDrugLink(
                claim_id=claim.id,
                term="depolarising neuromuscular blocker",
                term_kind="drug_class",
            ),
        ]
    )

    session.flush()
    return claim_set


def test_get_processed_claim_returns_serialised_payload(client, session):
    claim_set = _seed_processed_claim_set(session)

    response = client.get(f"/api/claims/{claim_set.id}")
    assert response.status_code == 200
    payload = response.get_json()

    assert payload["id"] == claim_set.id
    assert payload["slug"] == claim_set.slug
    assert payload["condition_label"] == "King Denborough syndrome"
    assert payload["mesh_signature"] == "king-denborough|anesthesia"
    assert payload["claims"]

    claim_payload = payload["claims"][0]
    assert claim_payload["claim_id"] == "risk:succinylcholine"
    assert claim_payload["classification"] == "risk"
    assert claim_payload["confidence"] == "high"
    assert claim_payload["drugs"] == ["succinylcholine"]
    assert claim_payload["drug_classes"] == ["depolarising neuromuscular blocker"]
    assert claim_payload["source_claim_ids"] == ["risk:succinylcholine"]
    assert claim_payload["severe_reaction"] == {"flag": True, "terms": ["cardiac arrest"]}

    evidence_payload = claim_payload["supporting_evidence"][0]
    assert evidence_payload["snippet_id"] == "42"
    assert evidence_payload["pmid"] == "11111111"
    assert evidence_payload["article_title"].startswith("Safety considerations")
    assert evidence_payload["citation_url"].startswith("https://pubmed")
    assert evidence_payload["key_points"]

    drug_links = claim_payload["drug_links"]
    assert {link["term_kind"] for link in drug_links} == {"drug", "drug_class"}


def test_get_processed_claim_returns_404_when_missing(client):
    response = client.get("/api/claims/999")
    assert response.status_code == 404
    payload = response.get_json()
    assert payload == {"detail": "Processed claim set not found"}


def test_get_processed_claim_allows_slug_lookup(client, session):
    claim_set = _seed_processed_claim_set(session)

    response = client.get(f"/api/claims/{claim_set.slug}")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["id"] == claim_set.id
    assert payload["slug"] == claim_set.slug


@patch("app.api.routes.resolve_condition_via_nih")
@patch("app.api.routes.enqueue_claim_pipeline")
def test_resolve_claims_returns_cached_set_when_available(
    enqueue_mock,
    resolve_mock,
    client,
    session,
):
    resolution = SearchResolution(
        normalized_condition="duchenne muscular dystrophy",
        mesh_terms=["Duchenne Muscular Dystrophy"],
        reused_cached=True,
        search_term_id=999,
    )
    mesh_signature = compute_mesh_signature(resolution.mesh_terms)
    claim_set = _seed_processed_claim_set(
        session,
        mesh_signature=mesh_signature,
        condition_label="Duchenne muscular dystrophy",
    )
    resolve_mock.return_value = resolution
    enqueue_mock.return_value = {
        "job_id": "ignored",
        "status": "queued",
    }

    response = client.post(
        "/api/claims/resolve",
        json={"condition": "Duchenne"},
    )

    assert response.status_code == 200
    payload = response.get_json()

    assert payload["claim_set"]
    assert payload["claim_set"]["slug"].startswith("duchenne")
    assert payload["job"] is None
    assert payload["resolution"] == {
        "normalized_condition": resolution.normalized_condition,
        "mesh_terms": resolution.mesh_terms,
        "reused_cached": True,
        "search_term_id": resolution.search_term_id,
    }
    enqueue_mock.assert_not_called()
    resolve_mock.assert_called_once()


@patch("app.api.routes.resolve_condition_via_nih")
@patch("app.api.routes.enqueue_claim_pipeline")
def test_resolve_claims_schedules_refresh_job(
    enqueue_mock,
    resolve_mock,
    client,
    session,
):
    resolution = SearchResolution(
        normalized_condition="duchenne muscular dystrophy",
        mesh_terms=["Duchenne Muscular Dystrophy"],
        reused_cached=False,
        search_term_id=7,
    )
    resolve_mock.return_value = resolution
    mesh_signature = compute_mesh_signature(resolution.mesh_terms)
    enqueue_mock.return_value = {
        "job_id": "task-123",
        "status": "queued",
    }

    response = client.post(
        "/api/claims/resolve",
        json={"condition": "Duchenne"},
    )

    assert response.status_code == 200
    payload = response.get_json()

    assert payload["job"] == {"job_id": "task-123", "status": "queued"}
    encoded_signature = quote(mesh_signature, safe="")
    assert payload["refresh_url"] == f"/api/claims/refresh/{encoded_signature}"
    assert payload["claim_set"] is None
    assert payload["resolution"] == {
        "normalized_condition": resolution.normalized_condition,
        "mesh_terms": resolution.mesh_terms,
        "reused_cached": False,
        "search_term_id": resolution.search_term_id,
    }
    enqueue_mock.assert_called_once()
    args, kwargs = enqueue_mock.call_args
    assert kwargs["resolution"] == resolution
    assert kwargs["condition_label"] == "Duchenne"
    assert kwargs["session"] is not None
    assert kwargs["mesh_signature"] == compute_mesh_signature(resolution.mesh_terms)
    resolve_mock.assert_called_once()


@patch("app.api.routes.enqueue_claim_refresh")
def test_enqueue_claim_pipeline_enqueues_job(enqueue_mock, session):
    from app.api.routes import enqueue_claim_pipeline

    enqueue_mock.return_value = {"job_id": "celery-123", "status": "queued"}

    resolution = SearchResolution(
        normalized_condition="duchenne muscular dystrophy",
        mesh_terms=["Duchenne Muscular Dystrophy"],
        reused_cached=False,
        search_term_id=21,
    )

    mesh_signature = compute_mesh_signature(resolution.mesh_terms)

    job = enqueue_claim_pipeline(
        session=session,
        resolution=resolution,
        condition_label="Duchenne",
        mesh_signature=mesh_signature,
    )

    enqueue_mock.assert_called_once_with(
        session=session,
        resolution=resolution,
        condition_label="Duchenne",
        mesh_signature=mesh_signature,
    )
    assert job == {"job_id": "celery-123", "status": "queued"}


@patch("app.api.routes.resolve_condition_via_nih")
@patch("app.api.routes.enqueue_claim_pipeline")
def test_resolve_claims_propogates_job_errors(
    enqueue_mock,
    resolve_mock,
    client,
    session,
):
    resolution = SearchResolution(
        normalized_condition="duchenne muscular dystrophy",
        mesh_terms=["Duchenne Muscular Dystrophy"],
        reused_cached=False,
        search_term_id=7,
    )
    resolve_mock.return_value = resolution
    enqueue_mock.side_effect = RuntimeError("celery not available")

    response = client.post(
        "/api/claims/resolve",
        json={"condition": "Duchenne"},
    )

    assert response.status_code == 503
    payload = response.get_json()
    assert payload == {
        "detail": "Failed to queue background refresh",
    }
    enqueue_mock.assert_called_once()


@patch("app.api.routes.resolve_condition_via_nih")
def test_resolve_claims_returns_suggestions_when_mesh_missing(
    resolve_mock,
    client,
):
    resolve_mock.side_effect = MeshTermsNotFoundError(
        normalized_condition="unknown condition",
        search_term_id=77,
        suggestions=["Alpha Condition", "Beta Disorder"],
    )

    response = client.post(
        "/api/claims/resolve",
        json={"condition": "Some rare thing"},
    )

    assert response.status_code == 422
    payload = response.get_json()
    assert payload["detail"].startswith("No MeSH terms matched")
    assert payload["resolution"] == {
        "normalized_condition": "unknown condition",
        "mesh_terms": [],
        "reused_cached": False,
        "search_term_id": 77,
    }
    assert payload["suggested_mesh_terms"] == ["Alpha Condition", "Beta Disorder"]
    resolve_mock.assert_called_once()


@patch("app.api.routes.resolve_condition_via_nih")
def test_resolve_claims_requests_clarification_when_preview_ambiguous(
    resolve_mock,
    client,
    stub_preview_mesh_resolution,
):
    stub_preview_mesh_resolution.return_value = MeshResolutionPreview(
        status="needs_clarification",
        raw_query="Porphyria",
        normalized_query="porphyria",
        mesh_terms=["Acute Intermittent Porphyria", "Porphyria, Variegate"],
        ranked_options=["Acute Intermittent Porphyria", "Porphyria, Variegate"],
        suggestions=[],
        espell_correction=None,
    )

    response = client.post(
        "/api/claims/resolve",
        json={"condition": "Porphyria"},
    )

    assert response.status_code == 409
    payload = response.get_json()
    assert payload["detail"].startswith("Multiple MeSH terms")
    assert payload["resolution_preview"]["status"] == "needs_clarification"
    assert payload["resolution_preview"]["mesh_terms"] == [
        "Acute Intermittent Porphyria",
        "Porphyria, Variegate",
    ]
    resolve_mock.assert_not_called()


@patch("app.api.routes.resolve_condition_via_nih")
def test_resolve_claims_returns_preview_when_not_found(
    resolve_mock,
    client,
    stub_preview_mesh_resolution,
):
    stub_preview_mesh_resolution.return_value = MeshResolutionPreview(
        status="not_found",
        raw_query="Porphyria",
        normalized_query="porphyria",
        mesh_terms=[],
        ranked_options=["Porphyria Variegata"],
        suggestions=["Porphyria Variegata", "Porphyria Cutanea Tarda"],
        espell_correction=None,
    )

    response = client.post(
        "/api/claims/resolve",
        json={"condition": "Porphyria"},
    )

    assert response.status_code == 422
    payload = response.get_json()
    assert payload["detail"].startswith("No MeSH terms matched")
    assert payload["resolution_preview"]["status"] == "not_found"
    assert payload["resolution_preview"]["suggestions"] == [
        "Porphyria Variegata",
        "Porphyria Cutanea Tarda",
    ]
    assert payload["resolution_preview"]["ranked_options"] == ["Porphyria Variegata"]
    resolve_mock.assert_not_called()


@patch("app.api.routes.resolve_condition_via_nih")
@patch("app.api.routes.enqueue_claim_pipeline")
def test_resolve_claims_accepts_manual_mesh_terms(
    enqueue_mock,
    resolve_mock,
    client,
    stub_preview_mesh_resolution,
):
    stub_preview_mesh_resolution.reset_mock()
    resolution = SearchResolution(
        normalized_condition="porphyria",
        mesh_terms=["Porphyria Variegata"],
        reused_cached=False,
        search_term_id=55,
    )
    resolve_mock.return_value = resolution
    enqueue_mock.return_value = {"job_id": "task-42", "status": "queued"}

    response = client.post(
        "/api/claims/resolve",
        json={
            "condition": "Porphyria",
            "mesh_terms": ["Porphyria Variegata"],
        },
    )

    assert response.status_code == 200
    assert stub_preview_mesh_resolution.call_count == 0
    builder = resolve_mock.call_args.kwargs["mesh_builder"]
    assert builder is not None
    manual_result = builder("porphyria")
    assert manual_result.mesh_terms == ["Porphyria Variegata"]
    assert manual_result.query_payload["manual_selection"] is True
    assert manual_result.query_payload["selected_mesh_terms"] == ["Porphyria Variegata"]
    assert manual_result.query_payload["esearch"]["query"]
    enqueue_mock.assert_called_once()


def test_resolve_claims_validates_mesh_terms_input(client, stub_preview_mesh_resolution):
    stub_preview_mesh_resolution.reset_mock()

    response = client.post(
        "/api/claims/resolve",
        json={
            "condition": "Porphyria",
            "mesh_terms": "Porphyria Variegata",
        },
    )

    assert response.status_code == 400
    payload = response.get_json()
    assert payload["detail"].startswith("mesh_terms")
    assert stub_preview_mesh_resolution.call_count == 0


@patch("app.api.routes.resolve_condition_via_nih")
@patch("app.api.routes.enqueue_claim_pipeline")
def test_resolve_claims_reuses_existing_refresh_job(
    enqueue_mock,
    resolve_mock,
    client,
    session,
):
    resolution = SearchResolution(
        normalized_condition="duchenne muscular dystrophy",
        mesh_terms=["Duchenne Muscular Dystrophy"],
        reused_cached=False,
        search_term_id=7,
    )
    resolve_mock.return_value = resolution
    mesh_signature = compute_mesh_signature(resolution.mesh_terms)
    session.add(
        ClaimSetRefresh(
            mesh_signature=mesh_signature,
            job_id="task-321",
            status="running",
        )
    )
    session.flush()

    response = client.post(
        "/api/claims/resolve",
        json={"condition": "Duchenne"},
    )

    assert response.status_code == 200
    payload = response.get_json()
    encoded_signature = quote(mesh_signature, safe="")
    assert payload["job"] == {"job_id": "task-321", "status": "running"}
    assert payload["refresh_url"] == f"/api/claims/refresh/{encoded_signature}"
    enqueue_mock.assert_not_called()


@patch("app.api.routes.resolve_condition_via_nih")
@patch("app.api.routes.enqueue_claim_pipeline")
def test_resolve_claims_ignores_force_refresh_parameter(
    enqueue_mock,
    resolve_mock,
    client,
    session,
):
    resolution = SearchResolution(
        normalized_condition="duchenne muscular dystrophy",
        mesh_terms=["Duchenne Muscular Dystrophy"],
        reused_cached=True,
        search_term_id=7,
    )
    resolve_mock.return_value = resolution
    mesh_signature = compute_mesh_signature(resolution.mesh_terms)
    refresh = ClaimSetRefresh(
        mesh_signature=mesh_signature,
        job_id="task-original",
        status="running",
        progress_state="collecting_articles",
        progress_payload={"stage": "collecting"},
    )
    session.add(refresh)
    session.flush()

    response = client.post(
        "/api/claims/resolve",
        json={"condition": "Duchenne", "force_refresh": True},
    )

    assert response.status_code == 200
    payload = response.get_json()
    encoded_signature = quote(mesh_signature, safe="")
    assert payload["job"] == {"job_id": "task-original", "status": "running"}
    assert payload["refresh_url"] == f"/api/claims/refresh/{encoded_signature}"
    enqueue_mock.assert_not_called()


@patch("app.api.routes.resolve_condition_via_nih")
@patch("app.api.routes.enqueue_claim_pipeline")
def test_resolve_claims_returns_cached_when_refresh_missing(
    enqueue_mock,
    resolve_mock,
    client,
    session,
):
    resolution = SearchResolution(
        normalized_condition="duchenne muscular dystrophy",
        mesh_terms=["Duchenne Muscular Dystrophy"],
        reused_cached=True,
        search_term_id=42,
    )
    resolve_mock.return_value = resolution
    mesh_signature = compute_mesh_signature(resolution.mesh_terms)
    claim_set = _seed_processed_claim_set(
        session,
        mesh_signature=mesh_signature,
        condition_label="Duchenne",
    )

    response = client.post(
        "/api/claims/resolve",
        json={"condition": "Duchenne"},
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["claim_set"]["id"] == claim_set.id
    assert payload["job"] is None
    assert "refresh_url" not in payload
    enqueue_mock.assert_not_called()

    refresh_record = session.execute(
        select(ClaimSetRefresh).where(ClaimSetRefresh.mesh_signature == mesh_signature)
    ).scalar_one_or_none()
    assert refresh_record is None


@patch("app.api.routes.resolve_condition_via_nih")
@patch("app.api.routes.enqueue_claim_pipeline")
def test_resolve_claims_requeues_stale_job(
    enqueue_mock,
    resolve_mock,
    client,
    session,
):
    resolution = SearchResolution(
        normalized_condition="duchenne muscular dystrophy",
        mesh_terms=["Duchenne Muscular Dystrophy"],
        reused_cached=True,
        search_term_id=7,
    )
    resolve_mock.return_value = resolution
    mesh_signature = compute_mesh_signature(resolution.mesh_terms)
    stale_time = datetime.now(timezone.utc) - timedelta(seconds=601)
    refresh = ClaimSetRefresh(
        mesh_signature=mesh_signature,
        job_id="task-stale",
        status="running",
        progress_state="collecting_articles",
        progress_payload={"stage": "collecting"},
        updated_at=stale_time,
    )
    session.add(refresh)
    session.flush()

    enqueue_mock.return_value = {
        "job_id": "task-refreshed",
        "status": "queued",
    }

    response = client.post(
        "/api/claims/resolve",
        json={"condition": "Duchenne"},
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["job"] == {"job_id": "task-refreshed", "status": "queued"}
    encoded_signature = quote(mesh_signature, safe="")
    assert payload["refresh_url"] == f"/api/claims/refresh/{encoded_signature}"
    enqueue_mock.assert_called_once()

    session.refresh(refresh)
    assert refresh.job_id == "task-refreshed"
    assert refresh.status == "queued"
    assert refresh.progress_state == "queued"


def _seed_search_term_with_artefact(session) -> tuple[SearchTerm, SearchArtefact]:
    term = SearchTerm(canonical="duchenne muscular dystrophy")
    session.add(term)
    session.flush()
    session.refresh(term)

    artefact = SearchArtefact(
        search_term_id=term.id,
        query_payload={
            "normalized_query": "duchenne muscular dystrophy",
            "esearch": {
                "ids": ["68020388"],
                "translation": None,
                "primary_id": "68020388",
                "query": "\"Duchenne Muscular Dystrophy\"[MeSH Terms]",
            },
            "esummary": {
                "primary_id": "68020388",
                "mesh_terms": ["Duchenne Muscular Dystrophy"],
            },
        },
        mesh_terms=["Duchenne Muscular Dystrophy"],
        mesh_signature="duchenne-muscular-dystrophy",
        result_signature="hash123",
        ttl_policy_seconds=86_400,
        last_refreshed_at=datetime.now(timezone.utc),
    )
    session.add(artefact)
    session.flush()

    return term, artefact


def _seed_articles_with_snippets(session, term: SearchTerm) -> tuple[ArticleArtefact, ArticleArtefact]:
    first_article = ArticleArtefact(
        search_term_id=term.id,
        pmid="11111111",
        rank=0,
        score=2.5,
        citation={
            "pmid": "11111111",
            "title": "Breakthrough in Duchenne",
            "preferred_url": "https://pubmed.ncbi.nlm.nih.gov/11111111/",
        },
        content_source="abstract",
        retrieved_at=datetime.now(timezone.utc),
    )
    second_article = ArticleArtefact(
        search_term_id=term.id,
        pmid="22222222",
        rank=1,
        score=1.7,
        citation={
            "pmid": "22222222",
            "title": "Review of Duchenne therapies",
            "preferred_url": "https://pubmed.ncbi.nlm.nih.gov/22222222/",
        },
        content_source="pmc",
        retrieved_at=datetime.now(timezone.utc),
    )
    session.add_all([first_article, second_article])
    session.flush()

    snippet = ArticleSnippet(
        article_artefact_id=first_article.id,
        snippet_hash="hash-snippet",
        drug="eteplirsen",
        classification="support",
        snippet_text="Eteplirsen shows improved dystrophin levels.",
        snippet_score=0.91,
        cues=["increased dystrophin"],
    )
    session.add(snippet)
    session.flush()

    return first_article, second_article


def test_get_search_query_returns_payload(client, session):
    term, artefact = _seed_search_term_with_artefact(session)

    response = client.get(f"/api/search/{term.id}/query")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["search_term_id"] == term.id
    assert payload["search_term_slug"] == term.slug
    assert payload["mesh_terms"] == artefact.mesh_terms
    assert payload["query"].startswith("\"Duchenne Muscular Dystrophy\"")
    assert payload["query_payload"]["esearch"]["primary_id"] == "68020388"


def test_get_search_query_accepts_slug_lookup(client, session):
    term, artefact = _seed_search_term_with_artefact(session)

    response = client.get(f"/api/search/{term.slug}/query")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["search_term_id"] == term.id
    assert payload["search_term_slug"] == term.slug
    assert payload["mesh_terms"] == artefact.mesh_terms


def test_get_search_query_returns_404_for_missing_term(client):
    response = client.get("/api/search/999/query")

    assert response.status_code == 404
    assert response.get_json() == {"detail": "Search term not found"}


def test_get_search_query_returns_404_when_no_artefact(client, session):
    term = SearchTerm(canonical="central core disease")
    session.add(term)
    session.flush()
    session.refresh(term)

    response = client.get(f"/api/search/{term.id}/query")

    assert response.status_code == 404
    assert response.get_json()["detail"] == "Search query not available"


def test_get_search_articles_returns_ordered_list(client, session):
    term, _ = _seed_search_term_with_artefact(session)
    _seed_articles_with_snippets(session, term)

    response = client.get(f"/api/search/{term.id}/articles")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["search_term_id"] == term.id
    assert payload["search_term_slug"] == term.slug
    articles = payload["articles"]
    assert [article["pmid"] for article in articles] == ["11111111", "22222222"]
    assert articles[0]["preferred_url"].endswith("11111111/")


def test_get_search_articles_accepts_slug_lookup(client, session):
    term, _ = _seed_search_term_with_artefact(session)
    _seed_articles_with_snippets(session, term)

    response = client.get(f"/api/search/{term.slug}/articles")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["search_term_id"] == term.id
    assert payload["search_term_slug"] == term.slug


def test_get_search_articles_returns_empty_collection(client, session):
    term = SearchTerm(canonical="rare disorder")
    session.add(term)
    session.flush()
    session.refresh(term)
    session.add(
        SearchArtefact(
            search_term_id=term.id,
            query_payload={},
            mesh_terms=[],
            mesh_signature="",
            result_signature=None,
            ttl_policy_seconds=86_400,
        )
    )
    session.flush()

    response = client.get(f"/api/search/{term.id}/articles")

    assert response.status_code == 200
    assert response.get_json()["articles"] == []


def test_get_search_snippets_returns_payload(client, session):
    term, _ = _seed_search_term_with_artefact(session)
    first_article, second_article = _seed_articles_with_snippets(session, term)
    # add snippet on second article to ensure ordering by article rank
    session.add(
        ArticleSnippet(
            article_artefact_id=second_article.id,
            snippet_hash="hash-second",
            drug="ataluren",
            classification="neutral",
            snippet_text="Ataluren shows mixed outcomes.",
            snippet_score=0.6,
            cues=["variable response"],
        )
    )
    session.flush()

    response = client.get(f"/api/search/{term.id}/snippets")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["search_term_id"] == term.id
    assert payload["search_term_slug"] == term.slug
    snippets = payload["snippets"]
    assert [snippet["pmid"] for snippet in snippets] == ["11111111", "22222222"]
    assert snippets[0]["drug"] == "eteplirsen"
    assert snippets[1]["drug"] == "ataluren"


def test_get_search_snippets_accepts_slug_lookup(client, session):
    term, _ = _seed_search_term_with_artefact(session)
    _seed_articles_with_snippets(session, term)

    response = client.get(f"/api/search/{term.slug}/snippets")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["search_term_id"] == term.id
    assert payload["search_term_slug"] == term.slug

def test_get_search_snippets_returns_404_for_missing_term(client):
    response = client.get("/api/search/123/snippets")

    assert response.status_code == 404
    assert response.get_json() == {"detail": "Search term not found"}


@patch("app.api.routes.resolve_condition_via_nih")
@patch("app.api.routes.enqueue_claim_pipeline")
def test_resolve_claims_logs_enqueue_failures(
    enqueue_mock,
    resolve_mock,
    client,
    caplog,
):
    caplog.set_level("ERROR")
    resolution = SearchResolution(
        normalized_condition="duchenne muscular dystrophy",
        mesh_terms=["Duchenne Muscular Dystrophy"],
        reused_cached=False,
        search_term_id=7,
    )
    resolve_mock.return_value = resolution
    enqueue_mock.side_effect = RuntimeError("celery not available")

    response = client.post(
        "/api/claims/resolve",
        json={"condition": "Duchenne"},
    )

    assert response.status_code == 503
    assert "Failed to queue background refresh" in response.get_json()["detail"]
    assert "celery not available" in caplog.text


def test_get_refresh_status_returns_job_metadata(client, session):
    mesh_signature = "duchenne muscular dystrophy"
    refresh = ClaimSetRefresh(
        mesh_signature=mesh_signature,
        job_id="task-123",
        status="running",
        error_message="processing",
        progress_state="collecting",
        progress_payload={"steps_completed": ["resolve_condition"]},
    )
    session.add(refresh)
    session.flush()

    response = client.get(f"/api/claims/refresh/{quote(mesh_signature, safe='')}")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["mesh_signature"] == mesh_signature
    assert payload["job_id"] == "task-123"
    assert payload["status"] == "running"
    assert payload["error_message"] == "processing"
    assert payload["claim_set_id"] is None
    assert payload.get("claim_set_slug") is None
    assert payload["created_at"]
    assert payload["updated_at"]
    assert payload["progress"] == {
        "stage": "collecting",
        "details": {"steps_completed": ["resolve_condition"]},
    }


def test_get_refresh_status_includes_claim_set_id_when_ready(client, session):
    mesh_signature = "duchenne muscular dystrophy"
    claim_set = _seed_processed_claim_set(
        session,
        mesh_signature=mesh_signature,
        condition_label="Duchenne",
    )
    refresh = ClaimSetRefresh(
        mesh_signature=mesh_signature,
        job_id="task-456",
        status="completed",
        progress_state="completed",
        progress_payload={"steps_completed": ["resolve_condition", "persist_claims"]},
    )
    session.add(refresh)
    session.flush()

    response = client.get(f"/api/claims/refresh/{quote(mesh_signature, safe='')}")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["claim_set_id"] == claim_set.id
    assert payload["claim_set_slug"] == claim_set.slug
    assert payload["status"] == "completed"
    assert payload["job_id"] == "task-456"
    assert payload["progress"]["stage"] == "completed"
    assert payload["resolution"] == {
        "normalized_condition": "duchenne muscular dystrophy",
        "mesh_terms": ["Duchenne Muscular Dystrophy"],
    }


def test_get_refresh_status_returns_404_for_unknown_signature(client):
    response = client.get("/api/claims/refresh/unknown-signature")
    assert response.status_code == 404
    assert response.get_json() == {"detail": "Refresh job not found"}


def test_get_refresh_status_accepts_claim_set_id_lookup(client, session):
    mesh_signature = "duchenne muscular dystrophy"
    claim_set = _seed_processed_claim_set(
        session,
        mesh_signature=mesh_signature,
        condition_label="Duchenne",
    )
    refresh = ClaimSetRefresh(
        mesh_signature=mesh_signature,
        job_id="task-789",
        status="running",
        progress_state="running",
        progress_payload={},
    )
    session.add(refresh)
    session.flush()

    response = client.get(f"/api/claims/refresh/{claim_set.id}")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["mesh_signature"] == mesh_signature
    assert payload["claim_set_id"] == claim_set.id
    assert payload["claim_set_slug"] == claim_set.slug
    assert payload["job_id"] == "task-789"


def test_get_refresh_status_accepts_claim_set_slug_lookup(client, session):
    mesh_signature = "duchenne muscular dystrophy"
    claim_set = _seed_processed_claim_set(
        session,
        mesh_signature=mesh_signature,
        condition_label="Duchenne",
    )
    refresh = ClaimSetRefresh(
        mesh_signature=mesh_signature,
        job_id="task-987",
        status="running",
        progress_state="running",
        progress_payload={},
    )
    session.add(refresh)
    session.flush()

    response = client.get(f"/api/claims/refresh/{claim_set.slug}")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["mesh_signature"] == mesh_signature
    assert payload["claim_set_id"] == claim_set.id
    assert payload["claim_set_slug"] == claim_set.slug
    assert payload["job_id"] == "task-987"


def test_get_refresh_status_exposes_failure_details(client, session):
    mesh_signature = "duchenne muscular dystrophy"
    refresh = ClaimSetRefresh(
        mesh_signature=mesh_signature,
        job_id="task-999",
        status="failed",
        error_message="OpenAI timeout",
        progress_state="failed",
        progress_payload={"error": "OpenAI timeout"},
    )
    session.add(refresh)
    session.flush()

    response = client.get(f"/api/claims/refresh/{quote(mesh_signature, safe='')}")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "failed"
    assert payload["error_message"] == "OpenAI timeout"
    assert payload["can_retry"] is True


def test_get_refresh_status_marks_stale_job_retryable(client, session):
    mesh_signature = "duchenne muscular dystrophy"
    stale_time = datetime.now(timezone.utc) - timedelta(seconds=601)
    refresh = ClaimSetRefresh(
        mesh_signature=mesh_signature,
        job_id="task-888",
        status="running",
        progress_state="collecting",
        progress_payload={},
        updated_at=stale_time,
    )
    session.add(refresh)
    session.flush()

    response = client.get(f"/api/claims/refresh/{quote(mesh_signature, safe='')}")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["can_retry"] is True
