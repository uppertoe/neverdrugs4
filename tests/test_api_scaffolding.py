from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from app.models import (
    ClaimSetRefresh,
    ProcessedClaim,
    ProcessedClaimDrugLink,
    ProcessedClaimEvidence,
    ProcessedClaimSet,
)
from app.services.search import SearchResolution, compute_mesh_signature


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

    claim = ProcessedClaim(
        claim_set_id=claim_set.id,
        claim_id="risk:succinylcholine",
        classification="risk",
        summary="Succinylcholine precipitates malignant hyperthermia.",
        confidence="high",
        drugs=["succinylcholine"],
        drug_classes=["depolarising neuromuscular blocker"],
        source_claim_ids=["risk:succinylcholine"],
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
    enqueue_mock.return_value = {
        "job_id": "task-123",
        "status": "queued",
    }

    response = client.post(
        "/api/claims/resolve",
        json={"condition": "Duchenne"},
    )

    assert response.status_code == 202
    payload = response.get_json()

    assert payload["job"] == {"job_id": "task-123", "status": "queued"}
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


@patch("app.api.routes.refresh_claims_for_condition.delay")
def test_enqueue_claim_pipeline_enqueues_celery_task(delay_mock, session):
    from app.api.routes import enqueue_claim_pipeline

    delay_mock.return_value = SimpleNamespace(id="celery-123")

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

    delay_mock.assert_called_once_with(
        resolution_id=resolution.search_term_id,
        condition_label="Duchenne",
        normalized_condition=resolution.normalized_condition,
        mesh_terms=list(resolution.mesh_terms),
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

    assert response.status_code == 202
    payload = response.get_json()
    assert payload["job"] == {"job_id": "task-321", "status": "running"}
    enqueue_mock.assert_not_called()


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
