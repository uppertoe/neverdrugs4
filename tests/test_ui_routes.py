from __future__ import annotations

from datetime import datetime, timezone
from urllib.parse import quote
from types import SimpleNamespace
from typing import Sequence

from app.models import (
    ProcessedClaim,
    ProcessedClaimDrugLink,
    ProcessedClaimEvidence,
    ProcessedClaimSet,
    ProcessedClaimSetVersion,
    SearchArtefact,
    SearchTerm,
)
from app.services.mesh_resolution import MeshResolutionPreview
from app.services.search import SearchResolution, compute_mesh_signature


def _seed_claim_set(
    session,
    *,
    mesh_terms: Sequence[str] = ("King Denborough syndrome", "Anesthesia"),
) -> ProcessedClaimSet:
    mesh_signature = compute_mesh_signature(list(mesh_terms))
    claim_set = ProcessedClaimSet(mesh_signature=mesh_signature, condition_label="King Denborough syndrome")
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
        claim_set=claim_set,
        claim_set_version=version,
        claim_id="claim:succinylcholine-risk",
        classification="risk",
        summary="Succinylcholine precipitates malignant hyperthermia.",
        confidence="high",
        canonical_hash="hash-risk",
        claim_group_id="group-risk",
        drugs=["Succinylcholine"],
        drug_classes=["depolarising neuromuscular blocker"],
        source_claim_ids=["claim:succinylcholine-risk"],
        severe_reaction_flag=True,
        severe_reaction_terms=["Cardiac arrest"],
    )
    session.add(claim)
    session.flush()

    evidence = ProcessedClaimEvidence(
        claim=claim,
        snippet_id="42",
        pmid="11111111",
        article_title="Safety considerations",
        citation_url="https://pubmed.ncbi.nlm.nih.gov/11111111/",
        key_points=["Case reports link succinylcholine to malignant hyperthermia"],
        notes=None,
    )
    session.add(evidence)

    session.add_all(
        [
            ProcessedClaimDrugLink(claim=claim, term="Succinylcholine", term_kind="drug"),
            ProcessedClaimDrugLink(
                claim=claim,
                term="depolarising neuromuscular blocker",
                term_kind="drug_class",
            ),
        ]
    )
    session.flush()
    return claim_set


def test_root_redirects_to_ui(client):
    response = client.get("/")
    assert response.status_code == 302
    assert response.headers["Location"].endswith("/ui/")


def test_ui_home_renders_search_form(client):
    response = client.get("/ui/")
    assert response.status_code == 200
    body = response.data.decode()
    assert '<form' in body
    assert 'hx-post="/ui/search-preview"' in body
    assert 'name="condition"' in body
    assert 'hx-indicator="#search-indicator"' in body
    assert "Request submitted. Awaiting a response..." in body
    assert "Preview search" in body
    assert 'hx-boost="true"' in body and 'hx-target="#page-main"' in body
    assert 'id="page-main"' in body


def test_ui_home_boosted_request_returns_full_layout(client):
    response = client.get(
        "/ui/",
        headers={
            "HX-Request": "true",
            "HX-Boosted": "true",
        },
    )

    assert response.status_code == 200
    body = response.data.decode()
    assert "<!doctype html>" in body.lower()
    assert "<main" in body.lower()


def test_ui_search_returns_cached_claims_html(client, session, monkeypatch):
    mesh_terms = ("King Denborough syndrome", "Anesthesia")
    claim_set = _seed_claim_set(session, mesh_terms=mesh_terms)

    resolution = SearchResolution(
        normalized_condition="king denborough syndrome",
        mesh_terms=list(mesh_terms),
        reused_cached=True,
        search_term_id=123,
        mesh_signature=claim_set.mesh_signature,
    )

    monkeypatch.setattr("app.ui.routes._load_cached_resolution", lambda _session, _condition: resolution)

    response = client.post(
        "/ui/search",
        data={"condition": "King Denborough"},
        headers={"HX-Request": "true"},
    )

    assert response.status_code == 200
    body = response.data.decode()
    assert "Succinylcholine precipitates malignant hyperthermia." in body
    assert "King Denborough syndrome" in body
    assert "Filter by drug" in body
    assert "data-claim-entry" in body
    assert "Resolved MeSH terms" in body
    assert "Anesthesia" in body

    runs_response = client.get("/ui/runs")
    runs_body = runs_response.data.decode()
    assert "Your recent searches" in runs_body
    assert "King Denborough" in runs_body
    assert 'href="/ui/claims/' in runs_body


def test_ui_search_preview_with_cached_claims_returns_results(client, session, monkeypatch):
    mesh_terms = ("King Denborough syndrome", "Anesthesia")
    claim_set = _seed_claim_set(session, mesh_terms=mesh_terms)

    resolution = SearchResolution(
        normalized_condition="king denborough syndrome",
        mesh_terms=list(mesh_terms),
        reused_cached=True,
        search_term_id=claim_set.last_search_term_id or 1,
        mesh_signature=claim_set.mesh_signature,
    )

    monkeypatch.setattr("app.ui.routes._load_cached_resolution", lambda _session, _condition: resolution)
    preview = MeshResolutionPreview(
        status="resolved",
        raw_query="King Denborough",
        normalized_query="king denborough",
        mesh_terms=list(mesh_terms),
        ranked_options=list(mesh_terms),
        suggestions=[],
    )
    monkeypatch.setattr("app.ui.routes.preview_mesh_resolution", lambda _condition: preview)

    response = client.post(
        "/ui/search-preview",
        data={"condition": "King Denborough"},
        headers={"HX-Request": "true"},
    )

    assert response.status_code == 200
    body = response.data.decode()
    assert "Confirm search" in body
    assert "Run refresh" not in body
    assert "View completed search" in body


def test_status_safety_claim_ignores_severe_flag(client, session):
    claim_set = _seed_claim_set(session)
    active_version = claim_set.get_active_version()
    assert active_version is not None
    safety_claim = active_version.claims[0]
    safety_claim.classification = "safety"
    safety_claim.severe_reaction_flag = True
    safety_claim.severe_reaction_terms = ["Hyperkalaemia"]
    session.flush()

    response = client.get(f"/ui/status/{quote(claim_set.mesh_signature, safe='')}")

    assert response.status_code == 200
    body = response.data.decode()
    assert "Generally safe" in body
    assert "Severe reaction reported." not in body


def test_status_deduplicates_volatile_drug_labels(client, session):
    mesh_terms = ("Central core disease", "Anaesthesia")
    mesh_signature = compute_mesh_signature(list(mesh_terms))
    claim_set = ProcessedClaimSet(mesh_signature=mesh_signature, condition_label="Central core disease")
    session.add(claim_set)
    session.flush()

    version = ProcessedClaimSetVersion(
        claim_set=claim_set,
        version_number=1,
        status="active",
        pipeline_metadata={},
    )
    session.add(version)
    session.flush()

    volatile_claim = ProcessedClaim(
        claim_set=claim_set,
        claim_set_version=version,
        claim_id="claim:volatile-risk",
        classification="risk",
        summary="Volatile agents can trigger malignant hyperthermia.",
        confidence="medium",
        canonical_hash="hash-volatile",
        claim_group_id="group-volatile",
        drugs=["Volatile agents", "Volatile anaesthetic", "Volatile anaesthetics"],
        drug_classes=["volatile anesthetic"],
        source_claim_ids=["claim:volatile-risk"],
        severe_reaction_flag=True,
    )
    session.add(volatile_claim)
    session.flush()

    response = client.get(f"/ui/status/{quote(mesh_signature, safe='')}")

    assert response.status_code == 200
    body = response.data.decode()
    assert body.count('id="drug-filter-volatile-anaesthetics"') == 1
    assert 'id="drug-filter-volatile-agents"' not in body
    assert 'id="drug-filter-volatile-anaesthetic"' not in body
    assert "Volatile anaesthetics" in body
    assert 'data-drugs="volatile-anaesthetics"' in body
    assert "Volatile agents, Volatile anaesthetic, Volatile anaesthetics" in body


def test_ui_search_shows_refresh_prompt_when_search_results_change(client, session, monkeypatch):
    mesh_terms = ("King Denborough syndrome", "Anesthesia")
    mesh_signature = compute_mesh_signature(list(mesh_terms))

    claim_set = _seed_claim_set(session, mesh_terms=mesh_terms)

    term = SearchTerm(canonical="king denborough syndrome", slug="king-denborough-syndrome")
    session.add(term)
    session.flush()

    claim_set.last_search_term_id = term.id
    active_version = claim_set.get_active_version()
    active_version.pipeline_metadata = {
        "search_result": {
            "signature": "old-signature",
            "refreshed_at": "2025-11-15T00:00:00+00:00",
        }
    }
    session.flush()

    artefact_old = SearchArtefact(
        search_term_id=term.id,
        query_payload={},
        mesh_terms=list(mesh_terms),
        mesh_signature=mesh_signature,
        result_signature="old-signature",
        ttl_policy_seconds=86_400,
        last_refreshed_at=datetime(2025, 11, 15, 0, 0, tzinfo=timezone.utc),
    )
    artefact_new = SearchArtefact(
        search_term_id=term.id,
        query_payload={},
        mesh_terms=list(mesh_terms),
        mesh_signature=mesh_signature,
        result_signature="new-signature",
        ttl_policy_seconds=86_400,
        last_refreshed_at=datetime(2025, 11, 16, 3, 0, tzinfo=timezone.utc),
    )
    session.add_all([artefact_old, artefact_new])
    session.flush()

    resolution = SearchResolution(
        normalized_condition="king denborough syndrome",
        mesh_terms=list(mesh_terms),
        reused_cached=True,
        search_term_id=term.id,
        mesh_signature=mesh_signature,
        result_signature="new-signature",
        artefact_id=artefact_new.id,
        artefact_refreshed_at=artefact_new.last_refreshed_at,
    )

    monkeypatch.setattr("app.ui.routes._load_cached_resolution", lambda _session, _condition: resolution)

    response = client.post(
        "/ui/search",
        data={"condition": "King Denborough"},
        headers={"HX-Request": "true"},
    )

    assert response.status_code == 200
    body = response.data.decode()
    assert "New research available." in body
    assert "Run refresh" in body
    expected_url = f"/ui/retry/{quote(mesh_signature, safe='')}"
    assert f'hx-post="{expected_url}"' in body
    assert "Filter by drug" in body


def test_ui_search_preview_returns_confirmation(client, monkeypatch):
    monkeypatch.setattr("app.ui.routes._load_cached_resolution", lambda _session, _condition: None)
    preview = MeshResolutionPreview(
        status="resolved",
        raw_query="Hypertension",
        normalized_query="hypertension",
        mesh_terms=["Hypertension"],
        ranked_options=["Hypertension"],
        suggestions=[],
        espell_correction=None,
    )
    monkeypatch.setattr("app.ui.routes.preview_mesh_resolution", lambda condition: preview)

    response = client.post(
        "/ui/search-preview",
        data={"condition": "Hypertension"},
        headers={"HX-Request": "true"},
    )

    assert response.status_code == 200
    body = response.data.decode()
    assert "Confirm search" in body
    assert "Run refresh" in body
    assert "Hypertension" in body
    assert "View status" in body


def test_ui_search_preview_uses_cached_resolution(client, monkeypatch):
    resolution = SearchResolution(
        normalized_condition="hypertension",
        mesh_terms=["Hypertension"],
        reused_cached=True,
        search_term_id=123,
    )
    monkeypatch.setattr("app.ui.routes._load_cached_resolution", lambda _session, _condition: resolution)
    preview = MeshResolutionPreview(
        status="resolved",
        raw_query="Hypertension",
        normalized_query="hypertension",
        mesh_terms=["Hypertension"],
        ranked_options=["Hypertension"],
        suggestions=[],
    )
    monkeypatch.setattr("app.ui.routes.preview_mesh_resolution", lambda _condition: preview)

    response = client.post(
        "/ui/search-preview",
        data={"condition": "Hypertension"},
        headers={"HX-Request": "true"},
    )

    assert response.status_code == 200
    body = response.data.decode()
    assert "Confirm search" in body
    assert "Hypertension" in body
    assert "View status" in body


def test_ui_search_without_cache_shows_progress_panel(client, monkeypatch):
    monkeypatch.setattr("app.ui.routes._load_cached_resolution", lambda _session, _condition: None)
    monkeypatch.setattr(
        "app.ui.routes.preview_mesh_resolution",
        lambda condition: MeshResolutionPreview(
            status="resolved",
            raw_query=condition,
            normalized_query=condition.lower(),
            mesh_terms=["Example"],
            ranked_options=["Example"],
            suggestions=[],
            espell_correction=None,
        ),
    )
    monkeypatch.setattr(
        "app.ui.routes.resolve_condition_via_nih",
        lambda condition, session=None, mesh_builder=None: SearchResolution(
            normalized_condition=condition.lower(),
            mesh_terms=["Example"],
            reused_cached=False,
            search_term_id=1,
        ),
    )
    monkeypatch.setattr("app.ui.routes.enqueue_claim_pipeline", lambda **kwargs: {"job_id": "123", "status": "queued"})

    response = client.post(
        "/ui/search",
        data={"condition": "Unknown"},
        headers={"HX-Request": "true"},
    )

    assert response.status_code == 200
    body = response.data.decode()
    assert "Refresh status for Unknown" in body
    assert "Waiting for a worker to contact NIH eSearch" in body
    assert "NIH eSearch lookup" in body
    assert "Retrieve PubMed articles" in body
    assert "Assemble LLM batches" in body
    assert "Await LLM responses" in body
    assert "Using MeSH term" in body
    assert "Example" in body


def test_ui_search_boosted_request_returns_partial_panel(client, monkeypatch):
    monkeypatch.setattr("app.ui.routes._load_cached_resolution", lambda _session, _condition: None)
    monkeypatch.setattr(
        "app.ui.routes.preview_mesh_resolution",
        lambda condition: MeshResolutionPreview(
            status="resolved",
            raw_query=condition,
            normalized_query=condition.lower(),
            mesh_terms=["Example"],
            ranked_options=["Example"],
            suggestions=[],
            espell_correction=None,
        ),
    )
    monkeypatch.setattr(
        "app.ui.routes.resolve_condition_via_nih",
        lambda condition, session=None, mesh_builder=None: SearchResolution(
            normalized_condition=condition.lower(),
            mesh_terms=["Example"],
            reused_cached=False,
            search_term_id=1,
        ),
    )
    monkeypatch.setattr(
        "app.ui.routes.enqueue_claim_pipeline",
        lambda **kwargs: {"job_id": "123", "status": "queued", "stage": "queued"},
    )

    response = client.post(
        "/ui/search",
        data={"condition": "Unknown"},
        headers={
            "HX-Request": "true",
            "HX-Boosted": "true",
            "HX-Target": "results",
        },
    )

    assert response.status_code == 200
    body = response.data.decode()
    assert "id=\"job-status-panel\"" in body
    assert "Refresh status for Unknown" in body
    assert "<!doctype" not in body.lower()


def test_ui_status_without_htmx_wraps_in_layout(client, session, monkeypatch):
    job_id = "job-456"
    refresh = SimpleNamespace(
        mesh_signature="dummy-signature",
        job_id=job_id,
        status="running",
        progress_state="resolving_mesh_terms",
        progress_payload={"description": "Resolving terms"},
        error_message=None,
        created_at=datetime(2025, 11, 17, 12, 0, tzinfo=timezone.utc),
        updated_at=datetime(2025, 11, 17, 12, 5, tzinfo=timezone.utc),
    )

    monkeypatch.setattr("app.ui.routes._load_refresh_job", lambda _session, _ref: refresh)
    monkeypatch.setattr("app.ui.routes._load_claim_set_by_signature", lambda _session, _sig: None)

    response = client.get(f"/ui/status/{job_id}")

    assert response.status_code == 200
    body = response.data.decode()
    assert "<!doctype html>" in body.lower()
    assert "NeverDrugs" in body
    assert "Resolving terms" in body


def test_ui_search_prompts_for_mesh_selection(client, monkeypatch):
    monkeypatch.setattr("app.ui.routes._load_cached_resolution", lambda _session, _condition: None)
    preview = MeshResolutionPreview(
        status="needs_clarification",
        raw_query="LGMD",
        normalized_query="lgmd",
        mesh_terms=[],
        ranked_options=["Limb-Girdle Muscular Dystrophy", "Limb-Girdle Muscular Dystrophy, Type 2B"],
        suggestions=[],
        espell_correction=None,
    )
    monkeypatch.setattr("app.ui.routes.preview_mesh_resolution", lambda condition: preview)

    response = client.post(
        "/ui/search",
        data={"condition": "LGMD"},
        headers={"HX-Request": "true"},
    )

    assert response.status_code == 200
    body = response.data.decode()
    assert "Select MeSH term" in body
    assert "name=\"mesh_term\"" in body
    assert "Search with this term" in body


def test_ui_mesh_select_requires_mesh_term(client):
    response = client.post(
        "/ui/mesh-select",
        data={"condition": "LGMD"},
        headers={"HX-Request": "true"},
    )

    assert response.status_code == 400
    body = response.data.decode()
    assert "Select a MeSH term" in body


def test_ui_mesh_select_enqueues_manual_resolution(client, session, monkeypatch):
    selected_term = "Limb-Girdle Muscular Dystrophy"

    def fake_resolve(condition, session=None, mesh_builder=None):  # noqa: D401 - test helper
        assert condition == "LGMD"
        assert mesh_builder is not None
        builder_result = mesh_builder(condition)
        assert builder_result.mesh_terms == [selected_term]
        return SearchResolution(
            normalized_condition="lgmd",
            mesh_terms=[selected_term],
            reused_cached=False,
            search_term_id=456,
        )

    monkeypatch.setattr("app.ui.routes.resolve_condition_via_nih", fake_resolve)
    monkeypatch.setattr(
        "app.ui.routes.enqueue_claim_pipeline",
        lambda **kwargs: {"job_id": "job-123", "status": "queued"},
    )

    response = client.post(
        "/ui/mesh-select",
        data={"condition": "LGMD", "mesh_term": selected_term},
        headers={"HX-Request": "true"},
    )

    assert response.status_code == 200
    body = response.data.decode()
    assert "Refresh status for LGMD" in body
    assert "Waiting for a worker to contact NIH eSearch" in body
    assert "NIH eSearch lookup" in body
    assert "Using MeSH term" in body
    assert selected_term in body


def test_ui_status_returns_progress_panel(client, session, monkeypatch):
    job_id = "job-123"
    mesh_terms = ["King Denborough syndrome", "Anesthesia"]
    signature = compute_mesh_signature(mesh_terms)
    refresh = SimpleNamespace(
        mesh_signature=signature,
        job_id=job_id,
        status="running",
        progress_state="fetching_pubmed_articles",
        progress_payload={"description": "Fetching PubMed articles and any available full-text content for the condition."},
        error_message=None,
        created_at=datetime(2025, 11, 16, 1, 0, tzinfo=timezone.utc),
        updated_at=datetime(2025, 11, 16, 1, 5, tzinfo=timezone.utc),
    )

    claim_set = _seed_claim_set(session, mesh_terms=mesh_terms)

    monkeypatch.setattr(
        "app.ui.routes._load_refresh_job",
        lambda _session, _ref: refresh,
    )
    monkeypatch.setattr(
        "app.ui.routes._load_claim_set_by_signature",
        lambda _session, sign: claim_set if sign == signature else None,
    )

    response = client.get(f"/ui/status/{job_id}", headers={"HX-Request": "true"})

    assert response.status_code == 200
    body = response.data.decode()
    assert "Collecting PubMed results and downloading candidate articles for full-text review." in body
    assert "Last updated" in body
    assert "No further updates will be requested automatically." not in body
    assert "NIH eSearch lookup" in body
    assert "Using MeSH term" in body
    assert "Anesthesia" in body


def test_ui_status_boosted_poll_returns_partial_panel(client, session, monkeypatch):
    job_id = "job-789"
    mesh_terms = ["King Denborough syndrome"]
    signature = compute_mesh_signature(mesh_terms)
    refresh = SimpleNamespace(
        mesh_signature=signature,
        job_id=job_id,
        status="running",
        progress_state="generating_claims",
        progress_payload={
            "description": "Calling the OpenAI API",
            "batch_count": 3,
            "batches_completed": 1,
            "average_batch_seconds": 150,
            "average_batch_tokens": 30_000,
            "estimated_total_seconds": 450,
            "estimated_total_tokens": 90_000,
            "estimated_remaining_seconds": 300,
            "estimated_remaining_tokens": 60_000,
        },
        error_message=None,
        created_at=datetime(2025, 11, 17, 10, 0, tzinfo=timezone.utc),
        updated_at=datetime(2025, 11, 17, 10, 2, tzinfo=timezone.utc),
    )

    monkeypatch.setattr("app.ui.routes._load_refresh_job", lambda _session, _ref: refresh)
    monkeypatch.setattr("app.ui.routes._load_claim_set_by_signature", lambda *_args, **_kwargs: None)

    response = client.get(
        f"/ui/status/{job_id}",
        headers={
            "HX-Request": "true",
            "HX-Boosted": "true",
            "HX-Target": "job-status-panel",
        },
    )

    assert response.status_code == 200
    body = response.data.decode()
    assert "id=\"job-status-panel\"" in body
    assert "Calling the OpenAI API" in body
    assert "Processing 3 LLM batches" in body
    assert "1 complete" in body
    assert "Approximately" in body
    assert "<!doctype" not in body.lower()


def test_ui_status_returns_completed_claims_with_alert(client, session, monkeypatch):
    job_id = "job-456"
    mesh_terms = ["King Denborough syndrome", "Anesthesia"]
    signature = compute_mesh_signature(mesh_terms)
    refresh = SimpleNamespace(
        mesh_signature=signature,
        job_id=job_id,
        status="completed",
        progress_state="completed",
        progress_payload={"description": "Claims assembled"},
        error_message=None,
        created_at=None,
        updated_at=None,
    )

    claim_set = _seed_claim_set(session, mesh_terms=mesh_terms)

    monkeypatch.setattr("app.ui.routes._load_refresh_job", lambda _session, _ref: refresh)
    monkeypatch.setattr(
        "app.ui.routes._load_claim_set_by_signature",
        lambda _session, sign: claim_set if sign == signature else None,
    )

    response = client.get(f"/ui/status/{job_id}", headers={"HX-Request": "true"})

    assert response.status_code == 200
    body = response.data.decode()
    assert "Claims refreshed successfully" in body
    assert "Succinylcholine precipitates malignant hyperthermia." in body
    assert f'href="/ui/claims/{claim_set.id}"' in body
    assert "Resolved MeSH terms" in body
    assert "Anesthesia" in body


def test_ui_status_empty_results_shows_notice(client, session, monkeypatch):
    job_id = "job-empty"
    mesh_terms = ["King Denborough syndrome", "Anesthesia"]
    signature = compute_mesh_signature(mesh_terms)
    refresh = SimpleNamespace(
        mesh_signature=signature,
        job_id=job_id,
        status="empty-results",
        progress_state="empty_results",
        progress_payload={"description": "No claims produced"},
        error_message=None,
        created_at=None,
        updated_at=None,
    )

    monkeypatch.setattr("app.ui.routes._load_refresh_job", lambda _session, _ref: refresh)
    monkeypatch.setattr("app.ui.routes._load_claim_set_by_signature", lambda _session, sign: None)

    response = client.get(f"/ui/status/{job_id}", headers={"HX-Request": "true"})

    assert response.status_code == 200
    body = response.data.decode()
    assert "No claims were generated" in body
    assert "Retry search" in body
    assert "No further updates will be requested automatically." in body
    assert "NIH eSearch lookup" in body
    assert "Using MeSH term" in body
    assert "Anesthesia" in body


def test_ui_status_failed_job_offers_retry(client, session, monkeypatch):
    job_id = "job-failed"
    mesh_terms = ["King Denborough syndrome", "Anesthesia"]
    signature = compute_mesh_signature(mesh_terms)
    refresh = SimpleNamespace(
        mesh_signature=signature,
        job_id=job_id,
        status="failed",
        progress_state="failed",
        progress_payload={"description": "Pipeline crashed"},
        error_message="OpenAI timeout",
        created_at=datetime(2025, 11, 16, 1, 0, tzinfo=timezone.utc),
        updated_at=datetime(2025, 11, 16, 1, 30, tzinfo=timezone.utc),
    )

    monkeypatch.setattr("app.ui.routes._load_refresh_job", lambda _session, _ref: refresh)
    monkeypatch.setattr("app.ui.routes._load_claim_set_by_signature", lambda _session, sign: None)

    response = client.get(f"/ui/status/{job_id}", headers={"HX-Request": "true"})

    assert response.status_code == 200
    body = response.data.decode()
    assert "OpenAI timeout" in body
    assert "Retry search" in body
    assert "Request failed" in body
    assert "No further updates will be requested automatically." in body
    assert "NIH eSearch lookup" in body
    assert "Using MeSH term" in body
    assert "Anesthesia" in body


def test_ui_claims_route_renders_claim_set(client, session):
    claim_set = _seed_claim_set(session)

    response = client.get(f"/ui/claims/{claim_set.id}")

    assert response.status_code == 200
    body = response.data.decode()
    assert "Succinylcholine precipitates malignant hyperthermia." in body
    assert "King Denborough syndrome" in body
    assert "Resolved MeSH terms" in body
    assert "Anesthesia" in body


def test_ui_status_placeholder_when_refresh_not_ready(client, monkeypatch):
    job_id = "job-pending"
    monkeypatch.setattr("app.ui.routes._load_refresh_job", lambda _session, _ref: None)
    monkeypatch.setattr("app.ui.routes._load_claim_set_by_signature", lambda _session, sign: None)

    response = client.get(
        f"/ui/status/{job_id}",
        headers={"HX-Request": "true"},
        query_string={"condition": "Test condition"},
    )

    assert response.status_code == 200
    body = response.data.decode()
    assert "Refresh status for Test condition" in body
    assert "Waiting for a worker to contact NIH eSearch" in body
    assert "NIH eSearch lookup" in body
    assert "Retry search" not in body

def test_ui_claims_deduplicates_evidence_entries(client, session):
    claim_set = _seed_claim_set(session)
    claim = claim_set.get_active_version().claims[0]

    session.add(
        ProcessedClaimEvidence(
            claim=claim,
            snippet_id="43",
            pmid="11111111",
            article_title="Safety considerations",
            citation_url="https://pubmed.ncbi.nlm.nih.gov/11111111/",
            key_points=["Additional supporting insight"],
            notes=None,
        )
    )
    session.commit()

    response = client.get(f"/ui/claims/{claim_set.id}")

    assert response.status_code == 200
    body = response.data.decode()
    assert body.count("Safety considerations") == 1
    assert "Case reports link succinylcholine to malignant hyperthermia" in body
    assert "Additional supporting insight" in body
