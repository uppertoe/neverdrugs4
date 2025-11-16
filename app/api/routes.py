from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from urllib.parse import quote, unquote

from flask import Blueprint, current_app, jsonify, g, request
from sqlalchemy import select

from app.models import (
    ArticleArtefact,
    ArticleSnippet,
    ClaimSetRefresh,
    ProcessedClaim,
    ProcessedClaimDrugLink,
    ProcessedClaimEvidence,
    ProcessedClaimSet,
    SearchArtefact,
    SearchTerm,
    SearchTermVariant,
)
from app.job_queue import enqueue_claim_refresh
from app.services.mesh_resolution import preview_mesh_resolution
from app.services.nih_pipeline import MeshTermsNotFoundError, resolve_condition_via_nih
from app.services.query_terms import build_nih_search_query
from app.services.search import MeshBuildResult, SearchResolution, compute_mesh_signature, normalize_condition

DEFAULT_REFRESH_STALE_SECONDS = 300
DEFAULT_REFRESH_STALE_QUEUE_SECONDS = 60
DEFAULT_EMPTY_RESULT_RETRY_SECONDS = 1800


api_blueprint = Blueprint("api", __name__, url_prefix="/api")


def _get_db_session():
    session = getattr(g, "db_session", None)
    if session is None:
        raise RuntimeError("Database session is not initialised on the request context")
    return session


def _load_claim_set(session, ref: str) -> ProcessedClaimSet | None:
    if ref.isdigit():
        return session.get(ProcessedClaimSet, int(ref))
    stmt = select(ProcessedClaimSet).where(ProcessedClaimSet.slug == ref)
    return session.execute(stmt).scalar_one_or_none()


def _load_search_term(session, ref: str) -> SearchTerm | None:
    if ref.isdigit():
        return session.get(SearchTerm, int(ref))
    stmt = select(SearchTerm).where(SearchTerm.slug == ref)
    return session.execute(stmt).scalar_one_or_none()


def _artefact_is_current(artefact: SearchArtefact) -> bool:
    ttl_seconds = int(artefact.ttl_policy_seconds or 0)
    if ttl_seconds <= 0:
        return True

    timestamp = artefact.last_refreshed_at or artefact.created_at
    if timestamp is None:
        return False
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)

    now = datetime.now(timestamp.tzinfo)
    age_seconds = (now - timestamp).total_seconds()
    return age_seconds < ttl_seconds


def _freshest_valid_artefact(session, term: SearchTerm) -> SearchArtefact | None:
    stmt = (
        select(SearchArtefact)
        .where(SearchArtefact.search_term_id == term.id)
        .order_by(SearchArtefact.last_refreshed_at.desc(), SearchArtefact.created_at.desc())
    )

    for artefact in session.execute(stmt).scalars():
        if _artefact_is_current(artefact):
            return artefact
    return None


def _load_cached_resolution(session, raw_condition: str) -> SearchResolution | None:
    normalized = normalize_condition(raw_condition)
    stmt = select(SearchTerm).where(SearchTerm.canonical == normalized)
    term = session.execute(stmt).scalar_one_or_none()

    if term is None:
        variant_stmt = (
            select(SearchTerm)
            .join(SearchTermVariant)
            .where(SearchTermVariant.normalized_value == normalized)
            .limit(1)
        )
        term = session.execute(variant_stmt).scalar_one_or_none()

    if term is None:
        return None

    artefact = _freshest_valid_artefact(session, term)
    if artefact is None:
        return None

    return SearchResolution(
        normalized_condition=term.canonical,
        mesh_terms=list(artefact.mesh_terms),
        reused_cached=True,
        search_term_id=term.id,
    )


def _serialise_claim_set(claim_set: ProcessedClaimSet) -> dict:
    active_version = claim_set.get_active_version()
    claims = []
    if active_version is not None:
        claims = [
            _serialise_claim(claim)
            for claim in sorted(active_version.claims, key=lambda c: c.id)
        ]

    return {
        "id": claim_set.id,
        "slug": claim_set.slug,
        "mesh_signature": claim_set.mesh_signature,
        "condition_label": claim_set.condition_label,
        "active_version": None
        if active_version is None
        else {
            "id": active_version.id,
            "version_number": active_version.version_number,
            "status": active_version.status,
            "created_at": active_version.created_at.isoformat()
            if active_version.created_at
            else None,
            "activated_at": active_version.activated_at.isoformat()
            if active_version.activated_at
            else None,
        },
        "claims": claims,
    }


def _serialise_claim(claim: ProcessedClaim) -> dict:
    return {
        "id": claim.id,
        "claim_id": claim.claim_id,
        "classification": claim.classification,
        "summary": claim.summary,
        "confidence": claim.confidence,
        "drugs": list(claim.drugs),
        "drug_classes": list(claim.drug_classes),
        "source_claim_ids": list(claim.source_claim_ids),
        "severe_reaction": {
            "flag": bool(claim.severe_reaction_flag),
            "terms": list(claim.severe_reaction_terms),
        },
        "supporting_evidence": [
            _serialise_evidence(evidence)
            for evidence in sorted(claim.evidence, key=lambda e: e.id)
        ],
        "drug_links": [
            _serialise_drug_link(link)
            for link in sorted(claim.drug_links, key=lambda l: (l.term_kind, l.term))
        ],
    }


def _serialise_evidence(evidence: ProcessedClaimEvidence) -> dict:
    return {
        "id": evidence.id,
        "snippet_id": evidence.snippet_id,
        "pmid": evidence.pmid,
        "article_title": evidence.article_title,
        "citation_url": evidence.citation_url,
        "key_points": list(evidence.key_points),
        "notes": evidence.notes,
    }


def _serialise_drug_link(link: ProcessedClaimDrugLink) -> dict:
    return {
        "id": link.id,
        "term": link.term,
        "term_kind": link.term_kind,
    }


def _mesh_terms_from_signature(signature: str | None) -> list[str]:
    if not signature:
        return []
    parts = [part.strip() for part in signature.split("|") if part.strip()]
    return [part.title() for part in parts]


def _build_resolution_snapshot(
    refresh: ClaimSetRefresh,
    claim_set: ProcessedClaimSet | None,
    progress_details: dict,
) -> dict | None:
    resolution_details = progress_details.get("resolution")
    normalized_condition: str | None = None
    mesh_terms: list[str] = []

    if isinstance(resolution_details, dict):
        normalized_condition = resolution_details.get("normalized_condition")
        mesh_terms = [str(term) for term in resolution_details.get("mesh_terms", []) if term]

    signature_source: str | None = None
    if claim_set and claim_set.mesh_signature:
        signature_source = claim_set.mesh_signature
    elif refresh.mesh_signature:
        signature_source = refresh.mesh_signature

    if not normalized_condition and signature_source:
        normalized_condition = signature_source.replace("|", " ").strip() or None

    if not mesh_terms:
        mesh_terms = _mesh_terms_from_signature(signature_source)

    if normalized_condition or mesh_terms:
        return {
            "normalized_condition": normalized_condition,
            "mesh_terms": mesh_terms,
        }
    return None


def _refresh_can_retry(refresh: ClaimSetRefresh) -> bool:
    return refresh.status in {"failed", "no-batches", "no-responses"}


def _empty_retry_threshold_seconds() -> int:
    configured = current_app.config.get(
        "REFRESH_EMPTY_RESULT_RETRY_SECONDS",
        DEFAULT_EMPTY_RESULT_RETRY_SECONDS,
    )
    try:
        threshold = int(configured)
    except (TypeError, ValueError):
        threshold = DEFAULT_EMPTY_RESULT_RETRY_SECONDS
    return max(threshold, 0)


def _refresh_empty_can_retry(refresh: ClaimSetRefresh) -> bool:
    if refresh.status != "empty-results":
        return False

    timestamp = refresh.updated_at or refresh.created_at
    if timestamp is None:
        return True

    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)

    age_seconds = (datetime.now(timezone.utc) - timestamp).total_seconds()
    return age_seconds >= _empty_retry_threshold_seconds()


def _claim_set_empty_can_retry(claim_set: ProcessedClaimSet | None) -> bool:
    if not claim_set:
        return False

    if claim_set.get_active_claims():
        return False

    timestamp = claim_set.updated_at or claim_set.created_at
    if timestamp is None:
        return True

    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)

    age_seconds = (datetime.now(timezone.utc) - timestamp).total_seconds()
    return age_seconds >= _empty_retry_threshold_seconds()


def _refresh_is_stale(refresh: ClaimSetRefresh) -> bool:
    if refresh.status not in {"queued", "running"}:
        return False
    updated_at = refresh.updated_at
    if updated_at is None:
        return False

    if updated_at.tzinfo is None:
        updated_at = updated_at.replace(tzinfo=timezone.utc)

    now = datetime.now(timezone.utc)
    stage = (refresh.progress_state or "").strip().lower()

    if stage == "queued":
        configured_threshold = current_app.config.get(
            "REFRESH_JOB_STALE_QUEUE_SECONDS",
            DEFAULT_REFRESH_STALE_QUEUE_SECONDS,
        )
        default_threshold = DEFAULT_REFRESH_STALE_QUEUE_SECONDS
    else:
        configured_threshold = current_app.config.get(
            "REFRESH_JOB_STALE_SECONDS",
            DEFAULT_REFRESH_STALE_SECONDS,
        )
        default_threshold = DEFAULT_REFRESH_STALE_SECONDS

    try:
        threshold_seconds = int(configured_threshold)
    except (TypeError, ValueError):
        threshold_seconds = default_threshold

    threshold_seconds = max(threshold_seconds, 0)
    age_seconds = (now - updated_at).total_seconds()

    return age_seconds >= threshold_seconds


def _parse_mesh_terms(raw_terms) -> list[str]:
    if raw_terms is None:
        return []
    if not isinstance(raw_terms, (list, tuple)):
        raise ValueError("mesh_terms must be provided as a list of strings")

    terms: list[str] = []
    seen: set[str] = set()
    for raw in raw_terms:
        if not isinstance(raw, str):
            raise ValueError("mesh_terms must be provided as a list of strings")
        cleaned = raw.strip()
        if not cleaned:
            continue
        normalized = normalize_condition(cleaned)
        if normalized in seen:
            continue
        seen.add(normalized)
        terms.append(cleaned)

    if not terms:
        raise ValueError("mesh_terms must include at least one non-empty entry")
    return terms


def _manual_mesh_builder(selected_terms: list[str]):
    payload_template = {
        "manual_selection": True,
        "selected_mesh_terms": list(selected_terms),
    }

    def _builder(_: str) -> MeshBuildResult:
        payload = deepcopy(payload_template)
        try:
            query = build_nih_search_query(selected_terms)
        except ValueError:
            query = None
        payload["esearch"] = {"query": query}
        payload["ranked_mesh_terms"] = [{"term": term} for term in selected_terms]
        return MeshBuildResult(
            mesh_terms=list(selected_terms),
            query_payload=payload,
            source="manual-selection",
        )

    return _builder


def _serialise_refresh_job(
    refresh: ClaimSetRefresh,
    claim_set: ProcessedClaimSet | None,
) -> dict:
    progress_details = refresh.progress_payload if isinstance(refresh.progress_payload, dict) else {}
    can_retry = (
        _refresh_can_retry(refresh)
        or _refresh_is_stale(refresh)
        or _refresh_empty_can_retry(refresh)
    )
    status_label = refresh.status
    description = progress_details.get("description") if isinstance(progress_details, dict) else None
    if isinstance(description, str) and description:
        status_label = description
    elif refresh.status in {"queued", "running"} and (refresh.progress_state or "").strip():
        status_label = refresh.progress_state.replace("_", " ")
    return {
        "mesh_signature": refresh.mesh_signature,
        "job_id": refresh.job_id,
        "status": refresh.status,
        "status_label": status_label,
        "error_message": refresh.error_message,
        "created_at": refresh.created_at.isoformat() if refresh.created_at else None,
        "updated_at": refresh.updated_at.isoformat() if refresh.updated_at else None,
        "progress": {
            "stage": refresh.progress_state,
            "details": progress_details,
        },
        "claim_set_id": claim_set.id if claim_set else None,
        "claim_set_slug": claim_set.slug if claim_set else None,
        "resolution": _build_resolution_snapshot(refresh, claim_set, progress_details),
        "can_retry": can_retry,
    }


@api_blueprint.get("/claims/<string:claim_set_ref>")
def get_processed_claim(claim_set_ref: str):
    session = _get_db_session()
    claim_set = _load_claim_set(session, claim_set_ref)
    if claim_set is None:
        return jsonify({"detail": "Processed claim set not found"}), 404

    payload = _serialise_claim_set(claim_set)
    return jsonify(payload), 200


@api_blueprint.get("/claims/refresh/<path:mesh_signature>")
def get_refresh_status(mesh_signature: str):
    session = _get_db_session()
    decoded_signature = unquote(mesh_signature)
    stmt = select(ClaimSetRefresh).where(ClaimSetRefresh.mesh_signature == decoded_signature)
    refresh_job = session.execute(stmt).scalar_one_or_none()
    claim_set: ProcessedClaimSet | None = None

    if refresh_job is None:
        claim_set = _load_claim_set(session, decoded_signature)
        if claim_set is not None:
            stmt = select(ClaimSetRefresh).where(ClaimSetRefresh.mesh_signature == claim_set.mesh_signature)
            refresh_job = session.execute(stmt).scalar_one_or_none()

    if refresh_job is None:
        return jsonify({"detail": "Refresh job not found"}), 404

    if claim_set is None:
        claim_stmt = select(ProcessedClaimSet).where(ProcessedClaimSet.mesh_signature == refresh_job.mesh_signature)
        claim_set = session.execute(claim_stmt).scalar_one_or_none()

    payload = _serialise_refresh_job(refresh_job, claim_set)
    return jsonify(payload), 200


@api_blueprint.post("/claims/resolve")
def resolve_claims():
    session = _get_db_session()
    body = request.get_json(silent=True) or {}
    condition = (body.get("condition") or "").strip()
    if not condition:
        return jsonify({"detail": "Condition is required"}), 400

    preview_result = None
    manual_mesh_terms: list[str] = []
    manual_builder = None
    mesh_terms_supplied = "mesh_terms" in body
    cached_resolution: SearchResolution | None = None
    if mesh_terms_supplied:
        try:
            manual_mesh_terms = _parse_mesh_terms(body.get("mesh_terms"))
        except ValueError as exc:
            return jsonify({"detail": str(exc)}), 400
        manual_builder = _manual_mesh_builder(manual_mesh_terms)
    else:
        cached_resolution = _load_cached_resolution(session, condition)
        if cached_resolution is None:
            preview_result = preview_mesh_resolution(condition)
            if preview_result.status == "not_found":
                return (
                    jsonify(
                        {
                            "detail": "No MeSH terms matched the supplied condition. Manual input is required.",
                            "resolution_preview": asdict(preview_result),
                        }
                    ),
                    422,
                )
            if preview_result.status == "needs_clarification":
                return (
                    jsonify(
                        {
                            "detail": "Multiple MeSH terms matched the supplied condition. Clarification required.",
                            "resolution_preview": asdict(preview_result),
                        }
                    ),
                    409,
                )

    resolution: SearchResolution | None = cached_resolution

    if resolution is None:
        try:
            kwargs = {"session": session}
            if manual_builder is not None:
                kwargs["mesh_builder"] = manual_builder
            resolution = resolve_condition_via_nih(condition, **kwargs)
        except MeshTermsNotFoundError as exc:
            return (
                jsonify(
                    {
                        "detail": "No MeSH terms matched the supplied condition. Manual input is required.",
                        "resolution": {
                            "normalized_condition": exc.normalized_condition,
                            "mesh_terms": [],
                            "reused_cached": False,
                            "search_term_id": exc.search_term_id,
                        },
                        "suggested_mesh_terms": list(exc.suggestions),
                    }
                ),
                422,
            )

    assert resolution is not None
    mesh_signature = compute_mesh_signature(list(resolution.mesh_terms))
    claim_set: ProcessedClaimSet | None = None
    refresh_job: ClaimSetRefresh | None = None
    if mesh_signature:
        refresh_stmt = select(ClaimSetRefresh).where(ClaimSetRefresh.mesh_signature == mesh_signature)
        refresh_job = session.execute(refresh_stmt).scalar_one_or_none()

    if resolution.reused_cached and mesh_signature:
        stmt = select(ProcessedClaimSet).where(ProcessedClaimSet.mesh_signature == mesh_signature)
        claim_set = session.execute(stmt).scalar_one_or_none()

    job_info: dict[str, object] | None = None
    refresh_url: str | None = None
    should_enqueue = False
    need_claim_set = claim_set is None
    claim_set_empty = claim_set is not None and not claim_set.get_active_claims()
    resolved_signature = mesh_signature or compute_mesh_signature(list(resolution.mesh_terms))

    if not mesh_signature:
        should_enqueue = True
    elif refresh_job is None:
        should_enqueue = need_claim_set or _claim_set_empty_can_retry(claim_set)
    elif (
        _refresh_can_retry(refresh_job)
        or _refresh_is_stale(refresh_job)
        or _refresh_empty_can_retry(refresh_job)
    ):
        should_enqueue = True
    elif claim_set_empty and _claim_set_empty_can_retry(claim_set):
        should_enqueue = True

    if should_enqueue:
        try:
            job_info = enqueue_claim_pipeline(
                session=session,
                resolution=resolution,
                condition_label=condition,
                mesh_signature=mesh_signature,
            )
        except Exception as exc:  # noqa: BLE001 - surface enqueue failures
            current_app.logger.exception(
                "Failed to queue background refresh",
                extra={"condition": condition, "mesh_signature": mesh_signature},
            )
            return jsonify({"detail": "Failed to queue background refresh"}), 503

        if mesh_signature:
            if refresh_job is None:
                refresh_job = ClaimSetRefresh(
                    mesh_signature=mesh_signature,
                    job_id=str(job_info.get("job_id", "")),
                    status=str(job_info.get("status", "queued")),
                    progress_state="queued",
                    progress_payload={},
                )
                session.add(refresh_job)
            else:
                refresh_job.job_id = str(job_info.get("job_id", ""))
                refresh_job.status = str(job_info.get("status", "queued"))
                refresh_job.error_message = None
                refresh_job.progress_state = "queued"
                refresh_job.progress_payload = {}
        if resolved_signature:
            refresh_url = f"/api/claims/refresh/{quote(resolved_signature, safe='')}"
    elif refresh_job is not None and refresh_job.status in {"queued", "running"}:
        job_info = {"job_id": refresh_job.job_id, "status": refresh_job.status}
        signature_for_url = refresh_job.mesh_signature or resolved_signature
        if signature_for_url:
            refresh_url = f"/api/claims/refresh/{quote(signature_for_url, safe='')}"
    elif refresh_job is not None and refresh_job.status == "empty-results":
        job_info = {"job_id": refresh_job.job_id, "status": refresh_job.status}
        signature_for_url = refresh_job.mesh_signature or resolved_signature
        if signature_for_url:
            refresh_url = f"/api/claims/refresh/{quote(signature_for_url, safe='')}"

    response_payload = {
        "resolution": {
            "normalized_condition": resolution.normalized_condition,
            "mesh_terms": list(resolution.mesh_terms),
            "reused_cached": resolution.reused_cached,
            "search_term_id": resolution.search_term_id,
        },
        "claim_set": _serialise_claim_set(claim_set) if claim_set else None,
        "job": job_info,
    }

    if refresh_url and job_info is not None:
        response_payload["refresh_url"] = refresh_url

    return jsonify(response_payload), 200


def enqueue_claim_pipeline(
    *,
    session,
    resolution,
    condition_label: str,
    mesh_signature: str | None,
) -> dict[str, object]:
    """Schedule the claim processing pipeline for the provided resolution."""
    resolved_signature = mesh_signature or compute_mesh_signature(list(resolution.mesh_terms))
    result = enqueue_claim_refresh(
        session=session,
        resolution=resolution,
        condition_label=condition_label,
        mesh_signature=resolved_signature,
    )
    return dict(result)


@api_blueprint.get("/search/<string:search_term_ref>/query")
def get_search_query(search_term_ref: str):
    session = _get_db_session()
    term = _load_search_term(session, search_term_ref)
    if term is None:
        return jsonify({"detail": "Search term not found"}), 404

    artefact_stmt = (
        select(SearchArtefact)
        .where(SearchArtefact.search_term_id == term.id)
        .order_by(SearchArtefact.last_refreshed_at.desc(), SearchArtefact.created_at.desc())
        .limit(1)
    )
    artefact = session.execute(artefact_stmt).scalar_one_or_none()
    if artefact is None:
        return jsonify({"detail": "Search query not available"}), 404

    payload = artefact.query_payload if isinstance(artefact.query_payload, dict) else {}
    esearch_payload = payload.get("esearch") if isinstance(payload.get("esearch"), dict) else {}
    query = esearch_payload.get("query")

    return (
        jsonify(
            {
                "search_term_id": term.id,
                "search_term_slug": term.slug,
                "canonical_condition": term.canonical,
                "mesh_terms": list(artefact.mesh_terms),
                "mesh_signature": artefact.mesh_signature,
                "query": query,
                "query_payload": payload,
                "result_signature": artefact.result_signature,
                "last_refreshed_at": artefact.last_refreshed_at.isoformat()
                if artefact.last_refreshed_at
                else None,
            }
        ),
        200,
    )


@api_blueprint.get("/search/<string:search_term_ref>/articles")
def get_search_articles(search_term_ref: str):
    session = _get_db_session()
    term = _load_search_term(session, search_term_ref)
    if term is None:
        return jsonify({"detail": "Search term not found"}), 404

    articles_stmt = (
        select(ArticleArtefact)
        .where(ArticleArtefact.search_term_id == term.id)
        .order_by(ArticleArtefact.rank.asc())
    )
    articles = session.execute(articles_stmt).scalars().all()

    items: list[dict[str, object]] = []
    for article in articles:
        citation = article.citation if isinstance(article.citation, dict) else {}
        items.append(
            {
                "pmid": article.pmid,
                "rank": article.rank,
                "score": article.score,
                "citation": citation,
                "preferred_url": citation.get("preferred_url") if isinstance(citation, dict) else None,
                "content_source": article.content_source,
                "retrieved_at": article.retrieved_at.isoformat() if article.retrieved_at else None,
            }
        )

    return (
        jsonify(
            {
                "search_term_id": term.id,
                "search_term_slug": term.slug,
                "canonical_condition": term.canonical,
                "articles": items,
            }
        ),
        200,
    )


@api_blueprint.get("/search/<string:search_term_ref>/snippets")
def get_search_snippets(search_term_ref: str):
    session = _get_db_session()
    term = _load_search_term(session, search_term_ref)
    if term is None:
        return jsonify({"detail": "Search term not found"}), 404

    snippet_stmt = (
        select(ArticleSnippet, ArticleArtefact)
        .join(ArticleArtefact, ArticleSnippet.article_artefact_id == ArticleArtefact.id)
        .where(ArticleArtefact.search_term_id == term.id)
        .order_by(ArticleArtefact.rank.asc(), ArticleSnippet.id.asc())
    )
    rows = session.execute(snippet_stmt).all()

    snippets: list[dict[str, object]] = []
    for snippet, article in rows:
        snippets.append(
            {
                "snippet_id": snippet.id,
                "pmid": article.pmid,
                "article_rank": article.rank,
                "drug": snippet.drug,
                "classification": snippet.classification,
                "snippet_text": snippet.snippet_text,
                "snippet_score": snippet.snippet_score,
                "cues": list(snippet.cues or []),
                "snippet_hash": snippet.snippet_hash,
            }
        )

    return (
        jsonify(
            {
                "search_term_id": term.id,
                "search_term_slug": term.slug,
                "canonical_condition": term.canonical,
                "snippets": snippets,
            }
        ),
        200,
    )
