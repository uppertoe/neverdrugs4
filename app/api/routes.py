from __future__ import annotations

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
)
from app.services.nih_pipeline import MeshTermsNotFoundError, resolve_condition_via_nih
from app.services.search import compute_mesh_signature
from app.tasks import refresh_claims_for_condition

DEFAULT_REFRESH_STALE_SECONDS = 300
DEFAULT_REFRESH_STALE_QUEUE_SECONDS = 60


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


def _serialise_claim_set(claim_set: ProcessedClaimSet) -> dict:
    return {
        "id": claim_set.id,
        "slug": claim_set.slug,
        "mesh_signature": claim_set.mesh_signature,
        "condition_label": claim_set.condition_label,
        "claims": [_serialise_claim(claim) for claim in sorted(claim_set.claims, key=lambda c: c.id)],
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


def _serialise_refresh_job(
    refresh: ClaimSetRefresh,
    claim_set: ProcessedClaimSet | None,
) -> dict:
    progress_details = refresh.progress_payload if isinstance(refresh.progress_payload, dict) else {}
    return {
        "mesh_signature": refresh.mesh_signature,
        "job_id": refresh.job_id,
        "status": refresh.status,
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
        "can_retry": _refresh_can_retry(refresh) or _refresh_is_stale(refresh),
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

    try:
        resolution = resolve_condition_via_nih(condition, session=session)
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
    resolved_signature = mesh_signature or compute_mesh_signature(list(resolution.mesh_terms))

    if not mesh_signature:
        should_enqueue = True
    elif refresh_job is None:
        should_enqueue = need_claim_set
    elif _refresh_can_retry(refresh_job) or _refresh_is_stale(refresh_job):
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
    _ = session  # reserved for later use (e.g., recording job metadata)
    resolved_signature = mesh_signature or compute_mesh_signature(list(resolution.mesh_terms))
    async_result = refresh_claims_for_condition.delay(
        resolution_id=resolution.search_term_id,
        condition_label=condition_label,
        normalized_condition=resolution.normalized_condition,
        mesh_terms=list(resolution.mesh_terms),
        mesh_signature=resolved_signature,
    )
    return {
        "job_id": async_result.id,
        "status": "queued",
    }


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
