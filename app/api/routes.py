from __future__ import annotations

from flask import Blueprint, current_app, jsonify, g, request
from sqlalchemy import select

from app.models import ClaimSetRefresh, ProcessedClaim, ProcessedClaimDrugLink, ProcessedClaimEvidence, ProcessedClaimSet
from app.services.nih_pipeline import resolve_condition_via_nih
from app.services.search import compute_mesh_signature
from app.tasks import refresh_claims_for_condition

api_blueprint = Blueprint("api", __name__, url_prefix="/api")


def _get_db_session():
    session = getattr(g, "db_session", None)
    if session is None:
        raise RuntimeError("Database session is not initialised on the request context")
    return session


def _serialise_claim_set(claim_set: ProcessedClaimSet) -> dict:
    return {
        "id": claim_set.id,
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


@api_blueprint.get("/claims/<int:claim_set_id>")
def get_processed_claim(claim_set_id: int):
    session = _get_db_session()
    claim_set = session.get(ProcessedClaimSet, claim_set_id)
    if claim_set is None:
        return jsonify({"detail": "Processed claim set not found"}), 404

    payload = _serialise_claim_set(claim_set)
    return jsonify(payload), 200


@api_blueprint.post("/claims/resolve")
def resolve_claims():
    session = _get_db_session()
    body = request.get_json(silent=True) or {}
    condition = (body.get("condition") or "").strip()
    if not condition:
        return jsonify({"detail": "Condition is required"}), 400

    resolution = resolve_condition_via_nih(condition, session=session)
    mesh_signature = compute_mesh_signature(list(resolution.mesh_terms))

    claim_set: ProcessedClaimSet | None = None
    refresh_job: ClaimSetRefresh | None = None
    if mesh_signature:
        refresh_stmt = select(ClaimSetRefresh).where(ClaimSetRefresh.mesh_signature == mesh_signature)
        refresh_job = session.execute(refresh_stmt).scalar_one_or_none()

    if resolution.reused_cached and mesh_signature:
        stmt = select(ProcessedClaimSet).where(ProcessedClaimSet.mesh_signature == mesh_signature)
        claim_set = session.execute(stmt).scalar_one_or_none()

    status_code = 200
    job_info: dict[str, object] | None = None
    if claim_set is None:
        if refresh_job and refresh_job.status in {"queued", "running"}:
            job_info = {"job_id": refresh_job.job_id, "status": refresh_job.status}
        else:
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
                    )
                    session.add(refresh_job)
                else:
                    refresh_job.job_id = str(job_info.get("job_id", ""))
                    refresh_job.status = str(job_info.get("status", "queued"))
                    refresh_job.error_message = None

        status_code = 202

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

    return jsonify(response_payload), status_code


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
