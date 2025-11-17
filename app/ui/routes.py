from __future__ import annotations

from collections.abc import Sequence
from contextlib import closing
from datetime import datetime, timezone
import re

from flask import Blueprint, current_app, render_template, request, url_for
from sqlalchemy import select

from app.api.routes import (
    _claim_set_empty_can_retry,
    _get_db_session,
    _load_cached_resolution,
    _manual_mesh_builder,
    _mesh_terms_from_signature,
    _refresh_can_retry,
    _refresh_empty_can_retry,
    enqueue_claim_pipeline,
)
from app.models import ClaimSetRefresh, ProcessedClaimSet, SearchArtefact
from app.services.mesh_resolution import MeshResolutionPreview, preview_mesh_resolution
from app.services.nih_pipeline import MeshTermsNotFoundError, resolve_condition_via_nih
from app.services.search import compute_mesh_signature, normalize_condition

ui_blueprint = Blueprint(
    "ui",
    __name__,
    url_prefix="/ui",
    template_folder="templates",
)

TERMINAL_STATUSES = {
    "completed",
    "failed",
    "empty-results",
    "no-batches",
    "no-responses",
    "skipped",
}

PIPELINE_STEPS: list[dict[str, str]] = [
    {
        "slug": "resolving_mesh_terms",
        "label": "NIH eSearch lookup",
        "description": "Contacting NIH eSearch to normalise the condition and pick MeSH terms.",
    },
    {
        "slug": "fetching_pubmed_articles",
        "label": "Retrieve PubMed articles",
        "description": "Collecting PubMed results and downloading candidate articles for full-text review.",
    },
    {
        "slug": "preparing_llm_batches",
        "label": "Assemble LLM batches",
        "description": "Grouping relevant snippets into batches ready for the language model.",
    },
    {
        "slug": "generating_claims",
        "label": "Await LLM responses",
        "description": "Waiting for the language model to generate refreshed claims.",
    },
    {
        "slug": "saving_processed_claims",
        "label": "Save refreshed claims",
        "description": "Persisting the refreshed claims and evidence to the database.",
    },
]

PIPELINE_STAGE_INDEX = {step["slug"]: idx for idx, step in enumerate(PIPELINE_STEPS)}

SEVERITY_ORDER: dict[str, int] = {
    "severe": 0,
    "risk": 1,
    "nuanced": 2,
    "uncertain": 3,
    "safety": 4,
    "info": 5,
}

SEVERITY_META: dict[str, dict[str, str]] = {
    "severe": {
        "label": "Severe reaction",
        "badge": "danger",
        "description": "Documented severe or life-threatening reaction.",
    },
    "risk": {
        "label": "Elevated risk",
        "badge": "danger",
        "description": "Evidence indicates a significant safety concern.",
    },
    "nuanced": {
        "label": "Nuanced response",
        "badge": "info",
        "description": "Outcome depends on patient factors or context.",
    },
    "uncertain": {
        "label": "Limited evidence",
        "badge": "warning",
        "description": "Conflicting or inconclusive findings.",
    },
    "safety": {
        "label": "Generally safe",
        "badge": "success",
        "description": "Evidence supports routine use without major concern.",
    },
    "info": {
        "label": "Informational",
        "badge": "secondary",
        "description": "Contextual information without a clear risk signal.",
    },
}

CONFIDENCE_ORDER: dict[str, int] = {
    "high": 0,
    "medium": 1,
    "low": 2,
}

STAGE_SUMMARIES: dict[str, str] = {
    "queued": "Waiting for a worker to contact NIH eSearch and start the refresh.",
    "running": "Processing the refresh request.",
    "empty-results": "Processing completed, but no claims were produced for this condition.",
    "failed": "The refresh failed before completion.",
    "no-batches": "No snippets were available to create language model batches.",
    "no-responses": "The language model returned no usable responses.",
    "skipped": "The refresh was skipped because the condition could not be resolved.",
}

for step in PIPELINE_STEPS:
    STAGE_SUMMARIES.setdefault(step["slug"], step["description"])


_RAW_DRUG_SYNONYM_GROUPS: dict[str, set[str]] = {
    "Volatile anaesthetics": {
        "Volatile anaesthetic",
        "Volatile anesthetic",
        "Volatile anesthetics",
        "Volatile agents",
        "Volatile agent",
    }
}

DRUG_SYNONYM_MAP: dict[str, set[str]] = {
    normalize_condition(canonical): {normalize_condition(option) for option in {canonical, *variants}}
    for canonical, variants in _RAW_DRUG_SYNONYM_GROUPS.items()
}

DRUG_CANONICAL_DISPLAY: dict[str, str] = {
    normalize_condition(canonical): canonical for canonical in _RAW_DRUG_SYNONYM_GROUPS
}


def _summarise_job_state(stage: str | None, status: str | None, details: dict | None) -> str:
    if isinstance(details, dict):
        candidate = details.get("description") or details.get("details")
        if candidate:
            return str(candidate)

    stage_key = stage or status or "queued"
    summary = STAGE_SUMMARIES.get(stage_key)
    if summary:
        return summary
    if status and status in STAGE_SUMMARIES:
        return STAGE_SUMMARIES[status]
    return "Processing request."


def _build_pipeline_outline(stage: str | None, status: str | None) -> list[dict[str, str]]:
    if not PIPELINE_STEPS:
        return []

    if stage in PIPELINE_STAGE_INDEX:
        current_index = PIPELINE_STAGE_INDEX[stage]
    elif status in {"queued", "running"}:
        current_index = 0
    else:
        current_index = len(PIPELINE_STEPS)

    outline: list[dict[str, str]] = []
    for idx, step in enumerate(PIPELINE_STEPS):
        state = "upcoming"
        if idx < current_index:
            state = "completed"
        elif idx == current_index and current_index < len(PIPELINE_STEPS):
            state = "current"
        step_entry = dict(step)
        step_entry["state"] = state
        outline.append(step_entry)
    return outline


def _load_claim_set_by_signature(session, signature: str | None) -> ProcessedClaimSet | None:
    if not signature:
        return None
    stmt = select(ProcessedClaimSet).where(ProcessedClaimSet.mesh_signature == signature)
    return session.execute(stmt).scalar_one_or_none()


def _load_refresh_job(session, ref: str) -> ClaimSetRefresh | None:
    stmt = select(ClaimSetRefresh).where(ClaimSetRefresh.job_id == ref)
    refresh = session.execute(stmt).scalar_one_or_none()
    if refresh is None:
        stmt = select(ClaimSetRefresh).where(ClaimSetRefresh.mesh_signature == ref)
        refresh = session.execute(stmt).scalar_one_or_none()
    return refresh


def _parse_timestamp(value: str | datetime | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        timestamp = value
    else:
        try:
            timestamp = datetime.fromisoformat(value)
        except (TypeError, ValueError):  # noqa: PERF203 - defensive parsing
            return None
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return timestamp


def _extract_stored_search_signature(claim_set: ProcessedClaimSet | None) -> tuple[str | None, datetime | None]:
    if claim_set is None:
        return None, None
    active_version = claim_set.get_active_version()
    if active_version is None:
        return None, None
    metadata = active_version.pipeline_metadata if isinstance(active_version.pipeline_metadata, dict) else {}
    if not isinstance(metadata, dict):
        return None, None
    search_metadata = metadata.get("search_result") if isinstance(metadata.get("search_result"), dict) else None
    if not isinstance(search_metadata, dict):
        return None, None

    signature = search_metadata.get("signature")
    refreshed_at = _parse_timestamp(search_metadata.get("refreshed_at"))

    if signature:
        return str(signature), refreshed_at
    return None, refreshed_at


def _load_latest_search_signature(session, search_term_id: int | None) -> tuple[str | None, datetime | None]:
    if not search_term_id:
        return None, None
    artefact_stmt = (
        select(SearchArtefact)
        .where(SearchArtefact.search_term_id == search_term_id)
        .order_by(SearchArtefact.last_refreshed_at.desc(), SearchArtefact.created_at.desc())
        .limit(1)
    )
    artefact = session.execute(artefact_stmt).scalar_one_or_none()
    if artefact is None:
        return None, None
    signature = artefact.result_signature or artefact.mesh_signature
    refreshed_at = artefact.last_refreshed_at or artefact.created_at
    if refreshed_at is not None and refreshed_at.tzinfo is None:
        refreshed_at = refreshed_at.replace(tzinfo=timezone.utc)
    return signature, refreshed_at


def _build_refresh_prompt(session, claim_set: ProcessedClaimSet | None) -> dict | None:
    if claim_set is None:
        return None
    stored_signature, _ = _extract_stored_search_signature(claim_set)
    if not stored_signature:
        return None

    latest_signature, latest_refreshed_at = _load_latest_search_signature(
        session, getattr(claim_set, "last_search_term_id", None)
    )

    if not latest_signature or latest_signature == stored_signature:
        return None

    job_ref = getattr(claim_set, "mesh_signature", None)
    if not job_ref:
        return None

    return {
        "message": "New research may be available. Conduct another literature search?",
        "job_ref": job_ref,
        "refreshed_at": latest_refreshed_at,
    }


def _slugify_label(label: str) -> str:
    base = str(label or "").lower()
    base = re.sub(r"[^a-z0-9]+", "-", base)
    base = base.strip("-")
    return base or "entry"


def _classify_claim_severity(claim) -> tuple[str, dict[str, str]]:
    classification = str(getattr(claim, "classification", "") or "").lower()
    if classification == "safety":
        key = "safety"
    elif getattr(claim, "severe_reaction_flag", False):
        key = "severe"
    else:
        key = classification if classification in SEVERITY_META else "info"
    return key, SEVERITY_META.get(key, SEVERITY_META["info"])


def _canonicalize_drug_label(label: str) -> tuple[str, str]:
    normalized = normalize_condition(label)
    if normalized:
        for canonical_key, variants in DRUG_SYNONYM_MAP.items():
            if normalized in variants:
                display = DRUG_CANONICAL_DISPLAY.get(canonical_key) or label
                slug = _slugify_label(display)
                return slug, display
    slug = _slugify_label(label)
    return slug, label
def _build_claim_catalog(
    claim_set: ProcessedClaimSet | None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if claim_set is None:
        return [], []

    active_version = claim_set.get_active_version()
    if active_version is None:
        return [], []

    claims = list(getattr(active_version, "claims", []) or [])
    if not claims:
        return [], []

    claim_entries: list[dict[str, object]] = []
    drug_catalog: dict[str, dict[str, object]] = {}

    for index, claim in enumerate(claims, start=1):
        severity_key, severity_meta = _classify_claim_severity(claim)
        severity_rank = SEVERITY_ORDER.get(severity_key, len(SEVERITY_ORDER))
        classification_raw = str(getattr(claim, "classification", "") or "")
        classification_value = classification_raw.title()
        classification_lower = classification_raw.lower()
        confidence_value = str(getattr(claim, "confidence", "") or "").title()
        confidence_rank = CONFIDENCE_ORDER.get(confidence_value.lower(), len(CONFIDENCE_ORDER))

        drug_names = list(getattr(claim, "drugs", []) or [])
        if not drug_names:
            drug_names = ["Unspecified drug"]

        drug_slugs: list[str] = []
        for drug_name in drug_names:
            slug, display_name = _canonicalize_drug_label(drug_name)
            drug_slugs.append(slug)
            catalog_entry = drug_catalog.setdefault(
                slug,
                {
                    "name": display_name,
                    "slug": slug,
                    "claim_count": 0,
                    "severity": dict(severity_meta),
                    "severity_rank": severity_rank,
                    "aliases": set(),
                },
            )
            catalog_entry["claim_count"] += 1
            aliases = catalog_entry.get("aliases")
            if isinstance(aliases, set):
                aliases.add(drug_name)
            if severity_rank < catalog_entry["severity_rank"]:
                catalog_entry["severity_rank"] = severity_rank
                catalog_entry["severity"] = dict(severity_meta)

        drug_slugs = list(dict.fromkeys(drug_slugs))
        raw_severe_flag = bool(getattr(claim, "severe_reaction_flag", False))
        effective_severe_flag = raw_severe_flag and classification_lower != "safety"

        evidence_payload = []
        evidence_index: dict[tuple[str, str], dict[str, object]] = {}
        for evidence in getattr(claim, "evidence", []) or []:
            pmid = getattr(evidence, "pmid", None)
            citation_url = getattr(evidence, "citation_url", None)
            article_title = getattr(evidence, "article_title", None)
            snippet_id = getattr(evidence, "snippet_id", None)

            key = ("pmid", str(pmid)) if pmid else ("url", str(citation_url)) if citation_url else (
                "title",
                str(article_title),
            ) if article_title else ("snippet", str(snippet_id)) if snippet_id else ("index", str(len(evidence_index)))

            entry = evidence_index.get(key)
            if entry is None:
                entry = {
                    "pmid": pmid,
                    "article_title": article_title,
                    "citation_url": citation_url,
                    "key_points": [],
                    "_key_points_set": set(),
                    "notes": getattr(evidence, "notes", None),
                }
                evidence_index[key] = entry
            else:
                if entry.get("article_title") is None and article_title:
                    entry["article_title"] = article_title
                if entry.get("citation_url") is None and citation_url:
                    entry["citation_url"] = citation_url
                if entry.get("pmid") is None and pmid:
                    entry["pmid"] = pmid
                if not entry.get("notes") and getattr(evidence, "notes", None):
                    entry["notes"] = getattr(evidence, "notes", None)

            key_point_entries = list(getattr(evidence, "key_points", []) or [])
            point_registry: set[str] = entry["_key_points_set"]  # type: ignore[assignment]
            for point in key_point_entries:
                cleaned = str(point).strip()
                if not cleaned or cleaned in point_registry:
                    continue
                entry["key_points"].append(cleaned)
                point_registry.add(cleaned)

        for entry in evidence_index.values():
            entry.pop("_key_points_set", None)
            evidence_payload.append(entry)

        claim_entries.append(
            {
                "anchor_id": f"claim-{getattr(claim, 'id', index)}",
                "summary": getattr(claim, "summary", ""),
                "classification": classification_value,
                "confidence": confidence_value,
                "confidence_rank": confidence_rank,
                "severity": dict(severity_meta),
                "severity_key": severity_key,
                "drugs": drug_names,
                "drug_slugs": drug_slugs,
                "drug_classes": list(getattr(claim, "drug_classes", []) or []),
                "severe_flag": effective_severe_flag,
                "severe_terms": list(getattr(claim, "severe_reaction_terms", []) or []),
                "evidence": evidence_payload,
            }
        )

    claim_entries.sort(
        key=lambda entry: (
            SEVERITY_ORDER.get(entry["severity_key"], len(SEVERITY_ORDER)),
            entry["confidence_rank"],
        )
    )

    drug_filters = list(drug_catalog.values())
    for entry in drug_filters:
        aliases = entry.get("aliases")
        if isinstance(aliases, set):
            entry["aliases"] = sorted(aliases, key=str.lower)
    drug_filters.sort(key=lambda entry: (entry["severity_rank"], entry["name"].lower()))

    return claim_entries, drug_filters


def _serialise_timestamp(timestamp: datetime | None) -> datetime | None:
    if timestamp is None:
        return None
    if not isinstance(timestamp, datetime):
        return None
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc)


def _format_job_payload(job: object | None) -> dict | None:
    if job is None:
        return None

    if isinstance(job, dict):
        details = job.get("details")
        if not isinstance(details, dict):
            details = {}
        status = str(job.get("status", "queued"))
        stage = job.get("stage")
        summary = _summarise_job_state(stage, status, details)
        return {
            "job_id": job.get("job_id"),
            "status": status,
            "stage": stage,
            "details": details,
            "created_at": _serialise_timestamp(job.get("created_at")),
            "updated_at": _serialise_timestamp(job.get("updated_at")),
            "summary": summary,
        }

    details = {}
    payload = getattr(job, "progress_payload", None)
    if isinstance(payload, dict):
        details = payload

    status = str(getattr(job, "status", "queued"))
    stage = getattr(job, "progress_state", None)

    return {
        "job_id": getattr(job, "job_id", None),
        "status": status,
        "stage": stage,
        "details": details,
        "created_at": _serialise_timestamp(getattr(job, "created_at", None)),
        "updated_at": _serialise_timestamp(getattr(job, "updated_at", None)),
        "summary": _summarise_job_state(stage, status, details),
    }


def _derive_condition_label(claim_set: ProcessedClaimSet | None, mesh_signature: str | None) -> str:
    if claim_set and claim_set.condition_label:
        return claim_set.condition_label
    if not mesh_signature:
        return "Current request"
    terms = [part.strip() for part in mesh_signature.split("|") if part.strip()]
    if not terms:
        return "Current request"
    return " / ".join(term.title() for term in terms)


def _sanitize_mesh_terms(values: Sequence[object] | None) -> list[str]:
    if not values:
        return []
    if isinstance(values, (str, bytes)):
        values = [values]
    cleaned: list[str] = []
    seen: set[str] = set()
    for candidate in values:
        text = str(candidate).strip()
        if not text:
            continue
        if text.lower() in seen:
            continue
        cleaned.append(text)
        seen.add(text.lower())
    return cleaned


def _extract_mesh_terms_from_resolution(resolution: object | None) -> list[str]:
    if resolution is None:
        return []
    terms = getattr(resolution, "mesh_terms", None)
    if terms is None:
        return []
    return _sanitize_mesh_terms(terms)


def _extract_mesh_terms_from_job(job_payload: dict | None) -> list[str]:
    if not isinstance(job_payload, dict):
        return []
    details = job_payload.get("details") if isinstance(job_payload.get("details"), dict) else {}
    resolution_details = details.get("resolution") if isinstance(details, dict) else {}
    if isinstance(resolution_details, dict):
        terms = _sanitize_mesh_terms(resolution_details.get("mesh_terms", []))
        if terms:
            return terms
    if isinstance(details, dict):
        detail_terms = details.get("mesh_terms")
        if isinstance(detail_terms, Sequence) and not isinstance(detail_terms, (str, bytes)):
            terms = _sanitize_mesh_terms(detail_terms)
            if terms:
                return terms
    fallback_terms = job_payload.get("mesh_terms")
    if isinstance(fallback_terms, Sequence) and not isinstance(fallback_terms, (str, bytes)):
        return _sanitize_mesh_terms(fallback_terms)
    if isinstance(fallback_terms, (str, bytes)):
        return _sanitize_mesh_terms([fallback_terms])
    return []


def _determine_mesh_terms(
    claim_set: ProcessedClaimSet | None,
    resolution: object | None = None,
    job_payload: dict | None = None,
    mesh_signature: str | None = None,
) -> list[str]:
    terms = _extract_mesh_terms_from_resolution(resolution)
    if terms:
        return terms

    terms = _extract_mesh_terms_from_job(job_payload)
    if terms:
        return terms

    if claim_set is not None:
        signature_terms = _mesh_terms_from_signature(getattr(claim_set, "mesh_signature", None))
        signature_terms = _sanitize_mesh_terms(signature_terms)
        if signature_terms:
            return signature_terms

    if mesh_signature:
        return _sanitize_mesh_terms(_mesh_terms_from_signature(mesh_signature))

    return []


def _record_refresh_stub(mesh_signature: str | None, job_payload: dict | None) -> None:
    if not mesh_signature or not job_payload:
        return

    job_id = str(job_payload.get("job_id") or "")
    status = str(job_payload.get("status") or "queued")
    stage = job_payload.get("stage") or "queued"
    details = job_payload.get("details") if isinstance(job_payload.get("details"), dict) else {}

    factory = current_app.extensions.get("session_factory")
    if factory is None:
        return

    with closing(factory()) as stub_session:
        existing = stub_session.execute(
            select(ClaimSetRefresh).where(ClaimSetRefresh.mesh_signature == mesh_signature)
        ).scalar_one_or_none()

        if existing is None:
            refresh = ClaimSetRefresh(
                mesh_signature=mesh_signature,
                job_id=job_id,
                status=status,
                progress_state=stage,
                progress_payload=details,
            )
            stub_session.add(refresh)
        else:
            existing.job_id = job_id or existing.job_id
            existing.status = status
            existing.progress_state = stage
            existing.progress_payload = details
            existing.error_message = None
        stub_session.commit()


def _render_claims_response(context: dict, status_code: int = 200):
    payload = dict(context)
    payload.setdefault("resolved_mesh_terms", [])
    is_htmx = bool(request.headers.get("HX-Request"))
    is_boosted = bool(request.headers.get("HX-Boosted"))
    target = (request.headers.get("HX-Target") or "").strip()
    prefers_fragment = target and target not in {"main", "document.body", "body"}
    if is_htmx and (not is_boosted or prefers_fragment):
        return render_template("ui/_claims_table.html", **payload), status_code

    payload.setdefault("completed", False)
    payload.setdefault("resolution", None)
    payload.setdefault("claims_url", None)
    return render_template("ui/claims.html", **payload), status_code


def _render_progress_response(context: dict, status_code: int = 200):
    payload = dict(context)
    payload.setdefault("resolved_mesh_terms", [])
    is_htmx = bool(request.headers.get("HX-Request"))
    is_boosted = bool(request.headers.get("HX-Boosted"))
    target = (request.headers.get("HX-Target") or "").strip()
    prefers_fragment = target and target not in {"main", "document.body", "body"}
    if is_htmx and (not is_boosted or prefers_fragment):
        return render_template("ui/_progress_panel.html", **payload), status_code
    return render_template("ui/status.html", **payload), status_code


@ui_blueprint.get("/")
def index() -> str:
    return render_template("ui/index.html")


@ui_blueprint.get("/runs")
def runs() -> str:
    session = _get_db_session()

    stmt = (
        select(ClaimSetRefresh)
        .order_by(ClaimSetRefresh.updated_at.desc(), ClaimSetRefresh.created_at.desc())
        .limit(50)
    )
    refreshes = session.execute(stmt).scalars().all()

    entries: list[dict[str, object]] = []
    for refresh in refreshes:
        payload = _format_job_payload(refresh) or {}
        entry = {
            "job_id": payload.get("job_id") or getattr(refresh, "job_id", None),
            "mesh_signature": getattr(refresh, "mesh_signature", None),
            "status": payload.get("status") or getattr(refresh, "status", "queued"),
            "details": payload.get("details") or (getattr(refresh, "progress_payload", {}) or {}),
            "updated_at": payload.get("updated_at")
            or _serialise_timestamp(getattr(refresh, "updated_at", None))
            or _serialise_timestamp(getattr(refresh, "created_at", None)),
        }
        entries.append(entry)

    return render_template("ui/runs.html", runs=entries)


@ui_blueprint.post("/search")
def search() -> tuple[str, int]:
    session = _get_db_session()
    condition = (request.form.get("condition") or "").strip()

    if not condition:
        return render_template("ui/_error_message.html", message="Condition is required"), 400

    cached_resolution = _load_cached_resolution(session, condition)
    if cached_resolution is not None:
        mesh_signature = cached_resolution.mesh_signature or compute_mesh_signature(list(cached_resolution.mesh_terms))
        claim_set = _load_claim_set_by_signature(session, mesh_signature)
        if claim_set is not None and claim_set.get_active_version() is not None:
            claim_entries, drug_filters = _build_claim_catalog(claim_set)
            resolved_terms = _determine_mesh_terms(
                claim_set,
                resolution=cached_resolution,
                mesh_signature=mesh_signature,
            )
            context = {
                "claim_set": claim_set,
                "resolution": cached_resolution,
                "completed": True,
                "claims_url": url_for("ui.view_claims", claim_set_id=claim_set.id),
                "refresh_prompt": _build_refresh_prompt(session, claim_set),
                "claim_entries": claim_entries,
                "drug_filters": drug_filters,
                "resolved_mesh_terms": resolved_terms,
            }
            return _render_claims_response(context)

    preview = preview_mesh_resolution(condition)
    if preview.status == "needs_clarification":
        return render_template("ui/_mesh_selector.html", condition=condition, preview=preview), 200
    if preview.status == "not_found":
        message = "No MeSH terms matched the supplied condition. Please refine your query."
        return render_template("ui/_error_message.html", message=message), 422

    try:
        resolution = resolve_condition_via_nih(condition, session=session)
    except MeshTermsNotFoundError as exc:
        current_app.logger.info(
            "Mesh resolution failed during UI search",
            extra={"condition": condition, "normalized": exc.normalized_condition},
        )
        message = "Unable to resolve MeSH terms automatically. Please choose from the suggestions."
        options = MeshResolutionPreview(
            status="needs_clarification",
            raw_query=condition,
            normalized_query=exc.normalized_condition,
            mesh_terms=[],
            ranked_options=list(exc.suggestions),
            suggestions=list(exc.suggestions),
            espell_correction=None,
        )
        return render_template("ui/_mesh_selector.html", condition=condition, preview=options), 200

    mesh_signature = compute_mesh_signature(list(resolution.mesh_terms))
    job_info = enqueue_claim_pipeline(
        session=session,
        resolution=resolution,
        condition_label=condition,
        mesh_signature=mesh_signature,
    )

    job_payload = _format_job_payload(job_info)
    _record_refresh_stub(mesh_signature, job_payload)

    context = {
        "condition": condition,
        "job": job_payload,
        "pipeline_outline": _build_pipeline_outline(
            job_payload.get("stage") if job_payload else None,
            job_payload.get("status") if job_payload else None,
        ),
        "terminal_statuses": TERMINAL_STATUSES,
        "can_retry": False,
        "notice": None,
        "notice_variant": "info",
        "resolved_mesh_terms": _determine_mesh_terms(
            None,
            resolution=resolution,
            job_payload=job_payload,
            mesh_signature=mesh_signature,
        ),
    }
    return _render_progress_response(context)


@ui_blueprint.post("/mesh-select")
def mesh_select() -> tuple[str, int]:
    session = _get_db_session()
    condition = (request.form.get("condition") or "").strip()
    selected_terms = [value.strip() for value in request.form.getlist("mesh_term") if value and value.strip()]

    if not condition:
        return render_template("ui/_error_message.html", message="Condition is required"), 400
    if not selected_terms:
        return render_template("ui/_error_message.html", message="Select a MeSH term to continue."), 400

    manual_builder = _manual_mesh_builder(selected_terms)

    try:
        resolution = resolve_condition_via_nih(condition, session=session, mesh_builder=manual_builder)
    except MeshTermsNotFoundError:
        message = "Unable to resolve the selected MeSH terms. Please try a different option."
        return render_template("ui/_error_message.html", message=message), 422

    mesh_signature = compute_mesh_signature(list(resolution.mesh_terms))
    job_info = enqueue_claim_pipeline(
        session=session,
        resolution=resolution,
        condition_label=condition,
        mesh_signature=mesh_signature,
    )

    job_payload = _format_job_payload(job_info)
    _record_refresh_stub(mesh_signature, job_payload)

    context = {
        "condition": condition,
        "job": job_payload,
        "pipeline_outline": _build_pipeline_outline(
            job_payload.get("stage") if job_payload else None,
            job_payload.get("status") if job_payload else None,
        ),
        "terminal_statuses": TERMINAL_STATUSES,
        "can_retry": False,
        "notice": None,
        "notice_variant": "info",
        "resolved_mesh_terms": _determine_mesh_terms(
            None,
            resolution=resolution,
            job_payload=job_payload,
            mesh_signature=mesh_signature,
        ),
    }
    return _render_progress_response(context)


@ui_blueprint.get("/status/<job_ref>")
def status(job_ref: str) -> tuple[str, int]:
    session = _get_db_session()
    refresh = _load_refresh_job(session, job_ref)
    condition_hint = request.args.get("condition") or "Current request"

    if refresh is None:
        claim_set = _load_claim_set_by_signature(session, job_ref)
        if claim_set is not None and claim_set.get_active_version() is not None:
            claim_entries, drug_filters = _build_claim_catalog(claim_set)
            resolved_terms = _determine_mesh_terms(
                claim_set,
                mesh_signature=getattr(claim_set, "mesh_signature", None),
            )
            context = {
                "claim_set": claim_set,
                "resolution": None,
                "completed": True,
                "claims_url": url_for("ui.view_claims", claim_set_id=claim_set.id),
                "refresh_prompt": _build_refresh_prompt(session, claim_set),
                "claim_entries": claim_entries,
                "drug_filters": drug_filters,
                "resolved_mesh_terms": resolved_terms,
            }
            return _render_claims_response(context)
        placeholder = {
            "job_id": job_ref,
            "status": "queued",
            "stage": "queued",
            "details": {"description": STAGE_SUMMARIES.get("queued")},
            "created_at": None,
            "updated_at": None,
            "summary": STAGE_SUMMARIES.get("queued"),
        }
        context = {
            "condition": condition_hint,
            "job": placeholder,
            "pipeline_outline": _build_pipeline_outline("queued", "queued"),
            "terminal_statuses": TERMINAL_STATUSES,
            "notice": None,
            "notice_variant": "info",
            "can_retry": False,
        }
        return _render_progress_response(context)

    claim_set = _load_claim_set_by_signature(session, refresh.mesh_signature)
    if claim_set is not None and claim_set.get_active_version() is not None and refresh.status == "completed":
        claim_entries, drug_filters = _build_claim_catalog(claim_set)
        resolved_terms = _determine_mesh_terms(
            claim_set,
            mesh_signature=getattr(claim_set, "mesh_signature", None),
        )
        context = {
            "claim_set": claim_set,
            "resolution": None,
            "completed": True,
            "claims_url": url_for("ui.view_claims", claim_set_id=claim_set.id),
            "refresh_prompt": _build_refresh_prompt(session, claim_set),
            "claim_entries": claim_entries,
            "drug_filters": drug_filters,
            "resolved_mesh_terms": resolved_terms,
        }
        return _render_claims_response(context)

    job_payload = _format_job_payload(refresh)
    condition_label = request.args.get("condition") or _derive_condition_label(
        claim_set, getattr(refresh, "mesh_signature", None)
    )
    resolved_terms = _determine_mesh_terms(
        claim_set,
        job_payload=job_payload,
        mesh_signature=getattr(refresh, "mesh_signature", None),
    )

    notice = None
    notice_variant = "info"
    status = job_payload.get("status") if job_payload else None
    can_retry = False
    if status == "empty-results":
        notice = "No claims were generated for this condition."
        notice_variant = "warning"
        can_retry = _refresh_empty_can_retry(refresh) or _claim_set_empty_can_retry(claim_set)
    elif status == "failed":
        error_detail = None
        if job_payload:
            error_detail = job_payload.get("details", {}).get("error")
        if not error_detail:
            error_detail = getattr(refresh, "error_message", None)
        notice = error_detail or "The refresh failed. Please try again later."
        notice_variant = "danger"
        can_retry = _refresh_can_retry(refresh)
    elif status == "no-batches":
        detail_terms = job_payload.get("details", {}).get("mesh_terms", []) if job_payload else []
        if detail_terms:
            notice = (
                "We could not find PubMed snippets for the resolved MeSH term(s): "
                + ", ".join(detail_terms)
                + ". Try refining the condition or selecting different terms."
            )
        else:
            notice = "We could not find PubMed snippets for the resolved MeSH terms. Try refining the condition."
        notice_variant = "warning"
        can_retry = _refresh_can_retry(refresh)
    elif status == "completed" and claim_set is None:
        notice = "Processing completed but no claim set is available yet."

    context = {
        "condition": condition_label,
        "job": job_payload,
        "notice": notice,
        "notice_variant": notice_variant,
        "pipeline_outline": _build_pipeline_outline(
            job_payload.get("stage") if job_payload else None,
            job_payload.get("status") if job_payload else None,
        ),
        "terminal_statuses": TERMINAL_STATUSES,
        "can_retry": can_retry,
        "resolved_mesh_terms": resolved_terms,
    }
    return _render_progress_response(context)


def _extract_retry_inputs(job_payload: dict | None, refresh: ClaimSetRefresh) -> tuple[str | None, list[str]]:
    if not job_payload:
        job_payload = {}
    details = job_payload.get("details") if isinstance(job_payload, dict) else {}
    resolution = details.get("resolution") if isinstance(details, dict) else None

    normalized_condition = None
    mesh_terms: list[str] = []

    if isinstance(resolution, dict):
        normalized_condition = resolution.get("normalized_condition")
        mesh_terms = [term for term in resolution.get("mesh_terms", []) if term]

    if not mesh_terms:
        mesh_terms = _mesh_terms_from_signature(getattr(refresh, "mesh_signature", None))

    return normalized_condition, mesh_terms


@ui_blueprint.post("/retry/<job_ref>")
def retry(job_ref: str) -> tuple[str, int]:
    session = _get_db_session()
    refresh = _load_refresh_job(session, job_ref)
    if refresh is None:
        return render_template("ui/_error_message.html", message="Refresh job not found"), 404

    claim_set = _load_claim_set_by_signature(session, getattr(refresh, "mesh_signature", None))
    condition_label = _derive_condition_label(claim_set, getattr(refresh, "mesh_signature", None))
    job_payload = _format_job_payload(refresh)
    normalized_condition, mesh_terms = _extract_retry_inputs(job_payload, refresh)

    condition_for_retry = normalized_condition or condition_label

    manual_builder = None
    if mesh_terms:
        manual_builder = _manual_mesh_builder(mesh_terms)

    try:
        resolution = resolve_condition_via_nih(
            condition_for_retry,
            session=session,
            mesh_builder=manual_builder,
        )
    except MeshTermsNotFoundError:
        message = "Unable to retry automatically. Please start a new search."
        return render_template("ui/_error_message.html", message=message), 422

    mesh_signature = compute_mesh_signature(list(resolution.mesh_terms))
    job_info = enqueue_claim_pipeline(
        session=session,
        resolution=resolution,
        condition_label=condition_label,
        mesh_signature=mesh_signature,
    )

    new_job_payload = _format_job_payload(job_info)
    _record_refresh_stub(mesh_signature, new_job_payload)

    return render_template(
        "ui/_progress_panel.html",
        condition=condition_label,
        job=new_job_payload,
        can_retry=False,
        notice="Retry requested. Tracking latest refresh.",
        notice_variant="info",
        pipeline_outline=_build_pipeline_outline(
            new_job_payload.get("stage") if new_job_payload else None,
            new_job_payload.get("status") if new_job_payload else None,
        ),
        terminal_statuses=TERMINAL_STATUSES,
    ), 200


@ui_blueprint.get("/claims/<int:claim_set_id>")
def view_claims(claim_set_id: int) -> tuple[str, int]:
    session = _get_db_session()
    claim_set = session.get(ProcessedClaimSet, claim_set_id)
    if claim_set is None or claim_set.get_active_version() is None:
        return render_template("ui/_error_message.html", message="Claim set not found"), 404

    claim_entries, drug_filters = _build_claim_catalog(claim_set)
    resolved_terms = _determine_mesh_terms(
        claim_set,
        mesh_signature=getattr(claim_set, "mesh_signature", None),
    )

    return render_template(
        "ui/claims.html",
        claim_set=claim_set,
        refresh_prompt=_build_refresh_prompt(session, claim_set),
        claim_entries=claim_entries,
        drug_filters=drug_filters,
        resolved_mesh_terms=resolved_terms,
    ), 200
