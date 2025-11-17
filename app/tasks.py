from __future__ import annotations

import logging
from contextlib import closing
from time import sleep
from types import SimpleNamespace
from typing import Callable, Sequence

from sqlalchemy import select, update
from sqlalchemy.orm import Session

# Module-level logger for debugging status propagation from Celery workers.
logger = logging.getLogger(__name__)

PROGRESS_DESCRIPTIONS: dict[str, str] = {
    "queued": "Waiting for a worker to begin processing the refresh job.",
    "resolving_mesh_terms": "Resolving the condition against NIH services to determine authoritative MeSH terms.",
    "fetching_pubmed_articles": "Fetching PubMed articles and any available full-text content for the condition.",
    "preparing_llm_batches": "Preparing snippet batches that will be sent to the language model.",
    "generating_claims": "Calling the OpenAI API to generate refreshed claims from the gathered snippets.",
    "saving_processed_claims": "Persisting generated claims and associated metadata in the database.",
    "no_batches": "No eligible snippets were available to build language-model batches.",
    "no_responses": "The language model returned no usable responses for this refresh.",
    "empty_results": "Processing completed, but no claims were produced for this condition.",
    "completed": "Claim refresh completed successfully.",
    "skipped": "Refresh skipped because required condition data could not be retrieved.",
    "failed": "Refresh failed due to an unexpected error during processing.",
}

LLM_BATCH_AVERAGE_SECONDS = 150
LLM_BATCH_AVERAGE_TOKENS = 30_000

from .database import create_session_factory
from .models import ClaimSetRefresh
from .services.full_text import FullTextSelectionPolicy, NIHFullTextFetcher, collect_pubmed_articles
from .services.llm_batches import build_llm_batches
from .services.nih_pipeline import MeshTermsNotFoundError, resolve_condition_via_nih
from .services.nih_pubmed import NIHPubMedSearcher
from .services.openai_client import OpenAIChatClient
from .services.processed_claims import persist_processed_claims
from .services.search import SearchResolution, compute_mesh_signature
from .settings import load_settings


def ping() -> str:
    """Simple health check used by background workers."""
    sleep(0.1)
    return "pong"


from .celery_app import celery  # noqa: E402  (delayed to avoid circular import during module load)


def refresh_claims_for_condition(
    *,
    resolution_id: int,
    condition_label: str,
    normalized_condition: str,
    mesh_terms: Sequence[str],
    mesh_signature: str | None,
    job_id: str | None = None,
    session_factory: Callable[[], Session] | None = None,
) -> str:
    """Execute the full ingest pipeline for a cached condition search."""
    factory = session_factory or create_session_factory()
    session = factory()
    refresh_job: SimpleNamespace | None = None

    try:
        settings = load_settings()
        article_defaults = settings.article_selection
        pubmed_searcher = NIHPubMedSearcher(retmax=article_defaults.pubmed_retmax)
        selection_policy = FullTextSelectionPolicy(
            base_full_text=article_defaults.base_full_text_articles,
            max_full_text=article_defaults.max_full_text_articles,
        )
        full_text_fetcher = NIHFullTextFetcher()

        def _ensure_refresh_job() -> SimpleNamespace | None:
            nonlocal refresh_job
            if refresh_job is not None:
                return refresh_job
            if not job_id and not mesh_signature:
                return None

            max_attempts = 10
            attempts = 0
            while attempts < max_attempts and refresh_job is None:
                candidate: ClaimSetRefresh | None = None
                if job_id:
                    candidate = session.execute(
                        select(ClaimSetRefresh).where(ClaimSetRefresh.job_id == job_id)
                    ).scalar_one_or_none()
                if candidate is None and mesh_signature:
                    candidate = session.execute(
                        select(ClaimSetRefresh).where(
                            ClaimSetRefresh.mesh_signature == mesh_signature
                        )
                    ).scalar_one_or_none()
                    if candidate is not None and job_id and candidate.job_id != job_id:
                        _commit_refresh_changes(
                            SimpleNamespace(id=candidate.id),
                            factory,
                            job_id=job_id,
                        )
                        candidate.job_id = job_id

                if candidate is not None:
                    refresh_job = SimpleNamespace(
                        id=candidate.id,
                        job_id=candidate.job_id,
                        status=candidate.status,
                        error_message=candidate.error_message,
                        mesh_signature=candidate.mesh_signature,
                        progress_state=candidate.progress_state,
                        progress_payload=candidate.progress_payload,
                    )
                    if refresh_job.status == "queued":
                        refresh_job.status = "running"
                        refresh_job.error_message = None
                        logger.debug(
                            "Marked refresh job as running",
                            extra={
                                "job_id": job_id,
                                "mesh_signature": mesh_signature,
                                "refresh_id": refresh_job.id,
                            },
                        )
                    return refresh_job

                attempts += 1
                if attempts < max_attempts:
                    sleep(0.2)

            if refresh_job is None:
                logger.debug(
                    "Refresh job lookup still pending",
                    extra={
                        "job_id": job_id,
                        "mesh_signature": mesh_signature,
                        "attempts": attempts,
                    },
                )
            return refresh_job

        job_record = _ensure_refresh_job()
        if job_record is not None:
            job_record.status = "running"
            job_record.error_message = None
            _update_refresh_progress(
                job_record,
                stage="resolving_mesh_terms",
                session_factory=factory,
            )

        try:
            fresh_resolution = resolve_condition_via_nih(
                condition_label,
                session=session,
                pubmed_searcher=pubmed_searcher,
                refresh_ttl_seconds=settings.search.refresh_ttl_seconds,
            )
        except MeshTermsNotFoundError as exc:
            job_record = _ensure_refresh_job()
            if job_record is not None:
                job_record.status = "skipped"
                job_record.error_message = None
                _update_refresh_progress(
                    job_record,
                    stage="skipped",
                    details={
                        "reason": "missing_mesh_terms",
                        "suggestions": list(exc.suggestions),
                    },
                    session_factory=factory,
                )
                session.commit()
            else:
                session.commit()
            return "skipped"

        resolved_mesh_terms = list(fresh_resolution.mesh_terms)
        fresh_mesh_signature = compute_mesh_signature(resolved_mesh_terms)

        if not fresh_mesh_signature:
            job_record = _ensure_refresh_job()
            if job_record is not None:
                job_record.status = "skipped"
                job_record.error_message = None
                _update_refresh_progress(
                    job_record,
                    stage="skipped",
                    details={"reason": "missing_mesh_signature"},
                    session_factory=factory,
                )
            session.commit()
            return "skipped"

        job_record = _ensure_refresh_job()
        if job_record is not None and job_record.mesh_signature != fresh_mesh_signature:
            job_record.mesh_signature = fresh_mesh_signature
            _commit_refresh_changes(
                job_record,
                factory,
                mesh_signature=fresh_mesh_signature,
            )

        job_record = _ensure_refresh_job()
        if job_record is not None:
            _update_refresh_progress(
                job_record,
                stage="fetching_pubmed_articles",
                details={"search_term_id": fresh_resolution.search_term_id},
                session_factory=factory,
            )

        collect_pubmed_articles(
            fresh_resolution,
            session=session,
            pubmed_searcher=pubmed_searcher,
            full_text_fetcher=full_text_fetcher,
            selection_policy=selection_policy,
        )
        session.flush()

        job_record = _ensure_refresh_job()
        if job_record is not None:
            _update_refresh_progress(
                job_record,
                stage="preparing_llm_batches",
                details={"search_term_id": fresh_resolution.search_term_id},
                session_factory=factory,
            )

        batches = build_llm_batches(
            session,
            search_term_id=fresh_resolution.search_term_id,
            condition_label=condition_label,
            mesh_terms=fresh_resolution.mesh_terms,
        )
        if not batches:
            job_record = _ensure_refresh_job()
            if job_record is not None:
                job_record.status = "no-batches"
                job_record.error_message = None
                _update_refresh_progress(
                    job_record,
                    stage="no_batches",
                    details={
                        "reason": "no_llm_batches",
                        "description": "No PubMed snippets were found for the resolved MeSH terms.",
                        "mesh_terms": list(fresh_resolution.mesh_terms),
                        "normalized_condition": fresh_resolution.normalized_condition,
                        "search_term_id": fresh_resolution.search_term_id,
                    },
                    session_factory=factory,
                )
            session.commit()
            return "no-batches"

        job_record = _ensure_refresh_job()
        if job_record is not None:
            _update_refresh_progress(
                job_record,
                stage="generating_claims",
                details=_build_batch_progress_details(len(batches), 0),
                session_factory=factory,
            )

        client = OpenAIChatClient()

        def _on_batch_progress(completed: int, total: int, _batch: object | None = None) -> None:
            job = _ensure_refresh_job()
            if job is None:
                return
            details = _build_batch_progress_details(total, completed)
            _update_refresh_progress(
                job,
                stage="generating_claims",
                details=details,
                session_factory=factory,
            )

        responses = client.run_batches(
            batches,
            progress_callback=_on_batch_progress,
        )
        if not responses:
            job_record = _ensure_refresh_job()
            if job_record is not None:
                job_record.status = "no-responses"
                job_record.error_message = None
                _update_refresh_progress(
                    job_record,
                    stage="no_responses",
                    details={"reason": "llm_returned_no_responses"},
                    session_factory=factory,
                )
            session.commit()
            return "no-responses"

        job_record = _ensure_refresh_job()
        if job_record is not None:
            _update_refresh_progress(
                job_record,
                stage="saving_processed_claims",
                details={"response_count": len(responses)},
                session_factory=factory,
            )

        claim_set = persist_processed_claims(
            session,
            search_term_id=fresh_resolution.search_term_id,
            mesh_signature=fresh_mesh_signature,
            condition_label=condition_label,
            llm_payloads=responses,
            search_result_signature=fresh_resolution.result_signature,
            search_result_refreshed_at=fresh_resolution.artefact_refreshed_at,
        )
        if hasattr(claim_set, "get_active_claims"):
            claim_count = len(claim_set.get_active_claims())
        elif getattr(claim_set, "claims", None) is not None:
            claim_count = len(claim_set.claims)
        else:
            claim_count = 0

        job_record = _ensure_refresh_job()
        if job_record is not None:
            job_record.error_message = None
            if claim_count == 0:
                job_record.status = "empty-results"
                _update_refresh_progress(
                    job_record,
                    stage="empty_results",
                    details={
                        "response_count": len(responses),
                        "claim_count": claim_count,
                    },
                    session_factory=factory,
                )
            else:
                job_record.status = "completed"
                _update_refresh_progress(
                    job_record,
                    stage="completed",
                    details={
                        "response_count": len(responses),
                        "claim_count": claim_count,
                    },
                    session_factory=factory,
                )

        session.commit()
        return "empty-results" if claim_count == 0 else "completed"
    except Exception as exc:  # noqa: BLE001
        session.rollback()
        if job_id:
            refreshed_job = session.execute(
                select(ClaimSetRefresh).where(ClaimSetRefresh.job_id == job_id)
            ).scalar_one_or_none()
            if refreshed_job is not None:
                refreshed_job.status = "failed"
                refreshed_job.error_message = str(exc)
                _update_refresh_progress(
                    refreshed_job,
                    stage="failed",
                    details={"error": str(exc)},
                    session_factory=factory,
                )
                session.commit()
            else:
                session.rollback()
        raise
    finally:
        session.close()


def _commit_refresh_changes(
    refresh_job: SimpleNamespace | ClaimSetRefresh | None,
    session_factory: Callable[[], Session],
    **values,
) -> None:
    if refresh_job is None:
        return
    with closing(session_factory()) as progress_session:
        persistent = progress_session.get(ClaimSetRefresh, refresh_job.id)
        if persistent is None:
            logger.warning(
                "Refresh progress update skipped; record not found",
                extra={"refresh_id": refresh_job.id, **values},
            )
            return
        for key, value in values.items():
            setattr(persistent, key, value)
        progress_session.commit()

    for key, value in values.items():
        setattr(refresh_job, key, value)


def _build_batch_progress_details(total_batches: int, completed_batches: int) -> dict[str, int]:
    total = max(int(total_batches or 0), 0)
    completed = max(int(completed_batches or 0), 0)
    if total > 0:
        completed = min(completed, total)
    remaining = max(total - completed, 0) if total else 0

    return {
        "batch_count": total,
        "batches_completed": completed,
        "average_batch_seconds": LLM_BATCH_AVERAGE_SECONDS,
        "average_batch_tokens": LLM_BATCH_AVERAGE_TOKENS,
        "estimated_total_seconds": total * LLM_BATCH_AVERAGE_SECONDS,
        "estimated_total_tokens": total * LLM_BATCH_AVERAGE_TOKENS,
        "estimated_remaining_seconds": remaining * LLM_BATCH_AVERAGE_SECONDS,
        "estimated_remaining_tokens": remaining * LLM_BATCH_AVERAGE_TOKENS,
    }


def _update_refresh_progress(
    refresh_job: SimpleNamespace | ClaimSetRefresh | None,
    *,
    stage: str,
    details: dict | None = None,
    session_factory: Callable[[], Session] | None = None,
) -> None:
    if refresh_job is None:
        return
    payload = dict(details or {})
    description = PROGRESS_DESCRIPTIONS.get(stage)
    if description and "description" not in payload:
        payload["description"] = description

    refresh_job.progress_state = stage
    refresh_job.progress_payload = payload
    if refresh_job.status == "queued":
        refresh_job.status = "running"

    if session_factory is None:
        raise ValueError("session_factory is required to persist progress updates")

    _commit_refresh_changes(
        refresh_job,
        session_factory,
        status=refresh_job.status,
        progress_state=refresh_job.progress_state,
        progress_payload=refresh_job.progress_payload,
        error_message=refresh_job.error_message,
    )


@celery.task(name="app.tasks.ping")
def ping_task() -> str:
    """Celery entrypoint delegating to the synchronous ping helper."""
    return ping()


@celery.task(name="app.tasks.refresh_claims_for_condition", bind=True)
def refresh_claims_for_condition_task(self, **kwargs) -> str:
    """Celery entrypoint forwarding execution to the synchronous pipeline."""
    job_id = getattr(self.request, "id", None)
    return refresh_claims_for_condition(job_id=job_id, **kwargs)

