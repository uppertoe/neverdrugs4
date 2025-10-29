from __future__ import annotations

from time import sleep
from typing import Sequence

from sqlalchemy import select

from .celery_app import celery
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


@celery.task(bind=True)
def ping(self) -> str:
    """Simple task to verify Celery worker wiring."""
    sleep(0.1)
    return "pong"


@celery.task(name="app.tasks.refresh_claims_for_condition", bind=True)
def refresh_claims_for_condition(
    self,
    *,
    resolution_id: int,
    condition_label: str,
    normalized_condition: str,
    mesh_terms: Sequence[str],
    mesh_signature: str | None,
) -> str:
    """Background refresh for a cached condition search."""
    session_factory = create_session_factory()
    session = session_factory()
    refresh_job: ClaimSetRefresh | None = None

    try:
        settings = load_settings()
        article_defaults = settings.article_selection
        pubmed_searcher = NIHPubMedSearcher(retmax=article_defaults.pubmed_retmax)
        selection_policy = FullTextSelectionPolicy(
            base_full_text=article_defaults.base_full_text_articles,
            max_full_text=article_defaults.max_full_text_articles,
        )
        full_text_fetcher = NIHFullTextFetcher()

        job_id = getattr(self.request, "id", None)
        if job_id:
            refresh_job = session.execute(
                select(ClaimSetRefresh).where(ClaimSetRefresh.job_id == job_id)
            ).scalar_one_or_none()
            if refresh_job is not None:
                refresh_job.status = "running"
                refresh_job.error_message = None
                _update_refresh_progress(session, refresh_job, stage="resolving_condition")

        resolution = SearchResolution(
            normalized_condition=normalized_condition,
            mesh_terms=list(mesh_terms),
            reused_cached=False,
            search_term_id=resolution_id,
        )

        actual_mesh_signature = mesh_signature or compute_mesh_signature(list(mesh_terms))
        if not actual_mesh_signature:
            if refresh_job is not None:
                refresh_job.status = "skipped"
                refresh_job.error_message = None
                _update_refresh_progress(
                    session,
                    refresh_job,
                    stage="skipped",
                    details={"reason": "missing_mesh_signature"},
                )
            session.commit()
            return "skipped"

        try:
            fresh_resolution = resolve_condition_via_nih(
                condition_label,
                session=session,
                pubmed_searcher=pubmed_searcher,
                refresh_ttl_seconds=settings.search.refresh_ttl_seconds,
            )
        except MeshTermsNotFoundError as exc:
            if refresh_job is not None:
                refresh_job.status = "skipped"
                refresh_job.error_message = None
                _update_refresh_progress(
                    session,
                    refresh_job,
                    stage="skipped",
                    details={
                        "reason": "missing_mesh_terms",
                        "suggestions": list(exc.suggestions),
                    },
                )
                session.commit()
            else:
                session.commit()
            return "skipped"

        if refresh_job is not None:
            _update_refresh_progress(
                session,
                refresh_job,
                stage="collecting_articles",
                details={"search_term_id": fresh_resolution.search_term_id},
            )

        collect_pubmed_articles(
            fresh_resolution,
            session=session,
            pubmed_searcher=pubmed_searcher,
            full_text_fetcher=full_text_fetcher,
            selection_policy=selection_policy,
        )
        session.flush()

        if refresh_job is not None:
            _update_refresh_progress(
                session,
                refresh_job,
                stage="building_batches",
                details={"search_term_id": fresh_resolution.search_term_id},
            )

        batches = build_llm_batches(
            session,
            search_term_id=fresh_resolution.search_term_id,
            condition_label=condition_label,
            mesh_terms=fresh_resolution.mesh_terms,
        )
        if not batches:
            if refresh_job is not None:
                refresh_job.status = "no-batches"
                refresh_job.error_message = None
                _update_refresh_progress(
                    session,
                    refresh_job,
                    stage="no_batches",
                    details={"reason": "no_llm_batches"},
                )
            session.commit()
            return "no-batches"

        if refresh_job is not None:
            _update_refresh_progress(
                session,
                refresh_job,
                stage="invoking_llm",
                details={"batch_count": len(batches)},
            )

        client = OpenAIChatClient()
        responses = client.run_batches(batches)
        if not responses:
            if refresh_job is not None:
                refresh_job.status = "no-responses"
                refresh_job.error_message = None
                _update_refresh_progress(
                    session,
                    refresh_job,
                    stage="no_responses",
                    details={"reason": "llm_returned_no_responses"},
                )
            session.commit()
            return "no-responses"

        if refresh_job is not None:
            _update_refresh_progress(
                session,
                refresh_job,
                stage="persisting_claims",
                details={"response_count": len(responses)},
            )

        claim_set = persist_processed_claims(
            session,
            search_term_id=fresh_resolution.search_term_id,
            mesh_signature=compute_mesh_signature(list(fresh_resolution.mesh_terms)),
            condition_label=condition_label,
            llm_payloads=responses,
        )
        claim_count = len(claim_set.claims) if getattr(claim_set, "claims", None) is not None else 0

        if refresh_job is not None:
            refresh_job.error_message = None
            if claim_count == 0:
                refresh_job.status = "empty-results"
                _update_refresh_progress(
                    session,
                    refresh_job,
                    stage="empty_results",
                    details={
                        "response_count": len(responses),
                        "claim_count": claim_count,
                    },
                )
            else:
                refresh_job.status = "completed"
                _update_refresh_progress(
                    session,
                    refresh_job,
                    stage="completed",
                    details={
                        "response_count": len(responses),
                        "claim_count": claim_count,
                    },
                )

        session.commit()
        return "empty-results" if claim_count == 0 else "completed"
    except Exception as exc:  # noqa: BLE001
        session.rollback()
        job_id = getattr(self.request, "id", None)
        if job_id:
            refreshed_job = session.execute(
                select(ClaimSetRefresh).where(ClaimSetRefresh.job_id == job_id)
            ).scalar_one_or_none()
            if refreshed_job is not None:
                refreshed_job.status = "failed"
                refreshed_job.error_message = str(exc)
                _update_refresh_progress(
                    session,
                    refreshed_job,
                    stage="failed",
                    details={"error": str(exc)},
                )
                session.commit()
            else:
                session.rollback()
        raise self.retry(exc=exc, countdown=30, max_retries=3)
    finally:
        session.close()


def _update_refresh_progress(
    session,
    refresh_job: ClaimSetRefresh,
    *,
    stage: str,
    details: dict | None = None,
) -> None:
    if refresh_job is None:
        return
    refresh_job.progress_state = stage
    refresh_job.progress_payload = details or {}
    session.flush()

