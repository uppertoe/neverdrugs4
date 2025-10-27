from __future__ import annotations

from time import sleep
from typing import Sequence

from sqlalchemy import select

from .celery_app import celery
from .database import create_session_factory
from .models import ClaimSetRefresh
from .services.llm_batches import build_llm_batches
from .services.nih_pipeline import resolve_condition_via_nih
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

        job_id = getattr(self.request, "id", None)
        if job_id:
            refresh_job = session.execute(
                select(ClaimSetRefresh).where(ClaimSetRefresh.job_id == job_id)
            ).scalar_one_or_none()
            if refresh_job is not None:
                refresh_job.status = "running"
                refresh_job.error_message = None
                session.flush()

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
            session.commit()
            return "skipped"

        fresh_resolution = resolve_condition_via_nih(
            condition_label,
            session=session,
            refresh_ttl_seconds=settings.search.refresh_ttl_seconds,
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
            session.commit()
            return "no-batches"

        client = OpenAIChatClient()
        responses = client.run_batches(batches)
        if not responses:
            if refresh_job is not None:
                refresh_job.status = "no-responses"
                refresh_job.error_message = None
            session.commit()
            return "no-responses"

        persist_processed_claims(
            session,
            search_term_id=fresh_resolution.search_term_id,
            mesh_signature=compute_mesh_signature(list(fresh_resolution.mesh_terms)),
            condition_label=condition_label,
            llm_payloads=responses,
        )
        if refresh_job is not None:
            refresh_job.status = "completed"
            refresh_job.error_message = None

        session.commit()
        return "completed"
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
                session.commit()
            else:
                session.rollback()
        raise self.retry(exc=exc, countdown=30, max_retries=3)
    finally:
        session.close()

