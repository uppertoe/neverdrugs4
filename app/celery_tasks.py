from __future__ import annotations

from app.celery_app import celery
from app.tasks import ping as _ping, refresh_claims_for_condition as _refresh_claims_for_condition


@celery.task(name="app.tasks.ping")
def ping() -> str:
    """Celery entrypoint that delegates to the service-level ping helper."""
    return _ping()


@celery.task(name="app.tasks.refresh_claims_for_condition", bind=True)
def refresh_claims_for_condition(self, **kwargs) -> str:
    """Celery entrypoint that forwards execution to the service pipeline."""
    job_id = getattr(self.request, "id", None)
    return _refresh_claims_for_condition(job_id=job_id, **kwargs)
