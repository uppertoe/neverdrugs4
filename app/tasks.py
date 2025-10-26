from __future__ import annotations

from time import sleep

from .celery_app import celery


@celery.task(bind=True)
def ping(self) -> str:
    """Simple task to verify Celery worker wiring."""
    sleep(0.1)
    return "pong"
