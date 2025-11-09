import os

from celery import Celery


def _default_redis_url() -> str:
    return os.environ.get("REDIS_URL", "redis://redis:6379/0")


def make_celery() -> Celery:
    broker_url = os.environ.get("CELERY_BROKER_URL", _default_redis_url())
    backend_url = os.environ.get("CELERY_RESULT_BACKEND", broker_url)

    celery_app = Celery(
        "nih_module",
        broker=broker_url,
        backend=backend_url,
        include=["app.tasks"],
    )

    celery_app.conf.update(
        task_track_started=True,
        result_expires=int(os.environ.get("CELERY_RESULT_TTL", "3600")),
        task_acks_late=True,
    )

    return celery_app


celery = make_celery()
