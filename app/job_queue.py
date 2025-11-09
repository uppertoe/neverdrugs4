from __future__ import annotations

from typing import Callable, Mapping

from app.services.search import SearchResolution

ClaimRefreshEnqueuer = Callable[..., Mapping[str, object]]

_claim_refresh_enqueuer: ClaimRefreshEnqueuer | None = None


def configure_claim_refresh_enqueuer(enqueuer: ClaimRefreshEnqueuer) -> None:
    """Register the callable responsible for queuing claim refresh jobs."""
    global _claim_refresh_enqueuer
    _claim_refresh_enqueuer = enqueuer


def clear_claim_refresh_enqueuer() -> None:
    """Remove any configured claim refresh enqueuer."""
    global _claim_refresh_enqueuer
    _claim_refresh_enqueuer = None


def enqueue_claim_refresh(
    *,
    session,
    resolution: SearchResolution,
    condition_label: str,
    mesh_signature: str | None,
) -> Mapping[str, object]:
    if _claim_refresh_enqueuer is None:
        raise RuntimeError("Claim refresh enqueuer is not configured.")
    return _claim_refresh_enqueuer(
        session=session,
        resolution=resolution,
        condition_label=condition_label,
        mesh_signature=mesh_signature,
    )
