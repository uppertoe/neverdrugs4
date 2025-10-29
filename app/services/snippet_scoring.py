from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable


@dataclass(frozen=True)
class SnippetScoringConfig:
    article_score_weight: float = 1.0
    pmc_ref_weight: float = 1.0 / 40.0
    pmc_ref_cap: float = 2.0
    risk_bonus: float = 0.5
    safety_bonus: float = 0.3
    cue_weight: float = 0.1
    condition_bonus: float = 0.4
    condition_penalty: float = -0.2


@dataclass(frozen=True)
class SnippetQuotaConfig:
    pmc_bonus_threshold: int = 10
    pmc_high_bonus_threshold: int = 30
    pmc_bonus_increment: int = 1
    pmc_high_bonus_increment: int = 1
    article_score_threshold: float = 4.0
    article_score_increment: int = 1


DEFAULT_SCORING_CONFIG = SnippetScoringConfig()
DEFAULT_QUOTA_CONFIG = SnippetQuotaConfig()


@runtime_checkable
class SupportsQuotaInputs(Protocol):
    pmc_ref_count: int
    article_score: float


def score_snippet(
    *,
    article_score: float,
    pmc_ref_count: int,
    classification: Literal["risk", "safety"],
    cue_count: int,
    condition_match: bool,
) -> float:
    config = DEFAULT_SCORING_CONFIG
    return _score_snippet_with_config(
        article_score=article_score,
        pmc_ref_count=pmc_ref_count,
        classification=classification,
        cue_count=cue_count,
        condition_match=condition_match,
        config=config,
    )


def score_snippet_with_config(
    *,
    article_score: float,
    pmc_ref_count: int,
    classification: Literal["risk", "safety"],
    cue_count: int,
    condition_match: bool,
    config: SnippetScoringConfig | None,
) -> float:
    resolved = config or DEFAULT_SCORING_CONFIG
    return _score_snippet_with_config(
        article_score=article_score,
        pmc_ref_count=pmc_ref_count,
        classification=classification,
        cue_count=cue_count,
        condition_match=condition_match,
        config=resolved,
    )


def _score_snippet_with_config(
    *,
    article_score: float,
    pmc_ref_count: int,
    classification: Literal["risk", "safety"],
    cue_count: int,
    condition_match: bool,
    config: SnippetScoringConfig,
) -> float:
    score = article_score * config.article_score_weight
    score += min(pmc_ref_count * config.pmc_ref_weight, config.pmc_ref_cap)
    score += config.risk_bonus if classification == "risk" else config.safety_bonus
    score += config.cue_weight * cue_count
    score += config.condition_bonus if condition_match else config.condition_penalty
    return round(score, 4)


def compute_quota(
    candidate: SupportsQuotaInputs,
    *,
    base_quota: int,
    max_quota: int,
) -> int:
    return compute_quota_with_config(
        candidate,
        base_quota=base_quota,
        max_quota=max_quota,
        config=DEFAULT_QUOTA_CONFIG,
    )


def compute_quota_with_config(
    candidate: SupportsQuotaInputs,
    *,
    base_quota: int,
    max_quota: int,
    config: SnippetQuotaConfig | None,
) -> int:
    resolved = config or DEFAULT_QUOTA_CONFIG
    quota = base_quota
    cfg = resolved
    if candidate.pmc_ref_count >= cfg.pmc_bonus_threshold:
        quota += cfg.pmc_bonus_increment
    if candidate.pmc_ref_count >= cfg.pmc_high_bonus_threshold:
        quota += cfg.pmc_high_bonus_increment
    if candidate.article_score >= cfg.article_score_threshold:
        quota += cfg.article_score_increment
    return min(max_quota, quota)


__all__ = [
    "SnippetScoringConfig",
    "SnippetQuotaConfig",
    "DEFAULT_SCORING_CONFIG",
    "DEFAULT_QUOTA_CONFIG",
    "score_snippet",
    "score_snippet_with_config",
    "compute_quota",
    "compute_quota_with_config",
]
