from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal, Protocol, Sequence, runtime_checkable


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
    study_high_bonus: float = 0.6
    study_review_bonus: float = 0.4
    study_guideline_bonus: float = 0.25
    study_observational_bonus: float = 0.2
    study_low_penalty: float = -0.3
    recency_half_life_years: float = 12.0
    recency_max_bonus: float = 0.5
    recency_old_years: int = 25
    recency_old_penalty: float = -0.35
    cohort_small_threshold: int = 20
    cohort_small_penalty: float = -0.2
    cohort_log_weight: float = 0.18
    cohort_log_base: float = 2.0
    cohort_max_bonus: float = 0.6


@dataclass(frozen=True)
class SnippetQuotaConfig:
    pmc_bonus_threshold: int = 0
    pmc_high_bonus_threshold: int = 0
    pmc_bonus_increment: int = 0
    pmc_high_bonus_increment: int = 0
    article_score_threshold: float = 0.0
    article_score_increment: int = 0


DEFAULT_SCORING_CONFIG = SnippetScoringConfig()
DEFAULT_QUOTA_CONFIG = SnippetQuotaConfig()

_HIGH_EVIDENCE_TYPES = {
    "clinical study",
    "clinical trial",
    "clinical trial, phase i",
    "clinical trial, phase ii",
    "clinical trial, phase iii",
    "clinical trial, phase iv",
    "controlled clinical trial",
    "multicenter study",
    "randomized controlled trial",
}
_REVIEW_EVIDENCE_TYPES = {
    "systematic review",
    "meta-analysis",
}
_GUIDELINE_EVIDENCE_TYPES = {
    "practice guideline",
    "guideline",
}
_OBSERVATIONAL_EVIDENCE_TYPES = {
    "comparative study",
    "cohort studies",
    "cross-sectional studies",
    "case-control studies",
    "observational study",
}
_LOW_EVIDENCE_TYPES = {
    "case reports",
    "letter",
    "editorial",
}


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
    study_types: Sequence[str] | None = None,
    publication_year: int | None = None,
    cohort_size: int | None = None,
) -> float:
    resolved = config or DEFAULT_SCORING_CONFIG
    return _score_snippet_with_config(
        article_score=article_score,
        pmc_ref_count=pmc_ref_count,
        classification=classification,
        cue_count=cue_count,
        condition_match=condition_match,
        config=resolved,
        study_types=study_types,
        publication_year=publication_year,
        cohort_size=cohort_size,
    )


def _score_snippet_with_config(
    *,
    article_score: float,
    pmc_ref_count: int,
    classification: Literal["risk", "safety"],
    cue_count: int,
    condition_match: bool,
    config: SnippetScoringConfig,
    study_types: Sequence[str] | None = None,
    publication_year: int | None = None,
    cohort_size: int | None = None,
) -> float:
    score = article_score * config.article_score_weight
    score += min(pmc_ref_count * config.pmc_ref_weight, config.pmc_ref_cap)
    score += config.risk_bonus if classification == "risk" else config.safety_bonus
    score += config.cue_weight * cue_count
    score += config.condition_bonus if condition_match else config.condition_penalty
    score += _apply_study_type_component(study_types, config)
    score += _apply_recency_component(publication_year, config)
    score += _apply_cohort_component(cohort_size, config)
    return round(score, 4)


def _apply_study_type_component(
    study_types: Sequence[str] | None,
    config: SnippetScoringConfig,
) -> float:
    if not study_types:
        return 0.0
    normalized = {entry.strip().lower() for entry in study_types if entry}
    if not normalized:
        return 0.0
    bonus = 0.0
    if normalized & _HIGH_EVIDENCE_TYPES:
        bonus += config.study_high_bonus
    if normalized & _REVIEW_EVIDENCE_TYPES:
        bonus += config.study_review_bonus
    if normalized & _GUIDELINE_EVIDENCE_TYPES:
        bonus += config.study_guideline_bonus
    if normalized & _OBSERVATIONAL_EVIDENCE_TYPES:
        bonus += config.study_observational_bonus
    if normalized & _LOW_EVIDENCE_TYPES:
        bonus += config.study_low_penalty
    return bonus


def _apply_recency_component(
    publication_year: int | None,
    config: SnippetScoringConfig,
) -> float:
    if not publication_year:
        return 0.0
    current_year = datetime.now(UTC).year
    years_old = max(0, current_year - publication_year)
    half_life = max(config.recency_half_life_years, 1.0)
    decay = math.exp(-years_old / half_life)
    bonus = config.recency_max_bonus * decay
    if years_old > config.recency_old_years:
        bonus += config.recency_old_penalty
    return bonus


def _apply_cohort_component(
    cohort_size: int | None,
    config: SnippetScoringConfig,
) -> float:
    if cohort_size is None or cohort_size <= 0:
        return 0.0
    if cohort_size < config.cohort_small_threshold:
        return config.cohort_small_penalty
    threshold = max(config.cohort_small_threshold, 1)
    ratio = cohort_size / threshold
    value = config.cohort_log_weight * math.log(ratio, config.cohort_log_base)
    return min(config.cohort_max_bonus, value)


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
    if cfg.pmc_bonus_increment > 0 and candidate.pmc_ref_count >= cfg.pmc_bonus_threshold:
        quota += cfg.pmc_bonus_increment
    if cfg.pmc_high_bonus_increment > 0 and candidate.pmc_ref_count >= cfg.pmc_high_bonus_threshold:
        quota += cfg.pmc_high_bonus_increment
    if cfg.article_score_increment > 0 and candidate.article_score >= cfg.article_score_threshold:
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
