from __future__ import annotations

from dataclasses import dataclass

import pytest

from datetime import UTC, datetime

from app.services.snippet_scoring import (
    SnippetScoringConfig,
    compute_quota,
    score_snippet,
    score_snippet_with_config,
)


def test_score_snippet_rewards_risk_and_condition_match() -> None:
    score = score_snippet(
        article_score=4.0,
        pmc_ref_count=40,
        classification="risk",
        cue_count=3,
        condition_match=True,
    )

    assert score == pytest.approx(4.0 + 1.0 + 0.5 + 0.3 + 0.4)


def test_score_snippet_penalises_missing_condition_match() -> None:
    score = score_snippet(
        article_score=2.0,
        pmc_ref_count=5,
        classification="safety",
        cue_count=1,
        condition_match=False,
    )

    assert score == pytest.approx(2.0 + 0.125 + 0.3 + 0.1 - 0.2)


def test_score_snippet_awards_high_evidence_bonus() -> None:
    config = SnippetScoringConfig()
    baseline = score_snippet_with_config(
        article_score=3.0,
        pmc_ref_count=10,
        classification="risk",
        cue_count=2,
        condition_match=True,
        config=config,
    )
    enhanced = score_snippet_with_config(
        article_score=3.0,
        pmc_ref_count=10,
        classification="risk",
        cue_count=2,
        condition_match=True,
        config=config,
        study_types=("Randomized Controlled Trial",),
    )

    assert enhanced == pytest.approx(baseline + config.study_high_bonus)


def test_score_snippet_penalises_low_evidence_types() -> None:
    config = SnippetScoringConfig()
    baseline = score_snippet_with_config(
        article_score=2.5,
        pmc_ref_count=2,
        classification="risk",
        cue_count=1,
        condition_match=True,
        config=config,
    )
    penalised = score_snippet_with_config(
        article_score=2.5,
        pmc_ref_count=2,
        classification="risk",
        cue_count=1,
        condition_match=True,
        config=config,
        study_types=("Case Reports",),
    )

    assert penalised == pytest.approx(baseline + config.study_low_penalty)


def test_score_snippet_applies_recency_bonus() -> None:
    config = SnippetScoringConfig(recency_max_bonus=0.6, recency_half_life_years=8.0, recency_old_years=40)
    current_year = datetime.now(UTC).year
    recent = score_snippet_with_config(
        article_score=2.0,
        pmc_ref_count=5,
        classification="risk",
        cue_count=1,
        condition_match=True,
        config=config,
        publication_year=current_year,
    )
    older = score_snippet_with_config(
        article_score=2.0,
        pmc_ref_count=5,
        classification="risk",
        cue_count=1,
        condition_match=True,
        config=config,
        publication_year=current_year - 20,
    )

    assert recent > older


def test_score_snippet_rewards_large_cohort() -> None:
    config = SnippetScoringConfig()
    small = score_snippet_with_config(
        article_score=2.0,
        pmc_ref_count=5,
        classification="risk",
        cue_count=1,
        condition_match=True,
        config=config,
        cohort_size=config.cohort_small_threshold - 1,
    )
    large = score_snippet_with_config(
        article_score=2.0,
        pmc_ref_count=5,
        classification="risk",
        cue_count=1,
        condition_match=True,
        config=config,
        cohort_size=config.cohort_small_threshold * 10,
    )

    assert large > small


@dataclass(slots=True)
class _FakeCandidate:
    pmc_ref_count: int
    article_score: float


@pytest.mark.parametrize(
    "candidate",
    [
        _FakeCandidate(pmc_ref_count=5, article_score=2.0),
        _FakeCandidate(pmc_ref_count=12, article_score=2.0),
        _FakeCandidate(pmc_ref_count=32, article_score=3.5),
        _FakeCandidate(pmc_ref_count=32, article_score=4.2),
        _FakeCandidate(pmc_ref_count=75, article_score=5.0),
    ],
)
def test_compute_quota_defaults_to_base(candidate: _FakeCandidate) -> None:
    assert compute_quota(candidate, base_quota=3, max_quota=6) == 3


def test_compute_quota_with_custom_config_applies_increments() -> None:
    from app.services.snippet_scoring import SnippetQuotaConfig, compute_quota_with_config

    candidate = _FakeCandidate(pmc_ref_count=35, article_score=4.5)
    config = SnippetQuotaConfig(
        pmc_bonus_threshold=10,
        pmc_high_bonus_threshold=30,
        pmc_bonus_increment=1,
        pmc_high_bonus_increment=1,
        article_score_threshold=4.0,
        article_score_increment=1,
    )

    quota = compute_quota_with_config(
        candidate,
        base_quota=2,
        max_quota=6,
        config=config,
    )

    assert quota == 5
