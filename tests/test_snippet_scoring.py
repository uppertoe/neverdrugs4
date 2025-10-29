from __future__ import annotations

from dataclasses import dataclass

import pytest

from app.services.snippet_scoring import compute_quota, score_snippet


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


@dataclass(slots=True)
class _FakeCandidate:
    pmc_ref_count: int
    article_score: float


@pytest.mark.parametrize(
    "candidate,expected",
    [
        (_FakeCandidate(pmc_ref_count=5, article_score=2.0), 3),
        (_FakeCandidate(pmc_ref_count=12, article_score=2.0), 4),
        (_FakeCandidate(pmc_ref_count=32, article_score=3.5), 5),
        (_FakeCandidate(pmc_ref_count=32, article_score=4.2), 6),
        (_FakeCandidate(pmc_ref_count=75, article_score=5.0), 6),
    ],
)
def test_compute_quota_respects_thresholds(candidate: _FakeCandidate, expected: int) -> None:
    assert compute_quota(candidate, base_quota=3, max_quota=6) == expected
