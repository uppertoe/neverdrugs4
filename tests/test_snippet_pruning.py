from __future__ import annotations

from typing import Literal

from app.services.snippet_candidates import SnippetSpan
from app.services.snippet_pruning import (
    WindowedCandidate,
    apply_article_quotas,
    prune_window_overlaps,
)
from app.services.snippets import SnippetCandidate


def _make_candidate(
    *,
    snippet_text: str,
    pmid: str = "pmid-1",
    drug: str = "propofol",
    classification: Literal["risk", "safety"] = "risk",
    score: float = 1.0,
    article_score: float = 2.0,
    pmc_ref_count: int = 0,
) -> SnippetCandidate:
    return SnippetCandidate(
        pmid=pmid,
        drug=drug,
    classification=classification,
        snippet_text=snippet_text,
        article_rank=1,
        article_score=article_score,
        preferred_url="https://example.org",
        pmc_ref_count=pmc_ref_count,
        snippet_score=score,
        cues=[],
        tags=[],
    )


def _span(left: int, right: int) -> SnippetSpan:
    return SnippetSpan(
        text="snippet text",
        left=left,
        right=right,
        match_start=left,
        match_end=left + 5,
    )


def test_pruner_prefers_higher_scoring_overlap() -> None:
    low_candidate_text = "alpha"
    low_candidate = WindowedCandidate(
        candidate=_make_candidate(snippet_text=low_candidate_text, score=1.0),
        span=_span(0, 100),
        key=("propofol", low_candidate_text.lower()),
    )
    high_candidate_text = "beta"
    high_candidate = WindowedCandidate(
        candidate=_make_candidate(snippet_text=high_candidate_text, score=3.0),
        span=_span(50, 150),
        key=("propofol", high_candidate_text.lower()),
        metadata={"source": "high"},
    )

    pruned = prune_window_overlaps([low_candidate, high_candidate])

    assert len(pruned) == 1
    assert pruned[0].candidate.snippet_text == "beta"
    assert pruned[0].candidate.snippet_score == 3.0
    assert pruned[0].metadata == {"source": "high"}


def test_pruner_keeps_overlaps_for_different_drugs() -> None:
    propofol_text = "alpha"
    propofol_candidate = WindowedCandidate(
        candidate=_make_candidate(snippet_text=propofol_text, drug="propofol", score=2.0),
        span=_span(0, 100),
        key=("propofol", propofol_text.lower()),
    )
    ketamine_text = "beta"
    ketamine_candidate = WindowedCandidate(
        candidate=_make_candidate(
            snippet_text=ketamine_text,
            drug="ketamine",
            score=1.0,
        ),
        span=_span(50, 150),
        key=("ketamine", ketamine_text.lower()),
    )

    pruned = prune_window_overlaps([propofol_candidate, ketamine_candidate])

    assert len(pruned) == 2
    drugs = {entry.candidate.drug for entry in pruned}
    assert drugs == {"propofol", "ketamine"}


def test_pruner_deduplicates_repeated_snippets() -> None:
    first_text = "gamma"
    first = WindowedCandidate(
        candidate=_make_candidate(snippet_text=first_text, score=2.0),
        span=_span(0, 100),
        key=("propofol", first_text.lower()),
    )
    duplicate = WindowedCandidate(
        candidate=_make_candidate(snippet_text=first_text, score=1.5),
        span=_span(200, 300),
        key=("propofol", first_text.lower()),
    )

    pruned = prune_window_overlaps([first, duplicate])

    assert len(pruned) == 1
    assert pruned[0].candidate.snippet_score == 2.0


def test_apply_article_quotas_scales_with_article_weight() -> None:
    rich_candidates = [
        _make_candidate(
            pmid="39618072",
            drug="succinylcholine",
            classification="risk",
            snippet_text=f"snippet {idx}",
            score=5.0 - idx * 0.2,
            article_score=4.5,
            pmc_ref_count=40,
        )
        for idx in range(5)
    ]

    low_candidates = [
        _make_candidate(
            pmid="15859443",
            drug="propofol",
            classification="safety",
            snippet_text=f"low {idx}",
            score=1.0 - idx * 0.1,
            article_score=1.1,
            pmc_ref_count=0,
        )
        for idx in range(4)
    ]

    selection = apply_article_quotas(rich_candidates + low_candidates)

    high_article_snippets = [candidate for candidate in selection if candidate.pmid == "39618072"]
    low_article_snippets = [candidate for candidate in selection if candidate.pmid == "15859443"]

    assert len(high_article_snippets) == 5
    assert len(low_article_snippets) == 3
    low_scores = [candidate.snippet_score for candidate in low_article_snippets]
    assert low_scores == sorted(low_scores, reverse=True)
