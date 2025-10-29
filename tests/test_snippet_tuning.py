from __future__ import annotations

from typing import Literal, Sequence

from app.services.snippet_candidates import SnippetSpan
from app.services.snippet_pipeline import SnippetPipelineConfig
from app.services.snippet_tuning import (
    SnippetArticleInput,
    generate_quota_grid,
    grid_search_pipeline_configs,
)
from app.services.snippets import SnippetCandidate, SnippetResult


class _StaticExtractor:
    def __init__(self, snippet_results: Sequence[SnippetResult]) -> None:
        self._results = list(snippet_results)

    def extract_snippet_results(self, **_: object) -> list[SnippetResult]:
        return [
            SnippetResult(
                candidate=result.candidate,
                span=result.span,
                metadata=dict(result.metadata),
            )
            for result in self._results
        ]


def _make_result(
    pmid: str,
    drug: str,
    score: float,
    classification: Literal["risk", "safety"] = "risk",
) -> SnippetResult:
    candidate = SnippetCandidate(
        pmid=pmid,
        drug=drug,
        classification=classification,
        snippet_text=f"snippet-{drug}",
        article_rank=1,
        article_score=4.0,
        preferred_url="https://example.org",
        pmc_ref_count=20,
        snippet_score=score,
        cues=["risk"],
        tags=[],
    )
    span = SnippetSpan(
        text=candidate.snippet_text,
        left=0,
        right=len(candidate.snippet_text),
        match_start=0,
        match_end=min(len(candidate.snippet_text), 5),
    )
    return SnippetResult(candidate=candidate, span=span, metadata={})


def test_generate_quota_grid_filters_invalid_pairs() -> None:
    configs = generate_quota_grid(base_range=(1, 2), max_range=(1, 2))

    assert all(config.max_quota >= config.base_quota for config in configs)
    assert any(config.base_quota == 1 and config.max_quota == 2 for config in configs)


def test_grid_search_selects_highest_scoring_config() -> None:
    snippet_results = [
        _make_result("pmid-1", "propofol", 3.0),
        _make_result("pmid-1", "ketamine", 2.0),
    ]
    extractor = _StaticExtractor(snippet_results)

    def evaluate(results: Sequence[SnippetResult]) -> float:
        return sum(result.candidate.snippet_score for result in results)

    articles = [
        SnippetArticleInput(
            article_text="",
            pmid="pmid-1",
            condition_terms=["condition"],
            article_rank=1,
            article_score=4.0,
            preferred_url="https://example.org",
            pmc_ref_count=20,
        )
    ]

    configs = [
        SnippetPipelineConfig(base_quota=1, max_quota=1),
        SnippetPipelineConfig(base_quota=2, max_quota=2),
    ]

    tuning_results = grid_search_pipeline_configs(
        configs,
        articles=articles,
        evaluate_results=evaluate,
    extractor=extractor,
    )

    assert tuning_results[0].config.base_quota == 2
    assert tuning_results[0].score > tuning_results[1].score
