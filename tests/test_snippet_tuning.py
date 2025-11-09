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
    snippet_text: str | None = None,
) -> SnippetResult:
    text = snippet_text if snippet_text is not None else f"snippet-{drug}-{score}"
    candidate = SnippetCandidate(
        pmid=pmid,
        drug=drug,
        classification=classification,
        snippet_text=text,
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


def test_generate_quota_grid_builds_cartesian_products() -> None:
    configs = generate_quota_grid(per_drug_limits=(1, 2), max_total_results=(None, 4))

    per_drug_values = {config.per_drug_limit for config in configs}
    total_caps = {config.max_total_snippets for config in configs}

    assert per_drug_values == {1, 2}
    assert total_caps == {None, 4}


def test_grid_search_selects_highest_scoring_config() -> None:
    snippet_results = [
        _make_result("pmid-1", "propofol", 3.0, snippet_text="propofol-primary"),
        _make_result("pmid-1", "propofol", 2.5, snippet_text="propofol-secondary"),
        _make_result("pmid-1", "ketamine", 2.0, snippet_text="ketamine-primary"),
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
        SnippetPipelineConfig(per_drug_limit=1),
        SnippetPipelineConfig(per_drug_limit=2),
    ]

    tuning_results = grid_search_pipeline_configs(
        configs,
        articles=articles,
        evaluate_results=evaluate,
        extractor=extractor,
    )

    assert tuning_results[0].config.per_drug_limit == 2
    assert tuning_results[0].score > tuning_results[1].score
