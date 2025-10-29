from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

from app.services.snippet_pipeline import (
    SnippetExtractionPipeline,
    SnippetPipelineConfig,
    SnippetPostProcessor,
)
from app.services.snippets import ArticleSnippetExtractor, SnippetResult
from app.services.snippet_scoring import SnippetQuotaConfig


@dataclass(slots=True)
class SnippetArticleInput:
    article_text: str
    pmid: str
    condition_terms: Sequence[str]
    article_rank: int
    article_score: float
    preferred_url: str
    pmc_ref_count: int


@dataclass(slots=True)
class TuningResult:
    config: SnippetPipelineConfig
    score: float
    metadata: dict[str, object] = field(default_factory=dict)


def grid_search_pipeline_configs(
    configurations: Sequence[SnippetPipelineConfig],
    *,
    articles: Sequence[SnippetArticleInput],
    evaluate_results: Callable[[Sequence[SnippetResult]], float],
    extractor: ArticleSnippetExtractor | None = None,
    post_processors: Sequence[SnippetPostProcessor] | None = None,
) -> list[TuningResult]:
    if not configurations:
        return []

    shared_extractor = extractor or ArticleSnippetExtractor()
    processors = tuple(post_processors or ())

    results: list[TuningResult] = []
    for config in configurations:
        pipeline = SnippetExtractionPipeline(
            extractor=shared_extractor,
            post_processors=processors,
            config=config,
        )
        total_score = 0.0
        evaluations = 0
        for article in articles:
            snippet_results = pipeline.run_results(
                article_text=article.article_text,
                pmid=article.pmid,
                condition_terms=article.condition_terms,
                article_rank=article.article_rank,
                article_score=article.article_score,
                preferred_url=article.preferred_url,
                pmc_ref_count=article.pmc_ref_count,
            )
            total_score += evaluate_results(snippet_results)
            evaluations += 1
        average_score = total_score / evaluations if evaluations else 0.0
        results.append(
            TuningResult(
                config=config,
                score=average_score,
            )
        )

    results.sort(key=lambda item: item.score, reverse=True)
    return results


def generate_quota_grid(
    *,
    base_range: Iterable[int],
    max_range: Iterable[int],
    quota_config: SnippetQuotaConfig | None = None,
) -> list[SnippetPipelineConfig]:
    configs: list[SnippetPipelineConfig] = []
    for base_quota in base_range:
        for max_quota in max_range:
            if max_quota < base_quota:
                continue
            configs.append(
                SnippetPipelineConfig(
                    base_quota=base_quota,
                    max_quota=max_quota,
                    quota_config=quota_config,
                )
            )
    return configs


__all__ = [
    "SnippetArticleInput",
    "SnippetPipelineConfig",
    "TuningResult",
    "generate_quota_grid",
    "grid_search_pipeline_configs",
]
