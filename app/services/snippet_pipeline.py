from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol, Sequence

from app.services.snippet_pruning import apply_article_quotas
from app.services.snippet_scoring import SnippetQuotaConfig
from app.services.snippets import ArticleSnippetExtractor, SnippetCandidate, SnippetResult


class SnippetPostProcessor(Protocol):
    def process(self, results: Sequence[SnippetResult]) -> Sequence[SnippetResult]:
        ...


@dataclass(slots=True)
class SnippetPipelineConfig:
    base_quota: int = 3
    max_quota: int = 8
    quota_strategy: Callable[..., Sequence[SnippetCandidate]] = field(
        default=apply_article_quotas,
        repr=False,
    )
    quota_config: SnippetQuotaConfig | None = None

    def apply_quota(self, candidates: Sequence[SnippetCandidate]) -> list[SnippetCandidate]:
        if not candidates:
            return []
        selected = self.quota_strategy(
            candidates,
            base_quota=self.base_quota,
            max_quota=self.max_quota,
            quota_config=self.quota_config,
        )
        return list(selected)


@dataclass
class SnippetExtractionPipeline:
    extractor: ArticleSnippetExtractor = field(default_factory=ArticleSnippetExtractor)
    post_processors: Sequence[SnippetPostProcessor] = field(default_factory=tuple)
    config: SnippetPipelineConfig = field(default_factory=SnippetPipelineConfig)

    def run_results(
        self,
        *,
        article_text: str,
        pmid: str,
        condition_terms: Sequence[str],
        article_rank: int,
        article_score: float,
        preferred_url: str,
        pmc_ref_count: int,
    ) -> list[SnippetResult]:
        results = self.extractor.extract_snippet_results(
            article_text=article_text,
            pmid=pmid,
            condition_terms=condition_terms,
            article_rank=article_rank,
            article_score=article_score,
            preferred_url=preferred_url,
            pmc_ref_count=pmc_ref_count,
        )
        for processor in self.post_processors:
            results = list(processor.process(results))
        return self._apply_quota(results)

    def run(
        self,
        *,
        article_text: str,
        pmid: str,
        condition_terms: Sequence[str],
        article_rank: int,
        article_score: float,
        preferred_url: str,
        pmc_ref_count: int,
    ) -> list[SnippetCandidate]:
        results = self.run_results(
            article_text=article_text,
            pmid=pmid,
            condition_terms=condition_terms,
            article_rank=article_rank,
            article_score=article_score,
            preferred_url=preferred_url,
            pmc_ref_count=pmc_ref_count,
        )
        return [result.candidate for result in results]

    def _apply_quota(self, results: Sequence[SnippetResult]) -> list[SnippetResult]:
        if not results:
            return []

        selected_candidates = self.config.apply_quota(
            [result.candidate for result in results]
        )
        if len(selected_candidates) == len(results):
            return list(results)

        candidate_to_result = {id(result.candidate): result for result in results}
        ordered: list[SnippetResult] = []
        for candidate in selected_candidates:
            mapped = candidate_to_result.get(id(candidate))
            if mapped is not None:
                ordered.append(mapped)
        return ordered


__all__ = [
    "SnippetExtractionPipeline",
    "SnippetPipelineConfig",
    "SnippetPostProcessor",
]
