from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Protocol, Sequence

from app.services.snippets import ArticleSnippetExtractor, SnippetCandidate, SnippetResult


class SnippetPostProcessor(Protocol):
    def process(self, results: Sequence[SnippetResult]) -> Sequence[SnippetResult]:
        ...


def _default_sort_key(result: SnippetResult) -> tuple[int, float]:
    candidate = result.candidate
    return (candidate.article_rank, -candidate.snippet_score)


@dataclass(slots=True)
class SnippetPipelineConfig:
    per_drug_limit: int = 3
    max_total_snippets: int | None = None
    sort_key: Callable[[SnippetResult], tuple[int, float]] = field(
        default=_default_sort_key,
        repr=False,
    )

    def apply_limits(self, results: Sequence[SnippetResult]) -> list[SnippetResult]:
        if not results:
            return []

        per_drug_limit = max(1, self.per_drug_limit)
        unique_drugs = {
            (result.candidate.drug or "").lower()
            for result in results
        }
        if not unique_drugs:
            return []

        total_cap = per_drug_limit * len(unique_drugs)
        if self.max_total_snippets is not None:
            total_cap = min(total_cap, max(1, self.max_total_snippets))

        counts: dict[str, int] = defaultdict(int)
        ordered = sorted(results, key=self.sort_key)
        limited: list[SnippetResult] = []
        for result in ordered:
            drug = (result.candidate.drug or "").lower()
            if counts[drug] >= per_drug_limit:
                continue
            counts[drug] += 1
            limited.append(result)
            if len(limited) >= total_cap:
                break

        return limited


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
        publication_date: str | None = None,
        publication_types: Sequence[str] | None = None,
        cohort_size: int | None = None,
    ) -> list[SnippetResult]:
        results = self.extractor.extract_snippet_results(
            article_text=article_text,
            pmid=pmid,
            condition_terms=condition_terms,
            article_rank=article_rank,
            article_score=article_score,
            preferred_url=preferred_url,
            pmc_ref_count=pmc_ref_count,
            publication_date=publication_date,
            publication_types=publication_types,
            cohort_size=cohort_size,
        )
        for processor in self.post_processors:
            results = list(processor.process(results))
        deduped = self._dedupe_results(results)
        limited = self.config.apply_limits(deduped)
        return limited

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
        publication_date: str | None = None,
        publication_types: Sequence[str] | None = None,
        cohort_size: int | None = None,
    ) -> list[SnippetCandidate]:
        results = self.run_results(
            article_text=article_text,
            pmid=pmid,
            condition_terms=condition_terms,
            article_rank=article_rank,
            article_score=article_score,
            preferred_url=preferred_url,
            pmc_ref_count=pmc_ref_count,
            publication_date=publication_date,
            publication_types=publication_types,
            cohort_size=cohort_size,
        )
        return [result.candidate for result in results]

    def _dedupe_results(self, results: Sequence[SnippetResult]) -> list[SnippetResult]:
        if not results:
            return []

        deduped: list[SnippetResult] = []
        seen: dict[str, int] = {}

        for result in results:
            key = _build_dedupe_key(result)

            existing_index = seen.get(key)
            if existing_index is None:
                seen[key] = len(deduped)
                deduped.append(result)
                continue

            existing = deduped[existing_index]
            if _prefers_new_result(existing, result):
                deduped[existing_index] = result

        return deduped


__all__ = [
    "SnippetExtractionPipeline",
    "SnippetPipelineConfig",
    "SnippetPostProcessor",
]


def _build_dedupe_key(result: SnippetResult) -> str:
    signature = _normalize_snippet_signature(result.candidate.snippet_text)
    drug = result.candidate.drug.lower()
    return f"{drug}|{signature}" if signature else drug


def _normalize_snippet_signature(value: str) -> str:
    if not value:
        return ""
    return " ".join(value.lower().split())


def _prefers_new_result(current: SnippetResult, challenger: SnippetResult) -> bool:
    current_score = current.candidate.snippet_score
    challenger_score = challenger.candidate.snippet_score
    if challenger_score > current_score:
        return True
    if challenger_score < current_score:
        return False

    current_rank = current.candidate.article_rank
    challenger_rank = challenger.candidate.article_rank
    if challenger_rank < current_rank:
        return True
    if challenger_rank > current_rank:
        return False

    return challenger.candidate.article_score > current.candidate.article_score
