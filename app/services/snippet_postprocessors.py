from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Sequence

from app.services.snippets import SnippetResult


class EnsureClassificationCoverage:
    def __init__(
        self,
        *,
        required_classifications: Iterable[str] = ("risk", "safety"),
        score_boost: float = 0.05,
    ) -> None:
        self.required = tuple({classification for classification in required_classifications})
        self.score_boost = float(score_boost)

    def process(self, results: Sequence[SnippetResult]) -> Sequence[SnippetResult]:
        if not results or not self.required:
            return results

        ranked = sorted(results, key=lambda item: item.candidate.snippet_score, reverse=True)
        for classification in self.required:
            candidate_result = next(
                (result for result in ranked if result.candidate.classification == classification),
                None,
            )
            if candidate_result is None:
                continue
            if not candidate_result.metadata.get("coverage_boosted"):
                candidate_result.candidate.snippet_score += self.score_boost
                candidate_result.metadata["coverage_boosted"] = True
                candidate_result.metadata["coverage_score_boost"] = self.score_boost
        return results


class LimitPerDrugPostProcessor:
    def __init__(self, *, max_per_drug: int) -> None:
        self.max_per_drug = max(0, max_per_drug)

    def process(self, results: Sequence[SnippetResult]) -> Sequence[SnippetResult]:
        if not results or self.max_per_drug == 0:
            return []

        kept: list[SnippetResult] = []
        counts: dict[str, int] = defaultdict(int)
        for result in results:
            key = result.candidate.drug.lower()
            counts[key] += 1
            if counts[key] <= self.max_per_drug:
                kept.append(result)
        return kept


__all__ = [
    "EnsureClassificationCoverage",
    "LimitPerDrugPostProcessor",
]
