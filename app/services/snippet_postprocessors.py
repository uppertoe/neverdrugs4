from __future__ import annotations

from collections import defaultdict
from typing import Sequence

from app.services.snippets import SnippetResult


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
    "LimitPerDrugPostProcessor",
]
