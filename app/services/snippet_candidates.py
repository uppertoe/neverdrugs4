from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Iterator, Protocol


def _normalize_whitespace(value: str) -> str:
    return " ".join(value.split())


@dataclass(frozen=True)
class SnippetSpan:
    text: str
    left: int
    right: int
    match_start: int
    match_end: int


class SnippetCandidateFinder(Protocol):
    def find_candidates(
        self,
        *,
        article_text: str,
        drug: str,
        normalized_text: str | None = None,
        lower_text: str | None = None,
    ) -> Iterator[SnippetSpan]:
        ...


class RegexSnippetCandidateFinder:
    def __init__(
        self,
        *,
        drug_terms: Iterable[str],
        window_chars: int = 600,
    ) -> None:
        self._window_chars = max(100, window_chars)
        terms = {term.lower() for term in drug_terms if term}
        self._patterns = {term: re.compile(rf"\b{re.escape(term)}\b") for term in terms}

    def find_candidates(
        self,
        *,
        article_text: str,
        drug: str,
        normalized_text: str | None = None,
        lower_text: str | None = None,
    ) -> Iterator[SnippetSpan]:
        source_text = normalized_text or article_text
        if not source_text:
            return iter(())

        normalized = normalized_text or _normalize_whitespace(article_text)
        lowered = lower_text or normalized.lower()
        pattern = self._patterns.get(drug.lower())
        if pattern is None:
            return iter(())

        for match in pattern.finditer(lowered):
            match_start = match.start()
            match_end = match.end()
            left_bound = max(0, match_start - self._window_chars)
            right_bound = min(len(normalized), match_end + self._window_chars)
            snippet = normalized[left_bound:right_bound].strip()
            adjusted_left = normalized.find(snippet, left_bound, right_bound) if snippet else left_bound
            if adjusted_left == -1:
                adjusted_left = left_bound
            adjusted_right = adjusted_left + len(snippet)

            yield SnippetSpan(
                text=snippet,
                left=adjusted_left,
                right=adjusted_right,
                match_start=match_start,
                match_end=match_end,
            )


__all__ = ["RegexSnippetCandidateFinder", "SnippetCandidateFinder", "SnippetSpan"]
