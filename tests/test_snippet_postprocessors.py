from __future__ import annotations

from typing import Literal

from app.services.snippet_candidates import SnippetSpan
from app.services.snippet_postprocessors import LimitPerDrugPostProcessor
from app.services.snippets import SnippetCandidate, SnippetResult


def _make_result(
    pmid: str,
    drug: str,
    classification: Literal["risk", "safety"],
    score: float,
) -> SnippetResult:
    candidate = SnippetCandidate(
        pmid=pmid,
        drug=drug,
        classification=classification,
        snippet_text=f"{drug}-{classification}",
        article_rank=1,
        article_score=4.0,
        preferred_url="https://example.org",
        pmc_ref_count=20,
        snippet_score=score,
        cues=[classification],
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


def test_limit_per_drug_truncates_results() -> None:
    results = [
        _make_result("pmid-1", "propofol", "safety", 3.0),
        _make_result("pmid-1", "propofol", "safety", 2.5),
        _make_result("pmid-1", "ketamine", "risk", 2.0),
    ]

    processor = LimitPerDrugPostProcessor(max_per_drug=1)
    limited = processor.process(results)

    assert len(limited) == 2
    assert {result.candidate.drug for result in limited} == {"propofol", "ketamine"}

