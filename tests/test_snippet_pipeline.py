from __future__ import annotations

from pathlib import Path
from typing import Literal, Sequence

from app.services.snippet_candidates import SnippetSpan
from app.services.snippet_pipeline import (
    SnippetExtractionPipeline,
    SnippetPipelineConfig,
    SnippetPostProcessor,
)
from app.services.snippets import ArticleSnippetExtractor, SnippetCandidate, SnippetResult


def _make_result(
    *,
    pmid: str,
    drug: str,
    classification: Literal["risk", "safety"],
    snippet_text: str,
    score: float,
) -> SnippetResult:
    candidate = SnippetCandidate(
        pmid=pmid,
        drug=drug,
        classification=classification,
        snippet_text=snippet_text,
        article_rank=1,
        article_score=4.0,
        preferred_url="https://example.org",
        pmc_ref_count=20,
        snippet_score=score,
        cues=[classification],
        tags=[],
    )
    span = SnippetSpan(
        text=snippet_text,
        left=0,
        right=len(snippet_text),
        match_start=0,
        match_end=min(len(snippet_text), 5),
    )
    return SnippetResult(candidate=candidate, span=span, metadata={})


class _DummyExtractor:
    def __init__(self, results: Sequence[SnippetResult]) -> None:
        self._results = list(results)

    def extract_snippet_results(self, **_: object) -> list[SnippetResult]:
        return [
            SnippetResult(
                candidate=result.candidate,
                span=result.span,
                metadata=dict(result.metadata),
            )
            for result in self._results
        ]


class _DropDrugProcessor:
    def __init__(self, drug: str) -> None:
        self.drug = drug

    def process(self, results: Sequence[SnippetResult]) -> Sequence[SnippetResult]:
        return [result for result in results if result.candidate.drug != self.drug]


def test_pipeline_applies_quota_per_article() -> None:
    results = [
        _make_result(pmid="pmid-1", drug="succinylcholine", classification="risk", snippet_text="high risk", score=5.0),
        _make_result(pmid="pmid-1", drug="propofol", classification="safety", snippet_text="moderate safety", score=2.0),
    ]
    extractor = _DummyExtractor(results)
    pipeline = SnippetExtractionPipeline(
        extractor=extractor,
        config=SnippetPipelineConfig(base_quota=1, max_quota=1),
    )

    candidates = pipeline.run(
        article_text="",
        pmid="pmid-1",
        condition_terms=["muscular dystrophy"],
        article_rank=1,
        article_score=4.0,
        preferred_url="https://example.org",
        pmc_ref_count=20,
    )

    assert len(candidates) == 1
    assert candidates[0].drug == "succinylcholine"


def test_pipeline_invokes_post_processors() -> None:
    results = [
        _make_result(pmid="pmid-2", drug="propofol", classification="safety", snippet_text="safe use", score=3.0),
        _make_result(pmid="pmid-2", drug="ketamine", classification="risk", snippet_text="avoid use", score=2.5),
    ]
    extractor = _DummyExtractor(results)
    processor: SnippetPostProcessor = _DropDrugProcessor(drug="ketamine")
    pipeline = SnippetExtractionPipeline(
        extractor=extractor,
        post_processors=(processor,),
        config=SnippetPipelineConfig(base_quota=2, max_quota=2),
    )

    candidates = pipeline.run(
        article_text="",
        pmid="pmid-2",
        condition_terms=["muscular dystrophy"],
        article_rank=1,
        article_score=4.0,
        preferred_url="https://example.org",
        pmc_ref_count=20,
    )

    assert len(candidates) == 1
    assert candidates[0].drug == "propofol"


def test_pipeline_emits_severe_reaction_tags_from_fixture() -> None:
    article_text = Path("tests/fixtures/severe_reaction_article.txt").read_text(encoding="utf-8")
    pipeline = SnippetExtractionPipeline(
        extractor=ArticleSnippetExtractor(min_snippet_chars=40),
        config=SnippetPipelineConfig(base_quota=5, max_quota=5),
    )

    results = pipeline.run_results(
        article_text=article_text,
        pmid="fixture-1",
        condition_terms=["Dravet syndrome"],
        article_rank=1,
        article_score=6.5,
        preferred_url="https://example.org",
        pmc_ref_count=12,
    )

    assert results, "Expected at least one snippet from fixture article"

    severe_labels: set[str] = set()
    therapy_roles: set[str] = set()
    mechanism_alerts: set[str] = set()
    for result in results:
        if result.candidate.drug != "propofol":
            continue
        severe_labels.update(
            tag.label for tag in result.candidate.tags if tag.kind == "severe_reaction"
        )
        therapy_roles.update(
            result.metadata.get("therapy_roles", ())
        )
        mechanism_alerts.update(
            result.metadata.get("mechanism_alerts", ())
        )

    assert severe_labels, "Expected severe reaction tags for propofol snippet"
    assert "respiratory depression" not in severe_labels
    assert "rescue" in therapy_roles
    assert "sodium channelopathy" in mechanism_alerts
    assert any(result.metadata.get("severe_reaction_flag") for result in results)