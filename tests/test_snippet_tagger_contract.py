from __future__ import annotations

from collections.abc import Sequence

from app.services.snippet_tags import Tag
from app.services.snippets import ArticleSnippetExtractor


def test_snippet_tagger_protocol_accepts_custom_implementations() -> None:
    # Import within the test to avoid circular import issues during discovery.
    from app.services.snippet_tagger import SnippetTagger

    class DummyTagger:
        def tag_snippet(
            self,
            snippet_text: str,
            *,
            drug: str,
            condition_terms: Sequence[str],
        ) -> list[Tag]:
            return [Tag(kind="risk", label="dummy", confidence=0.5, source="dummy")]

    dummy_tagger = DummyTagger()
    assert isinstance(dummy_tagger, SnippetTagger)

    extractor = ArticleSnippetExtractor(tagger=dummy_tagger)

    article_text = (
        "Propofol should be avoided in Duchenne muscular dystrophy patients because of prior "
        "reports describing serious adverse events in this population."
    )

    snippets = extractor.extract_snippets(
        article_text=article_text,
        pmid="1",
        condition_terms=["Duchenne muscular dystrophy"],
        article_rank=1,
        article_score=5.0,
        preferred_url="https://example.org",
        pmc_ref_count=5,
    )

    assert snippets, "Extractor should return a snippet when cue and condition match."
    assert snippets[0].tags == [Tag(kind="risk", label="dummy", confidence=0.5, source="dummy")]
