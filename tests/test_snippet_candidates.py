from __future__ import annotations

from app.services.snippet_candidates import RegexSnippetCandidateFinder


def test_regex_candidate_finder_emits_drug_spans() -> None:
    finder = RegexSnippetCandidateFinder(drug_terms=["propofol"], window_chars=120)

    article_text = """
    Propofol is widely used in anaesthesia. Patients with Duchenne muscular dystrophy may experience
    complications after receiving propofol, including serious adverse events that were previously
    reported in case studies.
    """

    spans = list(
        finder.find_candidates(
            article_text=article_text,
            drug="propofol",
        )
    )

    assert spans, "Expected at least one candidate window for propofol."
    text = spans[0].text.lower()
    assert "propofol" in text
    assert "duchenn" in text
