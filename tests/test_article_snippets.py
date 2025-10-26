from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pytest

from app.services.snippets import ArticleSnippetExtractor, SnippetCandidate, select_top_snippets

FIXTURES = Path(__file__).parent / "fixtures"


def _load_pmc_body_text(filename: str) -> str:
    import xml.etree.ElementTree as ET

    payload = (FIXTURES / filename).read_text(encoding="utf-8")
    root = ET.fromstring(payload)
    body = root.find(".//body")
    assert body is not None, "PMC payload missing body text"
    return " ".join(body.itertext())


def _load_pubmed_abstract(filename: str) -> str:
    import xml.etree.ElementTree as ET

    payload = (FIXTURES / filename).read_text(encoding="utf-8")
    root = ET.fromstring(payload)
    abstract_nodes = root.findall(".//AbstractText")
    assert abstract_nodes, "PubMed payload missing abstract text"
    parts: list[str] = []
    for node in abstract_nodes:
        parts.append(" ".join(node.itertext()))
    return " ".join(parts)


def _extract(
    text: str,
    *,
    condition_terms: Sequence[str],
    pmid: str,
    rank: int,
    score: float,
    pmc_ref_count: int,
) -> list[SnippetCandidate]:
    extractor = ArticleSnippetExtractor(window_chars=600)
    return extractor.extract_snippets(
        article_text=text,
        pmid=pmid,
        condition_terms=condition_terms,
        article_rank=rank,
        article_score=score,
        preferred_url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        pmc_ref_count=pmc_ref_count,
    )


def test_extract_snippets_flags_succinylcholine_risk() -> None:
    text = _load_pmc_body_text("pmc_full_text_PMC11623016.xml")
    snippets = _extract(
        text,
        condition_terms=["duchenne muscular dystrophy", "muscular dystrophy"],
        pmid="39618072",
        rank=1,
        score=4.5,
        pmc_ref_count=40,
    )

    matching = [s for s in snippets if s.drug == "succinylcholine" and s.classification == "risk"]
    assert matching, "Expected to find at least one succinylcholine risk snippet"
    snippet = matching[0]
    text = snippet.snippet_text.lower()
    assert (
        "avoid" in text
        or "malignant hyperthermia" in text
        or "alternative to succinylcholine" in text
        or any("alternative" in cue for cue in snippet.cues)
    )
    assert snippet.article_rank == 1
    assert snippet.pmc_ref_count == 40


def test_extract_snippets_captures_propofol_safety() -> None:
    text = _load_pubmed_abstract("pubmed_abstract_15859443.xml")
    snippets = _extract(
        text,
        condition_terms=[
            "progressive muscular dystrophy",
            "muscular dystrophy",
            "becker muscular dystrophy",
        ],
        pmid="15859443",
        rank=3,
        score=1.1,
        pmc_ref_count=26,
    )

    matching = [s for s in snippets if s.drug == "propofol" and s.classification == "safety"]
    assert matching, "Expected propofol safety snippet"
    snippet = matching[0]
    assert "safe" in snippet.snippet_text.lower() or "no complications" in snippet.snippet_text.lower()
    assert snippet.article_rank == 3
    assert snippet.pmc_ref_count == 26


def test_select_top_snippets_scales_with_article_weight() -> None:
    candidates = [
        SnippetCandidate(
            pmid="39618072",
            drug="succinylcholine",
            classification="risk",
            snippet_text=f"snippet {idx}",
            article_rank=1,
            article_score=4.5,
            preferred_url="https://doi.org/10.12659/MSM.945675",
            pmc_ref_count=40,
            snippet_score=5.0 - idx * 0.2,
            cues=["avoided"],
        )
        for idx in range(5)
    ]

    low_weight_candidates = [
        SnippetCandidate(
            pmid="15859443",
            drug="propofol",
            classification="safety",
            snippet_text=f"low {idx}",
            article_rank=3,
            article_score=1.1,
            preferred_url="https://doi.org/10.2344/0003-3006(2005)52[12:CIPGAF]2.0.CO;2",
            pmc_ref_count=0,
            snippet_score=1.0 - idx * 0.1,
            cues=["safe"],
        )
        for idx in range(4)
    ]

    selection = select_top_snippets(candidates + low_weight_candidates)

    high_article_snippets = [c for c in selection if c.pmid == "39618072"]
    low_article_snippets = [c for c in selection if c.pmid == "15859443"]

    assert len(high_article_snippets) == 5  # reaches max quota for highly cited article
    assert len(low_article_snippets) == 2  # limited to baseline quota
    # ensure snippets are sorted by per-article score
    scores = [c.snippet_score for c in low_article_snippets]
    assert scores == sorted(scores, reverse=True)
