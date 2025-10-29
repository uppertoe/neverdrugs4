from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pytest

from app.services.snippets import ArticleSnippetExtractor, SnippetCandidate
from app.services.snippet_tags import Tag

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
    assert any(isinstance(tag, Tag) and tag.kind == "risk" for tag in matching[0].tags)
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
    assert any(isinstance(tag, Tag) and tag.kind == "safety" for tag in matching[0].tags)
    snippet = matching[0]
    assert "safe" in snippet.snippet_text.lower() or "no complications" in snippet.snippet_text.lower()
    assert snippet.article_rank == 3
    assert snippet.pmc_ref_count == 26


def test_extract_snippets_labels_dantrolene_treatment_as_safety() -> None:
    text = (
        "Malignant hyperthermia is a life-threatening perioperative risk requiring rapid recognition. "
        "Prompt administration of dantrolene is the only effective treatment and should be available in every theatre. "
        "The initial dose of dantrolene reverses malignant hyperthermia crises when given without delay."
    )

    snippets = _extract(
        text,
        condition_terms=["malignant hyperthermia"],
        pmid="9990001",
        rank=1,
        score=3.2,
        pmc_ref_count=5,
    )

    dantrolene_snippets = [s for s in snippets if s.drug == "dantrolene"]
    assert dantrolene_snippets, "Expected to capture a dantrolene snippet"
    classification = {s.classification for s in dantrolene_snippets}
    assert classification == {"safety"}
    assert any(
        any(cue.startswith("therapy-role:") for cue in snippet.cues) for snippet in dantrolene_snippets
    )


def test_extract_snippets_keeps_dantrolene_toxicity_as_risk() -> None:
    text = (
        "Malignant hyperthermia preparedness remains essential, yet clinicians should avoid dantrolene in patients "
        "with severe hepatic disease because of the risk of hepatotoxicity and serious adverse events."
    )

    snippets = _extract(
        text,
        condition_terms=["malignant hyperthermia"],
        pmid="9990002",
        rank=1,
        score=2.8,
        pmc_ref_count=0,
    )

    dantrolene_snippets = [s for s in snippets if s.drug == "dantrolene"]
    assert dantrolene_snippets, "Expected to capture a dantrolene snippet"
    classification = {s.classification for s in dantrolene_snippets}
    assert classification == {"risk"}
