from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.models import ArticleArtefact, ArticleSnippet, SearchTerm
from app.services.snippets import (
    SnippetCandidate,
    persist_snippet_candidates,
)
from app.services.snippet_tags import Tag


def _make_article(session, *, pmid: str, rank: int, score: float) -> ArticleArtefact:
    term = SearchTerm(canonical="condition", created_at=datetime.now(timezone.utc))
    session.add(term)
    session.flush()

    artefact = ArticleArtefact(
        search_term_id=term.id,
        pmid=pmid,
        rank=rank,
        score=score,
        citation={
            "pmid": pmid,
            "preferred_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            "title": "Example",
        },
    )
    session.add(artefact)
    session.flush()
    return artefact


def test_persist_snippet_candidates_inserts_new_rows(session) -> None:
    artefact = _make_article(session, pmid="39618072", rank=1, score=4.5)

    candidates = [
        SnippetCandidate(
            pmid="39618072",
            drug="succinylcholine",
            classification="risk",
            snippet_text="Avoid succinylcholine in Duchenne muscular dystrophy.",
            article_rank=1,
            article_score=4.5,
            preferred_url="https://doi.org/10.12659/MSM.945675",
            pmc_ref_count=40,
            snippet_score=5.2,
            cues=["avoid", "malignant hyperthermia"],
            tags=[
                Tag(kind="risk", label="avoid", confidence=1.0, source="rule"),
            ],
        )
    ]

    persisted = persist_snippet_candidates(
        session,
        article_artefacts=[artefact],
        snippet_candidates=candidates,
    )

    assert len(persisted) == 1
    stored = session.query(ArticleSnippet).one()
    assert stored.drug == "succinylcholine"
    assert stored.classification == "risk"
    assert stored.snippet_score == pytest.approx(5.2)
    assert stored.cues == ["avoid", "malignant hyperthermia"]
    assert stored.tags == [
        {
            "kind": "risk",
            "label": "avoid",
            "confidence": 1.0,
            "source": "rule",
        }
    ]


def test_persist_snippet_candidates_updates_existing(session) -> None:
    artefact = _make_article(session, pmid="15859443", rank=3, score=1.1)

    initial_candidate = SnippetCandidate(
        pmid="15859443",
        drug="propofol",
        classification="safety",
        snippet_text="Propofol infusion was safe without complications.",
        article_rank=3,
        article_score=1.1,
        preferred_url="https://doi.org/10.2344/0003-3006(2005)52[12:CIPGAF]2.0.CO;2",
        pmc_ref_count=26,
        snippet_score=2.5,
        cues=["safe"],
    )
    persist_snippet_candidates(
        session,
        article_artefacts=[artefact],
        snippet_candidates=[initial_candidate],
    )

    updated_candidate = SnippetCandidate(
        pmid="15859443",
        drug="propofol",
        classification="safety",
        snippet_text="Propofol infusion was safe without complications.",
        article_rank=3,
        article_score=1.1,
        preferred_url="https://doi.org/10.2344/0003-3006(2005)52[12:CIPGAF]2.0.CO;2",
        pmc_ref_count=26,
        snippet_score=3.1,
        cues=["safe", "no complications"],
        tags=[
            Tag(kind="safety", label="safe", confidence=1.0, source="rule"),
        ],
    )
    persist_snippet_candidates(
        session,
        article_artefacts=[artefact],
        snippet_candidates=[updated_candidate],
    )

    stored = session.query(ArticleSnippet).one()
    assert stored.snippet_score == pytest.approx(3.1)
    assert stored.snippet_text.endswith("complications.")
    assert stored.cues == ["safe", "no complications"]
    assert stored.tags == [
        {
            "kind": "safety",
            "label": "safe",
            "confidence": 1.0,
            "source": "rule",
        }
    ]