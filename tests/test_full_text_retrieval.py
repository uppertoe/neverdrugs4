from __future__ import annotations

from pathlib import Path

import pytest

from app.models import ArticleArtefact, ArticleSnippet, SearchTerm
from app.services.full_text import (
    FullTextSelectionPolicy,
    NIHFullTextFetcher,
    collect_pubmed_articles,
)
from app.services.nih_pubmed import PubMedArticle, PubMedSearchResult
from app.services.search import SearchResolution


FIXTURES = Path(__file__).parent / "fixtures"


class _StubHttpResponse:
    def __init__(self, text: str, status_code: int = 200) -> None:
        self.text = text
        self.status_code = status_code


class _SequencedHttpClient:
    def __init__(self, responses: list[_StubHttpResponse]) -> None:
        self._responses = responses
        self.calls: list[dict[str, str]] = []

    def get(self, url: str, params: dict[str, str]) -> _StubHttpResponse:
        if not self._responses:
            raise RuntimeError("No more responses queued")
        self.calls.append(params)
        return self._responses.pop(0)


class _FixtureHttpClient:
    def __init__(self, mapping: dict[tuple[str, str], _StubHttpResponse]) -> None:
        self._mapping = mapping
        self.calls: list[dict[str, str]] = []

    def get(self, url: str, params: dict[str, str]) -> _StubHttpResponse:
        key = (params.get("db"), params.get("id"))
        self.calls.append(params)
        if key in self._mapping:
            response = self._mapping[key]
            return _StubHttpResponse(response.text, response.status_code)

        db, ids = key
        if db == "pmc" and ids and "," in ids:
            from copy import deepcopy
            from xml.etree import ElementTree

            aggregate_root = ElementTree.Element("pmc-articleset")
            for single_id in ids.split(","):
                single_key = (db, single_id)
                if single_key not in self._mapping:
                    raise AssertionError(f"Unexpected NIH request {single_key}")
                source = self._mapping[single_key]
                article_root = ElementTree.fromstring(source.text)
                article = article_root.find("article")
                if article is None:
                    raise AssertionError(f"PMC fixture missing article node for {single_id}")
                aggregate_root.append(deepcopy(article))

            payload = ElementTree.tostring(aggregate_root, encoding="unicode")
            return _StubHttpResponse(payload)

        if db == "pubmed" and ids and "," in ids:
            from copy import deepcopy
            from xml.etree import ElementTree

            aggregate_root = ElementTree.Element("PubmedArticleSet")
            for single_id in ids.split(","):
                single_key = (db, single_id)
                if single_key not in self._mapping:
                    raise AssertionError(f"Unexpected NIH request {single_key}")
                source = self._mapping[single_key]
                article_root = ElementTree.fromstring(source.text)
                article = article_root.find("PubmedArticle")
                if article is None:
                    raise AssertionError(f"PubMed fixture missing article node for {single_id}")
                aggregate_root.append(deepcopy(article))

            payload = ElementTree.tostring(aggregate_root, encoding="unicode")
            return _StubHttpResponse(payload)

        raise AssertionError(f"Unexpected NIH request {key}")


def _make_article(
    *,
    pmid: str,
    score: float,
    pmc_id: str | None = None,
    has_abstract: bool = True,
    pmc_ref_count: int = 0,
) -> PubMedArticle:
    return PubMedArticle(
        pmid=pmid,
        title=f"Title {pmid}",
        journal="Journal",
        publication_date="2024",
        authors=["Author"],
        publication_types=["Journal Article"],
        has_abstract=has_abstract,
        pmc_ref_count=pmc_ref_count,
        doi=None,
        pmc_id=pmc_id,
        preferred_url=f"https://example.org/{pmid}",
        score=score,
    )


def test_full_text_selection_respects_maximum_limit() -> None:
    policy = FullTextSelectionPolicy(
        base_full_text=1,
        max_full_text=2,
        bonus_score_threshold=2.0,
    )
    articles = [
        _make_article(pmid="1", score=3.0, pmc_id="PMC1"),
        _make_article(pmid="2", score=2.5, pmc_id=None),
        _make_article(pmid="3", score=1.8, pmc_id="PMC3"),
    ]

    selected = policy.select(articles)

    assert [article.pmid for article in selected] == ["1", "2"]


def test_full_text_fetcher_prefers_pmc_payload() -> None:
    pmc_xml = (FIXTURES / "pmc_full_text_PMC11623016.xml").read_text(encoding="utf-8")
    http_client = _SequencedHttpClient([
        _StubHttpResponse(pmc_xml),
    ])
    fetcher = NIHFullTextFetcher(http_client=http_client)
    article = PubMedArticle(
        pmid="39618072",
        title="A Review of Muscle Relaxants in Anesthesia in Patients with Neuromuscular Disorders Including Guillain-Barré Syndrome, Myasthenia Gravis, Duchenne Muscular Dystrophy, Charcot-Marie-Tooth Disease, and Inflammatory Myopathies.",
        journal="Med Sci Monit",
        publication_date="2024 Dec 2",
        authors=["Saito H"],
        publication_types=["Journal Article", "Review"],
        has_abstract=True,
        pmc_ref_count=40,
        doi="10.12659/MSM.945675",
        pmc_id="PMC11623016",
        preferred_url="https://doi.org/10.12659/MSM.945675",
        score=4.2,
    )

    result = fetcher.fetch_many([article])
    assert article.pmid in result
    content = result[article.pmid]

    assert content.source == "pmc-full-text"
    assert "Neuromuscular diseases" in content.text
    assert content.token_estimate > 500
    assert http_client.calls[0]["db"] == "pmc"
    assert http_client.calls[0]["id"] == "PMC11623016"


def test_full_text_fetcher_falls_back_to_pubmed_when_pmc_fails() -> None:
    pmc_xml = (FIXTURES / "pmc_full_text_PMC2526213.xml").read_text(encoding="utf-8")
    abstract_xml = (FIXTURES / "pubmed_abstract_15859443.xml").read_text(encoding="utf-8")
    http_client = _SequencedHttpClient([
        _StubHttpResponse(pmc_xml, status_code=200),
        _StubHttpResponse(abstract_xml),
    ])
    fetcher = NIHFullTextFetcher(http_client=http_client)
    article = PubMedArticle(
        pmid="15859443",
        title="Continuous infusion propofol general anesthesia for dental treatment in patients with progressive muscular dystrophy.",
        journal="Anesth Prog",
        publication_date="2005 Spring",
        authors=["Iida H"],
        publication_types=["Journal Article"],
        has_abstract=True,
        pmc_ref_count=26,
        doi="10.2344/0003-3006(2005)52[12:CIPGAF]2.0.CO;2",
        pmc_id="PMC2526213",
        preferred_url="https://doi.org/10.2344/0003-3006(2005)52[12:CIPGAF]2.0.CO;2",
        score=2.1,
    )

    result = fetcher.fetch_many([article])
    assert article.pmid in result
    content = result[article.pmid]

    assert content.source == "pubmed-abstract"
    assert "continuous infusion of propofol" in content.text.lower()
    assert http_client.calls[0]["db"] == "pmc"
    assert http_client.calls[0]["id"] == "PMC2526213"
    assert http_client.calls[1]["db"] == "pubmed"
    assert http_client.calls[1]["id"] == "15859443"


def test_collect_pubmed_articles_persists_metadata_and_full_text(session) -> None:
    term = SearchTerm(canonical="condition")
    session.add(term)
    session.flush()

    articles = [
        PubMedArticle(
            pmid="39618072",
            title="A Review of Muscle Relaxants in Anesthesia in Patients with Neuromuscular Disorders Including Guillain-Barré Syndrome, Myasthenia Gravis, Duchenne Muscular Dystrophy, Charcot-Marie-Tooth Disease, and Inflammatory Myopathies.",
            journal="Med Sci Monit",
            publication_date="2024 Dec 2",
            authors=["Saito H"],
            publication_types=["Journal Article", "Review"],
            has_abstract=True,
            pmc_ref_count=40,
            doi="10.12659/MSM.945675",
            pmc_id="PMC11623016",
            preferred_url="https://doi.org/10.12659/MSM.945675",
            score=4.5,
        ),
        PubMedArticle(
            pmid="23919455",
            title="Anesthesia and Duchenne or Becker muscular dystrophy: review of 117 anesthetic exposures.",
            journal="Paediatr Anaesth",
            publication_date="2013 Sep",
            authors=["Segura LG"],
            publication_types=["Journal Article"],
            has_abstract=True,
            pmc_ref_count=0,
            doi="10.1111/pan.12248",
            pmc_id=None,
            preferred_url="https://doi.org/10.1111/pan.12248",
            score=3.2,
        ),
        PubMedArticle(
            pmid="15859443",
            title="Continuous infusion propofol general anesthesia for dental treatment in patients with progressive muscular dystrophy.",
            journal="Anesth Prog",
            publication_date="2005 Spring",
            authors=["Iida H"],
            publication_types=["Journal Article"],
            has_abstract=True,
            pmc_ref_count=26,
            doi="10.2344/0003-3006(2005)52[12:CIPGAF]2.0.CO;2",
            pmc_id="PMC2526213",
            preferred_url="https://doi.org/10.2344/0003-3006(2005)52[12:CIPGAF]2.0.CO;2",
            score=1.1,
        ),
    ]
    search_result = PubMedSearchResult(
        query="query",
        pmids=[article.pmid for article in articles],
        articles=articles,
        raw_esearch="es",
        raw_esummary="sum",
    )

    class _StubSearcher:
        def __call__(self, mesh_terms):
            return search_result

    fixture_http_client = _FixtureHttpClient(
        {
            ("pmc", "PMC11623016"): _StubHttpResponse(
                (FIXTURES / "pmc_full_text_PMC11623016.xml").read_text(encoding="utf-8")
            ),
            ("pubmed", "23919455"): _StubHttpResponse(
                (FIXTURES / "pubmed_abstract_23919455.xml").read_text(encoding="utf-8")
            ),
                ("pubmed", "15859443"): _StubHttpResponse(
                    (FIXTURES / "pubmed_abstract_15859443.xml").read_text(encoding="utf-8")
                ),
            ("pmc", "PMC2526213"): _StubHttpResponse(
                (FIXTURES / "pmc_full_text_PMC2526213.xml").read_text(encoding="utf-8")
            ),
        }
    )
    full_text_fetcher = NIHFullTextFetcher(http_client=fixture_http_client)

    selection_policy = FullTextSelectionPolicy(
        base_full_text=3,
        max_full_text=3,
        bonus_score_threshold=1.0,
        prefer_pmc=False,
    )

    resolution = SearchResolution(
        normalized_condition="condition",
        mesh_terms=["Term"],
        reused_cached=False,
        search_term_id=term.id,
    )

    collect_pubmed_articles(
        resolution,
        session=session,
        pubmed_searcher=_StubSearcher(),
        full_text_fetcher=full_text_fetcher,
        selection_policy=selection_policy,
    )

    stored = session.query(ArticleArtefact).order_by(ArticleArtefact.rank).all()
    assert len(stored) == 3

    first = stored[0]
    assert first.pmid == "39618072"
    assert first.score == pytest.approx(4.5)
    assert first.content_source == "pmc-full-text"
    assert "Neuromuscular diseases" in first.content
    assert first.citation["preferred_url"] == "https://doi.org/10.12659/MSM.945675"

    second = stored[1]
    assert second.pmid == "23919455"
    assert second.content_source == "pubmed-abstract"
    assert "duchenne muscular dystrophy (dmd)" in second.content.lower()

    third = stored[2]
    assert third.pmid == "15859443"
    assert third.content_source in {"pmc-full-text", "pubmed-abstract"}
    assert third.content
    assert "propofol" in third.content.lower()
    assert third.citation["score"] == pytest.approx(1.1)
    assert fixture_http_client.calls[0]["db"] == "pmc"
    assert "PMC11623016" in fixture_http_client.calls[0]["id"]
    assert fixture_http_client.calls[1]["db"] == "pubmed"
    assert fixture_http_client.calls[1]["id"] == "23919455,15859443"

    snippet_rows = session.query(ArticleSnippet).order_by(ArticleSnippet.id).all()
    assert len(snippet_rows) >= 2
    risk_snippet = next(snippet for snippet in snippet_rows if snippet.drug == "succinylcholine")
    assert risk_snippet.classification == "risk"
    assert "succinylcholine" in risk_snippet.snippet_text.lower()
    propofol_snippet = next(snippet for snippet in snippet_rows if snippet.drug == "propofol")
    assert "propofol" in propofol_snippet.snippet_text.lower()
    assert "no complications" in propofol_snippet.snippet_text.lower()


def test_collect_pubmed_articles_fetches_abstracts_for_unselected_articles(session) -> None:
    term = SearchTerm(canonical="condition")
    session.add(term)
    session.flush()

    articles = [
        PubMedArticle(
            pmid="39618072",
            title="A Review of Muscle Relaxants in Anesthesia in Patients with Neuromuscular Disorders Including Guillain-Barré Syndrome, Myasthenia Gravis, Duchenne Muscular Dystrophy, Charcot-Marie-Tooth Disease, and Inflammatory Myopathies.",
            journal="Med Sci Monit",
            publication_date="2024 Dec 2",
            authors=["Saito H"],
            publication_types=["Journal Article", "Review"],
            has_abstract=True,
            pmc_ref_count=40,
            doi="10.12659/MSM.945675",
            pmc_id="PMC11623016",
            preferred_url="https://doi.org/10.12659/MSM.945675",
            score=5.2,
        ),
        PubMedArticle(
            pmid="23919455",
            title="Anesthesia and Duchenne or Becker muscular dystrophy: review of 117 anesthetic exposures.",
            journal="Paediatr Anaesth",
            publication_date="2013 Sep",
            authors=["Segura LG"],
            publication_types=["Journal Article"],
            has_abstract=True,
            pmc_ref_count=0,
            doi="10.1111/pan.12248",
            pmc_id=None,
            preferred_url="https://doi.org/10.1111/pan.12248",
            score=3.2,
        ),
        PubMedArticle(
            pmid="15859443",
            title="Continuous infusion propofol general anesthesia for dental treatment in patients with progressive muscular dystrophy.",
            journal="Anesth Prog",
            publication_date="2005 Spring",
            authors=["Iida H"],
            publication_types=["Journal Article"],
            has_abstract=True,
            pmc_ref_count=26,
            doi="10.2344/0003-3006(2005)52[12:CIPGAF]2.0.CO;2",
            pmc_id="PMC2526213",
            preferred_url="https://doi.org/10.2344/0003-3006(2005)52[12:CIPGAF]2.0.CO;2",
            score=2.8,
        ),
    ]

    search_result = PubMedSearchResult(
        query="query",
        pmids=[article.pmid for article in articles],
        articles=articles,
        raw_esearch="es",
        raw_esummary="sum",
    )

    class _StubSearcher:
        def __call__(self, mesh_terms):
            return search_result

    fixture_http_client = _FixtureHttpClient(
        {
            ("pmc", "PMC11623016"): _StubHttpResponse(
                (FIXTURES / "pmc_full_text_PMC11623016.xml").read_text(encoding="utf-8")
            ),
            ("pmc", "PMC2526213"): _StubHttpResponse(
                (FIXTURES / "pmc_full_text_PMC2526213.xml").read_text(encoding="utf-8")
            ),
            ("pubmed", "23919455"): _StubHttpResponse(
                (FIXTURES / "pubmed_abstract_23919455.xml").read_text(encoding="utf-8")
            ),
            ("pubmed", "15859443"): _StubHttpResponse(
                (FIXTURES / "pubmed_abstract_15859443.xml").read_text(encoding="utf-8")
            ),
        }
    )
    full_text_fetcher = NIHFullTextFetcher(http_client=fixture_http_client)

    selection_policy = FullTextSelectionPolicy(
        base_full_text=1,
        max_full_text=1,
        bonus_score_threshold=10.0,
    )

    resolution = SearchResolution(
        normalized_condition="condition",
        mesh_terms=["Term"],
        reused_cached=False,
        search_term_id=term.id,
    )

    collect_pubmed_articles(
        resolution,
        session=session,
        pubmed_searcher=_StubSearcher(),
        full_text_fetcher=full_text_fetcher,
        selection_policy=selection_policy,
    )

    stored = session.query(ArticleArtefact).order_by(ArticleArtefact.rank).all()
    assert len(stored) == 3

    second = stored[1]
    assert second.pmid == "23919455"
    assert second.content_source == "pubmed-abstract"
    assert "duchenne muscular dystrophy" in second.content.lower()

    third = stored[2]
    assert third.pmid == "15859443"
    assert third.content_source == "pubmed-abstract"
    assert "propofol" in third.content.lower()

    # Ensure we called PubMed fetch for the remaining articles in a single batch
    pubmed_calls = [call for call in fixture_http_client.calls if call.get("db") == "pubmed"]
    assert pubmed_calls, "Expected at least one PubMed abstract fetch"
    combined_ids = {call["id"] for call in pubmed_calls}
    assert any("23919455" in ids and "15859443" in ids for ids in combined_ids)