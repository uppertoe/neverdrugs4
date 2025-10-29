from __future__ import annotations

from pathlib import Path

import pytest

from app.services.nih_pubmed import NIHPubMedSearcher, PubMedArticle
from app.services.query_terms import ConditionTermExpansion

FIXTURES = Path(__file__).parent / "fixtures"


class _FakeResponse:
    def __init__(self, text: str, status_code: int = 200) -> None:
        self.text = text
        self.status_code = status_code


class _SequencedHttpClient:
    def __init__(self, responses: list[_FakeResponse]) -> None:
        self._responses = responses
        self.calls: list[tuple[str, dict[str, str]]] = []

    def post(self, url: str, data: dict[str, str]) -> _FakeResponse:
        if not self._responses:
            raise RuntimeError("No more responses queued")
        self.calls.append((url, data))
        return self._responses.pop(0)


def _passthrough_expander(term: str) -> ConditionTermExpansion:
    cleaned = (term or "").strip()
    if not cleaned:
        return ConditionTermExpansion(mesh_terms=(), alias_terms=())
    return ConditionTermExpansion(mesh_terms=(cleaned,), alias_terms=(cleaned,))


def test_pubmed_search_returns_ranked_articles() -> None:
    esearch_xml = (FIXTURES / "pubmed_esearch_duchenne.xml").read_text(encoding="utf-8")
    esummary_xml = (FIXTURES / "pubmed_esummary_top5.xml").read_text(encoding="utf-8")
    http_client = _SequencedHttpClient([
        _FakeResponse(esearch_xml),
        _FakeResponse(esummary_xml),
    ])
    searcher = NIHPubMedSearcher(http_client=http_client, retmax=5, condition_term_expander=_passthrough_expander)

    result = searcher([
        "Muscular Dystrophy, Duchenne",
        "Duchenne Muscular Dystrophy",
        "Duchenne-Type Progressive Muscular Dystrophy",
        "Progressive Muscular Dystrophy, Duchenne Type",
    ])

    assert result.query.startswith("(")
    assert "[mesh]" in result.query
    assert "[tiab]" in result.query
    assert len(result.articles) == 5
    assert [article.pmid for article in result.articles] == [
        "39503119",
        "30554118",
        "26070283",
        "23919455",
        "20228183",
    ]
    first = result.articles[0]
    assert isinstance(first, PubMedArticle)
    assert first.pmid == "39503119"
    assert first.title == "A Review of Muscular Dystrophies."
    assert first.preferred_url == "https://doi.org/10.2344/673191"
    assert first.doi == "10.2344/673191"
    assert first.journal == "Anesth Prog"
    assert first.authors == ["Hoang T", "Dowdy RAE"]
    assert first.has_abstract is True
    assert "Review" in first.publication_types
    assert first.pmc_ref_count == 22

    second = result.articles[1]
    assert second.pmid == "30554118"
    assert second.preferred_url == "https://doi.org/10.1016/j.biopha.2018.12.034"
    assert second.has_abstract is True
    assert second.pmc_ref_count == 0
    assert "Review" in second.publication_types

    assert http_client.calls[0][0].endswith("esearch.fcgi")
    assert http_client.calls[1][0].endswith("esummary.fcgi")
    assert "term" in http_client.calls[0][1]
    term_query = http_client.calls[0][1]["term"]
    assert http_client.calls[0][1]["db"] == "pubmed"
    assert "[tiab]" in term_query
    assert http_client.calls[1][1]["id"].startswith("39503119")


def test_pubmed_search_prefers_pmc_when_doi_missing() -> None:
    esearch_xml = """<?xml version='1.0'?><eSearchResult><Count>1</Count><RetMax>1</RetMax><RetStart>0</RetStart><IdList><Id>12345</Id></IdList></eSearchResult>"""
    esummary_xml = (FIXTURES / "pubmed_esummary_pmc_only.xml").read_text(encoding="utf-8")
    http_client = _SequencedHttpClient([
        _FakeResponse(esearch_xml),
        _FakeResponse(esummary_xml),
    ])
    searcher = NIHPubMedSearcher(http_client=http_client, retmax=1, condition_term_expander=_passthrough_expander)

    result = searcher(["Example Condition"])

    assert len(result.articles) == 1
    article = result.articles[0]
    assert article.doi is None
    assert article.pmc_id == "PMC9999999"
    assert article.preferred_url == "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9999999/"


def test_pubmed_search_falls_back_to_pubmed_when_no_preferred_ids() -> None:
    esearch_xml = """<?xml version='1.0'?><eSearchResult><Count>1</Count><RetMax>1</RetMax><RetStart>0</RetStart><IdList><Id>55555</Id></IdList></eSearchResult>"""
    esummary_xml = (FIXTURES / "pubmed_esummary_pubmed_only.xml").read_text(encoding="utf-8")
    http_client = _SequencedHttpClient([
        _FakeResponse(esearch_xml),
        _FakeResponse(esummary_xml),
    ])
    searcher = NIHPubMedSearcher(http_client=http_client, retmax=1, condition_term_expander=_passthrough_expander)

    result = searcher(["Another Example"])

    assert len(result.articles) == 1
    article = result.articles[0]
    assert article.doi is None
    assert article.pmc_id is None
    assert article.preferred_url == "https://pubmed.ncbi.nlm.nih.gov/55555/"


def test_pubmed_search_prioritises_full_text_and_citations() -> None:
    esearch_xml = """<?xml version='1.0'?><eSearchResult><Count>2</Count><RetMax>2</RetMax><RetStart>0</RetStart><IdList><Id>111</Id><Id>222</Id></IdList></eSearchResult>"""
    esummary_xml = """<?xml version='1.0'?>
<eSummaryResult>
    <DocSum>
        <Id>111</Id>
        <Item Name="Title" Type="String">Baseline Article</Item>
        <Item Name="Source" Type="String">Test Journal</Item>
        <Item Name="AuthorList" Type="List">
            <Item Name="Author" Type="String">Author A</Item>
        </Item>
        <Item Name="HasAbstract" Type="Integer">0</Item>
        <Item Name="PmcRefCount" Type="Integer">0</Item>
        <Item Name="ArticleIds" Type="List">
            <Item Name="pubmed" Type="String">111</Item>
        </Item>
    </DocSum>
    <DocSum>
        <Id>222</Id>
        <Item Name="Title" Type="String">Full Text Article</Item>
        <Item Name="Source" Type="String">Test Journal</Item>
        <Item Name="AuthorList" Type="List">
            <Item Name="Author" Type="String">Author B</Item>
        </Item>
        <Item Name="HasAbstract" Type="Integer">1</Item>
        <Item Name="PmcRefCount" Type="Integer">10</Item>
        <Item Name="PubTypeList" Type="List">
            <Item Name="PubType" Type="String">Review</Item>
        </Item>
        <Item Name="ArticleIds" Type="List">
            <Item Name="pubmed" Type="String">222</Item>
            <Item Name="pmc" Type="String">PMC12345</Item>
        </Item>
    </DocSum>
</eSummaryResult>
"""
    http_client = _SequencedHttpClient([
        _FakeResponse(esearch_xml),
        _FakeResponse(esummary_xml),
    ])
    searcher = NIHPubMedSearcher(
        http_client=http_client,
        retmax=2,
        condition_term_expander=_passthrough_expander,
    )

    result = searcher(["Example Condition"])

    assert [article.pmid for article in result.articles] == ["222", "111"]
    ranked_first = result.articles[0]
    assert ranked_first.pmc_id == "PMC12345"
    assert ranked_first.has_abstract is True
    assert ranked_first.pmc_ref_count == 10
    assert ranked_first.score > result.articles[1].score