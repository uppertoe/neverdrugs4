from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence
from xml.etree import ElementTree

import httpx

from app.services.query_terms import build_nih_search_query
from app.settings import DEFAULT_PUBMED_RETMAX

DEFAULT_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
DEFAULT_TIMEOUT_SECONDS = 10.0
DEFAULT_RETMAX = DEFAULT_PUBMED_RETMAX
FULL_TEXT_BOOST = 2.0
ABSTRACT_BOOST = 0.4
REVIEW_BOOST = 0.2
CITATION_WEIGHT = 0.02
CITATION_CAP = 200
BASELINE_DECAY = 0.05
REVIEW_LABELS = {
    "Review",
    "Systematic Review",
    "Meta-Analysis",
}


@dataclass(slots=True)
class PubMedArticle:
    pmid: str
    title: str
    journal: Optional[str]
    publication_date: Optional[str]
    authors: list[str]
    publication_types: list[str]
    has_abstract: bool
    pmc_ref_count: int
    doi: Optional[str]
    pmc_id: Optional[str]
    preferred_url: str
    score: float = 0.0

    def to_citation_dict(self) -> dict[str, Any]:
        return {
            "pmid": self.pmid,
            "title": self.title,
            "journal": self.journal,
            "publication_date": self.publication_date,
            "authors": list(self.authors),
            "publication_types": list(self.publication_types),
            "has_abstract": self.has_abstract,
            "pmc_ref_count": self.pmc_ref_count,
            "doi": self.doi,
            "pmc_id": self.pmc_id,
            "preferred_url": self.preferred_url,
            "score": self.score,
        }


@dataclass(slots=True)
class PubMedSearchResult:
    query: str
    pmids: list[str]
    articles: list[PubMedArticle]
    raw_esearch: str
    raw_esummary: str


class NIHPubMedSearcher:
    def __init__(
        self,
        *,
        http_client: Optional[httpx.Client] = None,
        base_url: str = DEFAULT_BASE_URL,
        database: str = "pubmed",
        retmax: int = DEFAULT_RETMAX,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self._http_client = http_client
        self.base_url = base_url.rstrip("/")
        self.database = database
        self.retmax = retmax
        self.timeout_seconds = timeout_seconds

    def __call__(
        self,
        condition_mesh_terms: Sequence[str],
        *,
        additional_text_terms: Iterable[str] | None = None,
    ) -> PubMedSearchResult:
        query = build_nih_search_query(
            condition_mesh_terms,
            additional_text_terms=additional_text_terms,
        )
        esearch_response = self._post(
            "esearch.fcgi",
            {
                "db": self.database,
                "retmax": str(self.retmax),
                "sort": "relevance",
                "term": query,
            },
        )
        esearch_payload = esearch_response.text
        pmids = _parse_esearch_ids(esearch_payload)

        if not pmids:
            return PubMedSearchResult(
                query=query,
                pmids=[],
                articles=[],
                raw_esearch=esearch_payload,
                raw_esummary="",
            )

        esummary_response = self._post(
            "esummary.fcgi",
            {
                "db": self.database,
                "id": ",".join(pmids),
            },
        )
        esummary_payload = esummary_response.text
        articles_by_pmid = _parse_esummary(esummary_payload)

        scored_articles: list[tuple[float, int, PubMedArticle]] = []
        for position, pmid in enumerate(pmids):
            article = articles_by_pmid.get(pmid)
            if article is None:
                continue
            score = _score_article(article, baseline_rank=position)
            article.score = score
            scored_articles.append((score, position, article))

        scored_articles.sort(key=lambda item: (-item[0], item[1]))
        ordered_articles: list[PubMedArticle] = [entry[2] for entry in scored_articles]
        ordered_pmids = [article.pmid for article in ordered_articles]

        return PubMedSearchResult(
            query=query,
            pmids=ordered_pmids,
            articles=ordered_articles,
            raw_esearch=esearch_payload,
            raw_esummary=esummary_payload,
        )

    def _post(self, endpoint: str, data: dict[str, str]) -> httpx.Response:
        url = f"{self.base_url}/{endpoint}"
        if self._http_client is not None:
            response = self._http_client.post(url, data=data)
        else:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                response = client.post(url, data=data)
        if response.status_code >= 400:
            raise RuntimeError(
                f"NIH request failed with status {response.status_code} for {endpoint}"
            )
        return response


def _parse_esearch_ids(xml_payload: str) -> list[str]:
    try:
        root = ElementTree.fromstring(xml_payload)
    except ElementTree.ParseError as exc:
        raise RuntimeError("Failed to parse PubMed eSearch response") from exc

    ids = [node.text.strip() for node in root.findall(".//IdList/Id") if node.text]
    return ids


def _parse_esummary(xml_payload: str) -> dict[str, PubMedArticle]:
    try:
        root = ElementTree.fromstring(xml_payload)
    except ElementTree.ParseError as exc:
        raise RuntimeError("Failed to parse PubMed eSummary response") from exc

    articles: dict[str, PubMedArticle] = {}
    for docsum in root.findall("DocSum"):
        pmid = _find_text(docsum, "Id")
        if not pmid:
            continue

        title = _find_item_text(docsum, "Title") or ""
        journal = _find_item_text(docsum, "Source")
        pub_date = _find_item_text(docsum, "PubDate") or _find_item_text(docsum, "EPubDate")
        authors = _find_item_list(docsum, "AuthorList")
        publication_types = _find_item_list(docsum, "PubTypeList")
        article_ids = _find_article_ids(docsum)
        has_abstract_flag = _find_item_int(docsum, "HasAbstract")
        pmc_ref_count = _find_item_int(docsum, "PmcRefCount") or 0
        has_abstract = bool(has_abstract_flag)

        doi = article_ids.get("doi")
        pmc_id = article_ids.get("pmc")
        preferred_url = _determine_preferred_url(pmid, doi, pmc_id)

        articles[pmid] = PubMedArticle(
            pmid=pmid,
            title=title,
            journal=journal,
            publication_date=pub_date,
            authors=authors,
            publication_types=publication_types,
            has_abstract=has_abstract,
            pmc_ref_count=pmc_ref_count,
            doi=doi,
            pmc_id=pmc_id,
            preferred_url=preferred_url,
        )

    return articles


def _find_text(element: ElementTree.Element, tag: str) -> Optional[str]:
    node = element.find(tag)
    if node is None or node.text is None:
        return None
    value = node.text.strip()
    return value or None


def _find_item_text(docsum: ElementTree.Element, name: str) -> Optional[str]:
    node = docsum.find(f"Item[@Name='{name}']")
    if node is None or node.text is None:
        return None
    value = node.text.strip()
    return value or None


def _find_item_list(docsum: ElementTree.Element, name: str) -> list[str]:
    node = docsum.find(f"Item[@Name='{name}']")
    if node is None:
        return []
    values: list[str] = []
    for child in node.findall("Item"):
        if child.text:
            stripped = child.text.strip()
            if stripped:
                values.append(stripped)
    return values


def _find_item_int(docsum: ElementTree.Element, name: str) -> Optional[int]:
    value = _find_item_text(docsum, name)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _find_article_ids(docsum: ElementTree.Element) -> dict[str, str]:
    article_ids: dict[str, str] = {}
    node = docsum.find("Item[@Name='ArticleIds']")
    if node is None:
        return article_ids
    for child in node.findall("Item"):
        key = child.get("Name")
        if not key or child.text is None:
            continue
        value = child.text.strip()
        if value:
            article_ids[key] = value
    return article_ids


def _score_article(article: PubMedArticle, *, baseline_rank: int) -> float:
    baseline = max(0.0, 1.0 - baseline_rank * BASELINE_DECAY)
    score = baseline
    if article.pmc_id:
        score += FULL_TEXT_BOOST
    elif article.has_abstract:
        score += ABSTRACT_BOOST
    if article.pmc_ref_count:
        citation_boost = min(article.pmc_ref_count, CITATION_CAP) * CITATION_WEIGHT
        score += citation_boost
    if any(pub_type in REVIEW_LABELS for pub_type in article.publication_types):
        score += REVIEW_BOOST
    return score


def _determine_preferred_url(pmid: str, doi: Optional[str], pmc_id: Optional[str]) -> str:
    if doi:
        normalized = doi.replace("doi:", "").strip()
        return f"https://doi.org/{normalized}"
    if pmc_id:
        normalized = pmc_id.strip()
        return f"https://www.ncbi.nlm.nih.gov/pmc/articles/{normalized}/"
    return f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
