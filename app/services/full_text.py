from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Dict, Optional, Sequence
from xml.etree import ElementTree

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import ArticleArtefact
from app.services.nih_pubmed import (
    DEFAULT_BASE_URL,
    DEFAULT_TIMEOUT_SECONDS,
    NIHPubMedSearcher,
    PubMedArticle,
)
from app.services.search import SearchResolution
from app.services.snippets import (
    ArticleSnippetExtractor,
    SnippetCandidate,
    persist_snippet_candidates,
    select_top_snippets,
)
from app.settings import (
    DEFAULT_BASE_FULL_TEXT_ARTICLES,
    DEFAULT_ESTIMATED_TOKENS_PER_ARTICLE,
    DEFAULT_FULL_TEXT_TOKEN_BUDGET,
    DEFAULT_MAX_FULL_TEXT_ARTICLES,
)


@dataclass(slots=True)
class FullTextSelectionPolicy:
    base_full_text: int = DEFAULT_BASE_FULL_TEXT_ARTICLES
    max_full_text: int = DEFAULT_MAX_FULL_TEXT_ARTICLES
    max_token_budget: int = DEFAULT_FULL_TEXT_TOKEN_BUDGET
    estimated_tokens_per_full_text: int = DEFAULT_ESTIMATED_TOKENS_PER_ARTICLE
    bonus_score_threshold: float = 1.5
    require_score_cutoff: float = 0.0
    prefer_pmc: bool = True

    def select(self, articles: Sequence[PubMedArticle]) -> list[PubMedArticle]:
        if not articles or self.max_full_text <= 0 or self.max_token_budget <= 0:
            return []

        max_candidates = min(len(articles), self.max_full_text)
        if max_candidates == 0:
            return []

        if self.estimated_tokens_per_full_text <= 0:
            allowed_by_budget = max_candidates
        else:
            allowed_by_budget = self.max_token_budget // self.estimated_tokens_per_full_text

        initial_count = min(self.base_full_text, max_candidates, allowed_by_budget)
        if initial_count == 0 and self.base_full_text > 0 and self.max_token_budget > 0:
            initial_count = min(1, max_candidates)

        selected_indices: set[int] = set(range(initial_count))
        total_tokens = initial_count * max(1, self.estimated_tokens_per_full_text)
        max_indices = min(self.max_full_text, len(articles))

        for idx in range(initial_count, len(articles)):
            if len(selected_indices) >= max_indices:
                break
            article = articles[idx]
            if article.score < self.require_score_cutoff:
                continue
            projected_tokens = total_tokens + max(1, self.estimated_tokens_per_full_text)
            if projected_tokens > self.max_token_budget:
                break
            if self.prefer_pmc and article.pmc_id:
                selected_indices.add(idx)
                total_tokens = projected_tokens
                continue
            if article.score >= self.bonus_score_threshold:
                selected_indices.add(idx)
                total_tokens = projected_tokens

        ordered_indices = sorted(selected_indices)
        return [articles[idx] for idx in ordered_indices]


@dataclass(slots=True)
class ArticleContent:
    pmid: str
    text: str
    source: str
    token_estimate: int
    fetched_at: datetime


class NIHFullTextFetcher:
    def __init__(
        self,
        *,
        http_client: Optional[httpx.Client] = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self._http_client = http_client
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def fetch_many(self, articles: Sequence[PubMedArticle]) -> dict[str, ArticleContent]:
        if not articles:
            return {}

        results: dict[str, ArticleContent] = {}
        pmc_candidates = [article for article in articles if article.pmc_id]
        if pmc_candidates:
            try:
                results.update(self._fetch_pmc_batch(pmc_candidates))
            except RuntimeError:
                # Fall back to abstracts for all PMC candidates if batch fails entirely
                pmc_candidates = []

        remaining = [article for article in articles if article.pmid not in results]
        if remaining:
            results.update(self._fetch_pubmed_batch(remaining))

        return results

    def fetch(self, article: PubMedArticle) -> ArticleContent:
        # Maintain single-article API for backward compatibility by delegating to fetch_many
        batch = self.fetch_many([article])
        if article.pmid not in batch:
            raise RuntimeError(f"Failed to fetch content for PMID {article.pmid}")
        return batch[article.pmid]

    def fetch_abstracts(self, articles: Sequence[PubMedArticle]) -> dict[str, ArticleContent]:
        if not articles:
            return {}

        return self._fetch_pubmed_batch(articles)

    def _fetch_pmc_batch(self, articles: Sequence[PubMedArticle]) -> dict[str, ArticleContent]:
        params = {
            "db": "pmc",
            "id": ",".join(article.pmc_id for article in articles if article.pmc_id),
            "retmode": "xml",
            "rettype": "xml",
        }
        response = self._request(params)
        payload = response.text
        contents = self._parse_pmc_payload(payload)

        results: dict[str, ArticleContent] = {}
        for article in articles:
            if article.pmid not in contents:
                continue
            text = contents[article.pmid]
            results[article.pmid] = ArticleContent(
                pmid=article.pmid,
                text=text,
                source="pmc-full-text",
                token_estimate=_estimate_token_count(text),
                fetched_at=datetime.now(timezone.utc),
            )
        return results

    def _fetch_pubmed_batch(self, articles: Sequence[PubMedArticle]) -> dict[str, ArticleContent]:
        params = {
            "db": "pubmed",
            "id": ",".join(article.pmid for article in articles),
            "rettype": "abstract",
            "retmode": "xml",
        }
        response = self._request(params)
        payload = response.text
        parsed = self._parse_pubmed_payload(payload)

        results: dict[str, ArticleContent] = {}
        for article in articles:
            text = parsed.get(article.pmid)
            if not text:
                continue
            results[article.pmid] = ArticleContent(
                pmid=article.pmid,
                text=text,
                source="pubmed-abstract",
                token_estimate=_estimate_token_count(text),
                fetched_at=datetime.now(timezone.utc),
            )
        return results

    def _request(self, params: dict[str, str]) -> httpx.Response:
        url = f"{self.base_url}/efetch.fcgi"
        if self._http_client is not None:
            response = self._http_client.get(url, params=params)
        else:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                response = client.get(url, params=params)
        if response.status_code >= 400:
            raise RuntimeError(
                f"NIH full-text request failed with status {response.status_code}"
            )
        return response

    def _parse_pmc_payload(self, payload: str) -> dict[str, str]:
        try:
            root = ElementTree.fromstring(payload)
        except ElementTree.ParseError as exc:
            raise RuntimeError("Failed to parse PMC full-text payload") from exc

        contents: dict[str, str] = {}
        for article in root.findall("article"):
            pmid_node = article.find(".//article-id[@pub-id-type='pmid']")
            if pmid_node is None or not pmid_node.text:
                continue
            body = article.find(".//body")
            if body is None:
                continue
            text = _normalize_whitespace(" ".join(body.itertext()))
            if not text:
                continue
            contents[pmid_node.text.strip()] = text
        return contents

    def _parse_pubmed_payload(self, payload: str) -> dict[str, str]:
        try:
            root = ElementTree.fromstring(payload)
        except ElementTree.ParseError as exc:
            raise RuntimeError("Failed to parse PubMed abstract payload") from exc

        contents: dict[str, str] = {}
        for article in root.findall("PubmedArticle"):
            pmid_node = article.find(".//PMID")
            if pmid_node is None or pmid_node.text is None:
                continue
            abstract_texts = []
            for abstract in article.findall(".//AbstractText"):
                if abstract.text:
                    abstract_texts.append(abstract.text.strip())
            if not abstract_texts:
                continue
            contents[pmid_node.text.strip()] = _normalize_whitespace(" ".join(abstract_texts))
        return contents


def collect_pubmed_articles(
    search_resolution: SearchResolution,
    *,
    session: Session,
    pubmed_searcher: NIHPubMedSearcher,
    full_text_fetcher: NIHFullTextFetcher,
    selection_policy: FullTextSelectionPolicy | None = None,
    snippet_extractor: ArticleSnippetExtractor | None = None,
    snippet_selector: Callable[[Sequence[SnippetCandidate]], list[SnippetCandidate]] | None = None,
) -> list[ArticleArtefact]:
    policy = selection_policy or FullTextSelectionPolicy()
    search_result = pubmed_searcher(search_resolution.mesh_terms)
    selected = policy.select(search_result.articles)

    contents: dict[str, ArticleContent] = {}
    if selected:
        try:
            contents = full_text_fetcher.fetch_many(selected)
        except RuntimeError:
            contents = {}

    abstract_candidates = [
        article for article in search_result.articles if article.pmid not in contents
    ]
    if abstract_candidates:
        try:
            abstract_contents = full_text_fetcher.fetch_abstracts(abstract_candidates)
        except RuntimeError:
            abstract_contents = {}
        else:
            for pmid, content in abstract_contents.items():
                contents.setdefault(pmid, content)

    persisted_articles = persist_pubmed_articles(
        session,
        search_term_id=search_resolution.search_term_id,
        articles=search_result.articles,
        contents=contents,
    )

    extractor = snippet_extractor or ArticleSnippetExtractor()
    condition_terms = list(search_resolution.mesh_terms)
    if search_resolution.normalized_condition:
        condition_terms.append(search_resolution.normalized_condition)

    snippet_candidates: list[SnippetCandidate] = []
    for artefact in persisted_articles:
        content = contents.get(artefact.pmid)
        text = content.text if content is not None else artefact.content
        if not text:
            continue
        citation = artefact.citation or {}
        preferred_url = citation.get("preferred_url") if isinstance(citation, dict) else None
        if not preferred_url:
            preferred_url = f"https://pubmed.ncbi.nlm.nih.gov/{artefact.pmid}/"
        pmc_ref_count_val = citation.get("pmc_ref_count") if isinstance(citation, dict) else None
        try:
            pmc_ref_count = int(pmc_ref_count_val) if pmc_ref_count_val is not None else 0
        except (TypeError, ValueError):
            pmc_ref_count = 0

        extracted = extractor.extract_snippets(
            article_text=text,
            pmid=artefact.pmid,
            condition_terms=condition_terms,
            article_rank=artefact.rank,
            article_score=artefact.score,
            preferred_url=preferred_url,
            pmc_ref_count=pmc_ref_count,
        )
        snippet_candidates.extend(extracted)

    if snippet_candidates:
        selector = snippet_selector or select_top_snippets
        selected_snippets = selector(snippet_candidates)
        persist_snippet_candidates(
            session,
            article_artefacts=persisted_articles,
            snippet_candidates=selected_snippets,
        )

    return persisted_articles


def persist_pubmed_articles(
    session: Session,
    *,
    search_term_id: int,
    articles: Sequence[PubMedArticle],
    contents: Dict[str, ArticleContent],
) -> list[ArticleArtefact]:
    persisted: list[ArticleArtefact] = []
    for rank, article in enumerate(articles, start=1):
        citation = article.to_citation_dict()
        content = contents.get(article.pmid)
        stmt = select(ArticleArtefact).where(
            ArticleArtefact.search_term_id == search_term_id,
            ArticleArtefact.pmid == article.pmid,
        )
        existing = session.execute(stmt).scalar_one_or_none()
        if existing is None:
            artefact = ArticleArtefact(
                search_term_id=search_term_id,
                pmid=article.pmid,
                rank=rank,
                score=article.score,
                citation=citation,
                content=content.text if content else None,
                content_source=content.source if content else None,
                token_estimate=content.token_estimate if content else None,
                retrieved_at=content.fetched_at if content else None,
            )
            session.add(artefact)
            persisted.append(artefact)
            continue

        existing.rank = rank
        existing.score = article.score
        existing.citation = citation
        if content is not None:
            existing.content = content.text
            existing.content_source = content.source
            existing.token_estimate = content.token_estimate
            existing.retrieved_at = content.fetched_at
        persisted.append(existing)

    session.flush()
    return persisted


def _estimate_token_count(text: str) -> int:
    if not text:
        return 0
    return max(1, math.ceil(len(text.split()) * 1.05))


def _normalize_whitespace(value: str) -> str:
    return " ".join(value.split())
