from __future__ import annotations

import os
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Optional, Tuple
from xml.etree import ElementTree
from xml.etree.ElementTree import tostring

import httpx

from app.services.espell import DEFAULT_BASE_URL as NIH_BASE_URL
from app.services.nih_http import dispatch_nih_request
from app.services.query_terms import build_nih_search_query
from app.services.search import MeshBuildResult, normalize_condition
from app.settings import DEFAULT_NIH_API_KEY, DEFAULT_NIH_CONTACT_EMAIL


_TOKEN_TRANSLATION = str.maketrans({ch: " " for ch in "-,/"})

DEFAULT_DISALLOWED_EXTRA_TOKENS: tuple[str, ...] = (
    "mouse",
    "mice",
    "murine",
    "veterinary",
    "veterinarian",
    "virology",
)

_GENERIC_ALIAS_TOKENS: set[str] = {
    "disease",
    "diseases",
    "disorder",
    "disorders",
    "syndrome",
    "syndromes",
    "type",
    "types",
    "of",
}


@dataclass(slots=True)
class ESearchResult:
    ids: list[str]
    translation: Optional[str]
    raw_xml: str

    @property
    def primary_id(self) -> Optional[str]:
        return self.ids[0] if self.ids else None


@dataclass(slots=True)
class ESummaryResult:
    mesh_terms: list[str]
    raw_xml: str


class NIHMeshBuilder:
    def __init__(
        self,
        *,
        http_client: Optional[httpx.Client] = None,
        base_url: str = NIH_BASE_URL,
        database: str = "mesh",
        retmax: int = 5,
        timeout_seconds: float = 5.0,
        max_terms: int = 5,
        disallowed_extra_tokens: Optional[set[str]] = None,
        require_primary_token: bool = True,
        contact_email: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self._http_client = http_client
        self.base_url = base_url.rstrip("/")
        self.database = database
        self.retmax = retmax
        self.timeout_seconds = timeout_seconds
        self.max_terms = max_terms
        self.disallowed_extra_tokens = (
            set(disallowed_extra_tokens)
            if disallowed_extra_tokens is not None
            else set(DEFAULT_DISALLOWED_EXTRA_TOKENS)
        )
        self.require_primary_token = require_primary_token
        resolved_email = contact_email or os.getenv("NIH_CONTACT_EMAIL") or DEFAULT_NIH_CONTACT_EMAIL
        self.contact_email = resolved_email.strip() if resolved_email and resolved_email.strip() else None
        resolved_key = api_key or os.getenv("NIH_API_KEY") or os.getenv("NCBI_API_KEY") or DEFAULT_NIH_API_KEY
        self.api_key = resolved_key.strip() if resolved_key and resolved_key.strip() else None

    def __call__(self, normalized_term: str) -> MeshBuildResult:
        esearch = self._fetch_esearch(normalized_term)
        payload: dict[str, object] = {
            "normalized_query": normalized_term,
            "esearch": {
                "ids": esearch.ids,
                "translation": esearch.translation,
                "primary_id": None,
            },
        }

        primary_candidate, candidates, confident_candidates = self._select_esummary(
            esearch.ids,
            normalized_term,
        )
        payload["esearch"]["primary_id"] = primary_candidate.mesh_id if primary_candidate else None
        payload["esearch"]["candidates"] = [candidate.to_dict() for candidate in candidates]

        if not primary_candidate:
            return MeshBuildResult(mesh_terms=[], query_payload=payload, source="nih-esearch+esummary")

        esummary = primary_candidate.summary
        canonical_term = primary_candidate.canonical_term
        alias_tokens = self._collect_alias_tokens(esummary.mesh_terms, normalized_term)

        ranked_terms, query_tokens, primary_token = self._rank_terms(
            esummary.mesh_terms,
            normalized_term,
            alias_tokens=alias_tokens,
        )
        ranked_selection = self._select_terms(
            ranked_terms,
            query_tokens=query_tokens,
            primary_token=primary_token,
            alias_tokens=alias_tokens,
        )

        ranked_selection = self._harmonise_selected_terms(
            ranked_selection,
            canonical_term=canonical_term,
        )

        resolved_mesh_terms = self._resolve_mesh_terms(primary_candidate, confident_candidates)
        if len(resolved_mesh_terms) > 1:
            selected_terms = resolved_mesh_terms
        else:
            selected_terms = ranked_selection

        payload["esummary"] = {
            "primary_id": primary_candidate.mesh_id,
            "mesh_terms": esummary.mesh_terms,
            "raw_xml": esummary.raw_xml,
        }
        payload["ranked_mesh_terms"] = [entry.to_dict() for entry in ranked_terms]
        if canonical_term:
            payload["canonical_mesh_term"] = canonical_term
        payload["selected_mesh_terms"] = list(selected_terms)

        payload["esearch"]["resolved_mesh_terms"] = list(resolved_mesh_terms)

        try:
            payload["esearch"]["query"] = build_nih_search_query(selected_terms)
        except ValueError:
            selected_terms = []
            payload["selected_mesh_terms"] = []
            payload["esearch"]["query"] = None

        mesh_terms = (
            resolved_mesh_terms
            if resolved_mesh_terms
            else ([canonical_term] if canonical_term else list(selected_terms))
        )

        return MeshBuildResult(
            mesh_terms=mesh_terms,
            query_payload=payload,
            source="nih-esearch+esummary",
        )

    def _fetch_esearch(self, normalized_term: str) -> ESearchResult:
        params = {
            "db": self.database,
            "term": normalized_term,
            "retmax": str(self.retmax),
        }
        response = self._dispatch_request("esearch.fcgi", params)
        root = self._parse_xml(response.text)

        ids = [node.text.strip() for node in root.findall(".//IdList/Id") if node.text]
        translation_node = root.find(".//TranslationSet/Translation/To")
        translation = translation_node.text.strip() if translation_node is not None and translation_node.text else None

        return ESearchResult(ids=ids, translation=translation, raw_xml=response.text)

    def _fetch_esummary(self, mesh_ids: list[str]) -> dict[str, ESummaryResult]:
        if not mesh_ids:
            return {}

        params = {
            "db": self.database,
            "id": ",".join(mesh_ids),
            "version": "2.0",
        }
        response = self._dispatch_request("esummary.fcgi", params)
        root = self._parse_xml(response.text)

        summaries: dict[str, ESummaryResult] = {}
        for docsum in self._iter_docsum_nodes(root):
            mesh_id = self._extract_docsum_id(docsum)
            if not mesh_id:
                continue
            mesh_terms = self._extract_mesh_terms(docsum)
            summaries[mesh_id] = ESummaryResult(
                mesh_terms=mesh_terms,
                raw_xml=tostring(docsum, encoding="unicode"),
            )

        if not summaries:
            return {}

        # Ensure every requested id is represented, even if missing from payload
        for mesh_id in mesh_ids:
            summaries.setdefault(mesh_id, ESummaryResult(mesh_terms=[], raw_xml=response.text))

        return summaries

    def _select_esummary(
        self,
        mesh_ids: list[str],
        normalized_query: str,
    ) -> tuple["_DescriptorCandidate" | None, list["_DescriptorCandidate"], list["_DescriptorCandidate"]]:
        if not mesh_ids:
            return None, [], []

        query_tokens = set(_tokenize_for_mesh(normalize_condition(normalized_query)))
        candidates: list[_DescriptorCandidate] = []

        esummary_map = self._fetch_esummary(mesh_ids)

        for index, mesh_id in enumerate(mesh_ids):
            summary = esummary_map.get(mesh_id)
            canonical_term = self._canonical_mesh_term(summary.mesh_terms)
            if not canonical_term:
                continue

            canonical_normalized = normalize_condition(canonical_term)
            canonical_tokens = set(_tokenize_for_mesh(canonical_normalized))

            alias_tokens = self._collect_alias_tokens(summary.mesh_terms, normalized_query)
            ranked_terms, _, _ = self._rank_terms(
                summary.mesh_terms,
                normalized_query,
                alias_tokens=alias_tokens,
            )
            best_entry = ranked_terms[0] if ranked_terms else None
            ranked_tokens = best_entry.tokens if best_entry else set()

            canonical_overlap = len(query_tokens & canonical_tokens)
            ranked_overlap = len(query_tokens & ranked_tokens)
            has_primary_token = bool(query_tokens and ranked_tokens and query_tokens & ranked_tokens)
            score = best_entry.score if best_entry else 0.0

            candidates.append(
                _DescriptorCandidate(
                    mesh_id=mesh_id,
                    summary=summary,
                    canonical_term=canonical_term,
                    canonical_tokens=canonical_tokens,
                    canonical_overlap=canonical_overlap,
                    ranked_overlap=ranked_overlap,
                    has_query_token=has_primary_token,
                    score=score,
                    index=index,
                )
            )

        if not candidates:
            return None, [], []

        confident = [
            candidate
            for candidate in candidates
            if candidate.canonical_overlap > 0 or candidate.ranked_overlap > 0
        ]

        confident_sorted = sorted(
            confident,
            key=lambda c: (
                -c.canonical_overlap,
                -c.ranked_overlap,
                -c.score,
                c.index,
            ),
        )

        if confident_sorted:
            primary = confident_sorted[0]
        else:
            primary = min(candidates, key=lambda c: c.index)

        return primary, candidates, confident_sorted

    def _iter_docsum_nodes(self, root: ElementTree.Element) -> list[ElementTree.Element]:
        docsum_nodes: list[ElementTree.Element] = []
        docsum_nodes.extend(root.findall("DocSum"))
        docsum_nodes.extend(root.findall(".//DocSum"))
        document_summaries = root.findall(".//DocumentSummarySet/DocumentSummary")
        if document_summaries:
            docsum_nodes.extend(document_summaries)
        seen = set()
        ordered: list[ElementTree.Element] = []
        for node in docsum_nodes:
            identity = id(node)
            if identity in seen:
                continue
            seen.add(identity)
            ordered.append(node)
        return ordered

    @staticmethod
    def _extract_docsum_id(node: ElementTree.Element) -> str | None:
        if node.tag == "DocSum":
            identifier = node.findtext("Id")
            return identifier.strip() if identifier else None
        if node.tag == "DocumentSummary":
            identifier = node.get("uid")
            return identifier.strip() if identifier else None
        return None

    def _extract_mesh_terms(self, node: ElementTree.Element) -> list[str]:
        def _collect(xpath: str) -> list[str]:
            collected: list[str] = []
            for child in node.findall(xpath):
                text = (child.text or "").strip()
                if text:
                    collected.append(text)
            return collected

        for candidate_xpath in (
            './/Item[@Name="DS_MeshTerms"]/Item',
            './/DS_MeshTerms//DS_MeshTerm',
            './/DS_MeshTerms/*',
        ):
            terms = _collect(candidate_xpath)
            if terms:
                return terms

        return []

    def _dispatch_request(self, endpoint: str, params: dict[str, str]) -> httpx.Response:
        return dispatch_nih_request(
            http_client=self._http_client,
            method="get",
            base_url=self.base_url,
            endpoint=endpoint,
            params=params,
            contact_email=self.contact_email,
            api_key=self.api_key,
            timeout_seconds=self.timeout_seconds,
        )

    def _parse_xml(self, xml_payload: str) -> ElementTree.Element:
        try:
            return ElementTree.fromstring(xml_payload)
        except ElementTree.ParseError as exc:
            raise RuntimeError("Failed to parse NIH XML response") from exc

    def _rank_terms(
        self,
        candidates: list[str],
        normalized_query: str,
        *,
        alias_tokens: Optional[set[str]] = None,
    ) -> tuple[list["_RankedTerm"], set[str], Optional[str]]:
        ranked: list[_RankedTerm] = []
        query_norm = normalize_condition(normalized_query)
        query_token_list = _tokenize_for_mesh(query_norm)
        query_tokens = set(query_token_list)
        alias_tokens = set(alias_tokens or set())
        primary_token = query_token_list[0] if query_token_list else None
        ordered_query = " ".join(query_token_list)
        sorted_query = " ".join(sorted(query_token_list))

        for index, term in enumerate(candidates):
            term_norm = normalize_condition(term)
            term_token_list = _tokenize_for_mesh(term_norm)
            tokens = set(term_token_list)
            if not tokens:
                continue
            base_overlap = len(tokens & query_tokens)
            base_union = len(tokens | query_tokens) or 1
            base_jaccard = base_overlap / base_union
            alias_overlap = len(tokens & alias_tokens)
            has_primary_token = bool(primary_token and primary_token in tokens)
            if base_jaccard == 0 and alias_overlap:
                combined_tokens = tokens | query_tokens | alias_tokens
                union_size = len(combined_tokens) or 1
                jaccard = alias_overlap / union_size
                overlap = alias_overlap
            else:
                jaccard = base_jaccard
                overlap = base_overlap
            if has_primary_token and jaccard < 1.0:
                jaccard = min(1.0, jaccard + 0.2)
            extra_tokens_set = tokens - query_tokens
            ordered_term = " ".join(term_token_list)
            sorted_term = " ".join(sorted(term_token_list))
            similarity = max(
                SequenceMatcher(None, ordered_query, ordered_term).ratio(),
                SequenceMatcher(None, sorted_query, sorted_term).ratio(),
            )
            score = (jaccard * 0.7) + (similarity * 0.3)
            generic_extra = extra_tokens_set & _GENERIC_ALIAS_TOKENS
            non_generic_extra = extra_tokens_set - _GENERIC_ALIAS_TOKENS
            if generic_extra and not non_generic_extra:
                score += 0.02
            if "becker" in tokens and "becker" not in query_tokens:
                score -= 0.2
            ranked.append(
                _RankedTerm(
                    term=term,
                    normalized=term_norm,
                    tokens=tokens,
                    token_sequence=tuple(term_token_list),
                    overlap=overlap,
                    jaccard=jaccard,
                    similarity=similarity,
                    score=score,
                    alias_overlap=alias_overlap,
                    has_primary_token=has_primary_token,
                    index=index,
                    extra_tokens=sorted(extra_tokens_set),
                )
            )
        ranked.sort(key=lambda entry: (-entry.score, entry.index))
        return ranked, query_tokens, primary_token

    def _select_terms(
        self,
        ranked_terms: list["_RankedTerm"],
        *,
        query_tokens: set[str],
        primary_token: Optional[str],
        alias_tokens: set[str],
    ) -> list[str]:
        selected: list[str] = []
        seen_sequences: set[Tuple[str, ...]] = set()
        alias_tokens = set(alias_tokens)
        for entry in ranked_terms:
            if entry.token_sequence in seen_sequences:
                continue
            if self._is_disallowed(entry, primary_token=primary_token, alias_tokens=alias_tokens):
                continue
            seen_sequences.add(entry.token_sequence)
            selected.append(entry.term)
            if len(selected) >= self.max_terms:
                break
        return selected

    def _is_disallowed(
        self,
        entry: "_RankedTerm",
        *,
        primary_token: Optional[str],
        alias_tokens: set[str],
    ) -> bool:
        alias_hit = bool(entry.alias_overlap)
        extra_tokens = set(entry.extra_tokens)
        if extra_tokens & self.disallowed_extra_tokens:
            return True
        if entry.jaccard == 0 and not alias_hit:
            return True
        if self.require_primary_token and primary_token and not entry.has_primary_token and not alias_hit:
            return True
        if "type" in extra_tokens and "progressive" not in entry.tokens:
            return True
        return False

    def _collect_alias_tokens(self, mesh_terms: list[str], normalized_query: str) -> set[str]:
        if not mesh_terms:
            return set()

        normalized_terms = [normalize_condition(term) for term in mesh_terms]
        query_tokens = set(_tokenize_for_mesh(normalize_condition(normalized_query)))
        tokens_by_index = [set(_tokenize_for_mesh(term)) for term in normalized_terms]

        focus_indices: list[int] = []
        if query_tokens:
            focus_indices = [idx for idx, tokens in enumerate(tokens_by_index) if tokens & query_tokens]

        if not focus_indices and query_tokens:
            best_index: Optional[int] = None
            best_score = 0.0
            for idx, tokens in enumerate(tokens_by_index):
                if not tokens:
                    continue
                overlap = len(tokens & query_tokens)
                if overlap == 0:
                    continue
                union = len(tokens | query_tokens) or 1
                score = overlap / union
                if score > best_score:
                    best_score = score
                    best_index = idx
            if best_index is not None:
                focus_indices.append(best_index)

        if not focus_indices:
            return set()

        window_start = max(0, min(focus_indices) - 6)
        window_end = min(len(mesh_terms), max(focus_indices) + 7)
        tokens: set[str] = set()
        for term in mesh_terms[window_start:window_end]:
            normalized = normalize_condition(term)
            tokens.update(_tokenize_for_mesh(normalized))

        meaningful_tokens = {token for token in tokens if token and token not in _GENERIC_ALIAS_TOKENS}
        return meaningful_tokens

    @staticmethod
    def _canonical_mesh_term(mesh_terms: list[str]) -> str | None:
        for term in mesh_terms or []:
            cleaned = (term or "").strip()
            if cleaned:
                return cleaned
        return None

    def _resolve_mesh_terms(
        self,
        primary: Optional[_DescriptorCandidate],
        confident_candidates: list[_DescriptorCandidate],
    ) -> list[str]:
        if primary is None:
            return []

        if not confident_candidates:
            return [primary.canonical_term]

        top = confident_candidates[0]
        if len(confident_candidates) == 1:
            return [top.canonical_term]

        second = confident_candidates[1]

        if top.canonical_overlap > second.canonical_overlap:
            return [top.canonical_term]
        if top.ranked_overlap > second.ranked_overlap:
            return [top.canonical_term]
        if top.score - second.score >= 0.1:
            return [top.canonical_term]

        terms: list[str] = []
        for candidate in confident_candidates:
            if candidate.canonical_term not in terms:
                terms.append(candidate.canonical_term)
            if len(terms) >= self.max_terms:
                break

        if not terms:
            return [primary.canonical_term]

        return terms

    def _harmonise_selected_terms(
        self,
        selected_terms: list[str],
        *,
        canonical_term: str | None,
    ) -> list[str]:
        ordered: list[str] = []
        seen: set[str] = set()

        def _add(term: str | None) -> None:
            if not term:
                return
            cleaned = term.strip()
            if not cleaned:
                return
            key = cleaned.lower()
            if key in seen:
                return
            seen.add(key)
            ordered.append(cleaned)

        _add(canonical_term)
        for term in selected_terms:
            _add(term)

        if self.max_terms and len(ordered) > self.max_terms:
            ordered = ordered[: self.max_terms]

        return ordered


@dataclass(slots=True)
class _DescriptorCandidate:
    mesh_id: str
    summary: ESummaryResult
    canonical_term: str
    canonical_tokens: set[str]
    canonical_overlap: int
    ranked_overlap: int
    has_query_token: bool
    score: float
    index: int

    def to_dict(self) -> dict[str, object]:
        return {
            "mesh_id": self.mesh_id,
            "canonical_term": self.canonical_term,
            "canonical_overlap": self.canonical_overlap,
            "ranked_overlap": self.ranked_overlap,
            "has_query_token": self.has_query_token,
            "score": round(self.score, 4),
            "index": self.index,
        }


@dataclass(slots=True)
class _RankedTerm:
    term: str
    normalized: str
    tokens: set[str]
    token_sequence: tuple[str, ...]
    overlap: int
    jaccard: float
    similarity: float
    score: float
    alias_overlap: int
    has_primary_token: bool
    index: int
    extra_tokens: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "term": self.term,
            "normalized": self.normalized,
            "overlap": self.overlap,
            "jaccard": round(self.jaccard, 4),
            "similarity": round(self.similarity, 4),
            "score": round(self.score, 4),
            "alias_overlap": self.alias_overlap,
            "has_primary_token": self.has_primary_token,
            "rank": self.index,
            "extra_tokens": self.extra_tokens,
            "token_sequence": list(self.token_sequence),
        }


def _tokenize_for_mesh(value: str) -> list[str]:
    return value.translate(_TOKEN_TRANSLATION).split()
