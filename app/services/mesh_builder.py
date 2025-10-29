from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
from xml.etree import ElementTree

import httpx

from app.services.espell import DEFAULT_BASE_URL as NIH_BASE_URL
from app.services.search import MeshBuildResult, normalize_condition
from app.services.query_terms import build_nih_search_query


_TOKEN_TRANSLATION = str.maketrans({ch: " " for ch in "-,/"})

DEFAULT_DISALLOWED_EXTRA_TOKENS: tuple[str, ...] = (
    "becker",
    "mouse",
    "mice",
    "murine",
    "veterinary",
    "veterinarian",
    "virology",
)


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

    def __call__(self, normalized_term: str) -> MeshBuildResult:
        esearch = self._fetch_esearch(normalized_term)
        payload: dict[str, object] = {
            "normalized_query": normalized_term,
            "esearch": {
                "ids": esearch.ids,
                "translation": esearch.translation,
                "primary_id": esearch.primary_id,
            },
        }

        if not esearch.primary_id:
            return MeshBuildResult(mesh_terms=[], query_payload=payload, source="nih-esearch+esummary")

        esummary = self._fetch_esummary(esearch.primary_id)
        ranked_terms, query_tokens, primary_token = self._rank_terms(
            esummary.mesh_terms,
            normalized_term,
        )
        selected_terms = self._select_terms(
            ranked_terms,
            query_tokens=query_tokens,
            primary_token=primary_token,
        )

        payload["esummary"] = {
            "primary_id": esearch.primary_id,
            "mesh_terms": esummary.mesh_terms,
            "raw_xml": esummary.raw_xml,
        }
        payload["ranked_mesh_terms"] = [entry.to_dict() for entry in ranked_terms]

        try:
            payload["esearch"]["query"] = build_nih_search_query(selected_terms)
        except ValueError:
            selected_terms = []
            payload["esearch"]["query"] = None

        return MeshBuildResult(
            mesh_terms=selected_terms,
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

    def _fetch_esummary(self, mesh_id: str) -> ESummaryResult:
        params = {
            "db": self.database,
            "id": mesh_id,
        }
        response = self._dispatch_request("esummary.fcgi", params)
        root = self._parse_xml(response.text)

        mesh_terms = [node.text.strip() for node in root.findall('.//Item[@Name="DS_MeshTerms"]/Item') if node.text]

        return ESummaryResult(mesh_terms=mesh_terms, raw_xml=response.text)

    def _dispatch_request(self, endpoint: str, params: dict[str, str]) -> httpx.Response:
        url = f"{self.base_url}/{endpoint}"
        if self._http_client is not None:
            response = self._http_client.get(url, params=params)
        else:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                response = client.get(url, params=params)

        if response.status_code >= 400:
            raise RuntimeError(f"NIH request failed with status {response.status_code} for {endpoint}")
        return response

    def _parse_xml(self, xml_payload: str) -> ElementTree.Element:
        try:
            return ElementTree.fromstring(xml_payload)
        except ElementTree.ParseError as exc:
            raise RuntimeError("Failed to parse NIH XML response") from exc

    def _rank_terms(
        self,
        candidates: list[str],
        normalized_query: str,
    ) -> tuple[list["_RankedTerm"], set[str], Optional[str]]:
        ranked: list[_RankedTerm] = []
        query_norm = normalize_condition(normalized_query)
        query_token_list = _tokenize_for_mesh(query_norm)
        query_tokens = set(query_token_list)
        primary_token = query_token_list[0] if query_token_list else None
        for index, term in enumerate(candidates):
            term_norm = normalize_condition(term)
            term_token_list = _tokenize_for_mesh(term_norm)
            tokens = set(term_token_list)
            if not tokens:
                continue
            overlap = len(tokens & query_tokens)
            union_size = len(tokens | query_tokens) or 1
            jaccard = overlap / union_size
            ranked.append(
                _RankedTerm(
                    term=term,
                    normalized=term_norm,
                    tokens=tokens,
                    token_sequence=tuple(term_token_list),
                    overlap=overlap,
                    jaccard=jaccard,
                    index=index,
                    extra_tokens=sorted(tokens - query_tokens),
                )
            )
        ranked.sort(key=lambda entry: (-entry.jaccard, entry.index))
        return ranked, query_tokens, primary_token

    def _select_terms(
        self,
        ranked_terms: list["_RankedTerm"],
        *,
        query_tokens: set[str],
        primary_token: Optional[str],
    ) -> list[str]:
        selected: list[str] = []
        seen_sequences: set[Tuple[str, ...]] = set()
        for entry in ranked_terms:
            if entry.token_sequence in seen_sequences:
                continue
            if self._is_disallowed(entry, primary_token=primary_token):
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
    ) -> bool:
        extra_tokens = set(entry.extra_tokens)
        if extra_tokens & self.disallowed_extra_tokens:
            return True
        if entry.jaccard == 0:
            return True
        if self.require_primary_token and primary_token and primary_token not in entry.tokens:
            return True
        if "type" in extra_tokens and "progressive" not in entry.tokens:
            return True
        return False


@dataclass(slots=True)
class _RankedTerm:
    term: str
    normalized: str
    tokens: set[str]
    token_sequence: tuple[str, ...]
    overlap: int
    jaccard: float
    index: int
    extra_tokens: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "term": self.term,
            "normalized": self.normalized,
            "overlap": self.overlap,
            "jaccard": round(self.jaccard, 4),
            "rank": self.index,
            "extra_tokens": self.extra_tokens,
            "token_sequence": list(self.token_sequence),
        }


def _tokenize_for_mesh(value: str) -> list[str]:
    return value.translate(_TOKEN_TRANSLATION).split()
