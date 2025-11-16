from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional
from xml.etree import ElementTree

import httpx

from app.services.nih_http import dispatch_nih_request
from app.settings import DEFAULT_NIH_API_KEY, DEFAULT_NIH_CONTACT_EMAIL


DEFAULT_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
DEFAULT_TIMEOUT_SECONDS = 5.0


@dataclass(slots=True)
class ESpellSuggestion:
    query: str
    corrected_query: Optional[str] = None
    replaced: Optional[str] = None

    @property
    def suggestion(self) -> Optional[str]:
        return (self.corrected_query or self.replaced or None)


def extract_espell_correction(xml_payload: str) -> ESpellSuggestion:
    try:
        root = ElementTree.fromstring(xml_payload)
    except ElementTree.ParseError:
        return ESpellSuggestion(query="")

    query = _safe_text(root.find("Query"))
    corrected = _safe_text(root.find("CorrectedQuery"))
    replaced = _safe_text(root.find("SpelledQuery/Replaced"))

    return ESpellSuggestion(
        query=query,
        corrected_query=corrected,
        replaced=replaced,
    )


class NIHESpellClient:
    def __init__(
        self,
        *,
        http_client: Optional[httpx.Client] = None,
        database: str = "mesh",
        base_url: str = DEFAULT_BASE_URL,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        contact_email: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self._http_client = http_client
        self.database = database
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        resolved_email = contact_email or os.getenv("NIH_CONTACT_EMAIL") or DEFAULT_NIH_CONTACT_EMAIL
        self.contact_email = resolved_email.strip() if resolved_email and resolved_email.strip() else None
        resolved_key = api_key or os.getenv("NIH_API_KEY") or os.getenv("NCBI_API_KEY") or DEFAULT_NIH_API_KEY
        self.api_key = resolved_key.strip() if resolved_key and resolved_key.strip() else None

    def __call__(self, term: str) -> Optional[str]:
        params = {"db": self.database, "term": term}
        try:
            response = self._dispatch_request(params)
        except (httpx.HTTPError, RuntimeError):
            return None

        suggestion = extract_espell_correction(response.text)
        return suggestion.suggestion

    def _dispatch_request(self, params: dict[str, str]) -> httpx.Response:
        return dispatch_nih_request(
            http_client=self._http_client,
            method="get",
            base_url=self.base_url,
            endpoint="espell.fcgi",
            params=params,
            contact_email=self.contact_email,
            api_key=self.api_key,
            timeout_seconds=self.timeout_seconds,
        )


def _safe_text(node: ElementTree.Element | None) -> Optional[str]:
    if node is None or node.text is None:
        return None
    value = node.text.strip()
    return value or None
