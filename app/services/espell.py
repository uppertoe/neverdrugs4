from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from xml.etree import ElementTree

import httpx


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
    ) -> None:
        self._http_client = http_client
        self.database = database
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def __call__(self, term: str) -> Optional[str]:
        params = {"db": self.database, "term": term}
        try:
            response = self._dispatch_request(params)
        except httpx.HTTPError:
            return None

        if response.status_code >= 400:
            return None

        suggestion = extract_espell_correction(response.text)
        return suggestion.suggestion

    def _dispatch_request(self, params: dict[str, str]) -> httpx.Response:
        endpoint = f"{self.base_url}/espell.fcgi"
        if self._http_client is not None:
            return self._http_client.get(endpoint, params=params)

        with httpx.Client(timeout=self.timeout_seconds) as client:
            return client.get(endpoint, params=params)


def _safe_text(node: ElementTree.Element | None) -> Optional[str]:
    if node is None or node.text is None:
        return None
    value = node.text.strip()
    return value or None
