from __future__ import annotations

from difflib import SequenceMatcher
from typing import Iterable

import httpx

DEFAULT_BASE_URL = "https://id.nlm.nih.gov/mesh"
DEFAULT_TIMEOUT_SECONDS = 5.0
DEFAULT_LIMIT = 10
FETCH_MULTIPLIER = 3


class NIHMeshSuggestionClient:
    """Fetch nearby MeSH descriptor labels for a supplied query term."""

    def __init__(
        self,
        *,
        http_client: httpx.Client | None = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        limit: int = DEFAULT_LIMIT,
    ) -> None:
        self._http_client = http_client
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.limit = max(1, limit)

    def suggest(self, label: str) -> list[str]:
        cleaned = (label or "").strip()
        if not cleaned:
            return []

        fetch_limit = max(self.limit * FETCH_MULTIPLIER, self.limit)
        candidate_scores: dict[str, tuple[float, int]] = {}
        insertion_order = 0

        for endpoint in ("descriptor", "supplementalRecord"):
            candidates = self._lookup(endpoint, cleaned, limit=fetch_limit)
            for candidate in candidates:
                if candidate in candidate_scores:
                    continue
                score = _similarity_score(cleaned, candidate)
                candidate_scores[candidate] = (score, insertion_order)
                insertion_order += 1

        ranked = sorted(
            candidate_scores.items(),
            key=lambda item: (-item[1][0], item[1][1]),
        )
        return [candidate for candidate, _ in ranked[: self.limit]]

    def _lookup(self, endpoint: str, label: str, *, limit: int) -> Iterable[str]:
        url = f"{self.base_url}/lookup/{endpoint}"
        params = {
            "label": label,
            "match": "contains",
            "limit": str(max(1, limit)),
        }

        try:
            response = self._request(url, params)
            payload = response.json()
        except (httpx.HTTPError, ValueError):
            return []

        results: list[str] = []
        for entry in payload:
            candidate = str(entry.get("label", "")).strip()
            if candidate:
                results.append(candidate)
        return results

    def _request(self, url: str, params: dict[str, str]) -> httpx.Response:
        if self._http_client is not None:
            response = self._http_client.get(url, params=params)
        else:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                response = client.get(url, params=params)
        response.raise_for_status()
        return response


def _similarity_score(query: str, candidate: str) -> float:
    query_norm = query.lower()
    candidate_norm = candidate.lower()
    ratio = SequenceMatcher(None, query_norm, candidate_norm).ratio()
    prefix_bonus = 0.1 if candidate_norm.startswith(query_norm) else 0.0
    return ratio + prefix_bonus
