from __future__ import annotations

from typing import Iterable

import httpx

DEFAULT_BASE_URL = "https://id.nlm.nih.gov/mesh"
DEFAULT_TIMEOUT_SECONDS = 5.0
DEFAULT_LIMIT = 5


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

        suggestions: list[str] = []
        seen: set[str] = set()

        for endpoint in ("descriptor", "supplementalRecord"):
            if len(suggestions) >= self.limit:
                break
            candidates = self._lookup(endpoint, cleaned)
            for candidate in candidates:
                if candidate not in seen:
                    suggestions.append(candidate)
                    seen.add(candidate)
                if len(suggestions) >= self.limit:
                    break

        return suggestions

    def _lookup(self, endpoint: str, label: str) -> Iterable[str]:
        url = f"{self.base_url}/lookup/{endpoint}"
        params = {
            "label": label,
            "match": "contains",
            "limit": str(self.limit),
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
