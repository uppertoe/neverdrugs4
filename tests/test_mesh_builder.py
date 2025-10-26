from __future__ import annotations

from pathlib import Path

import pytest

from app.services.mesh_builder import NIHMeshBuilder

FIXTURES = Path(__file__).parent / "fixtures"


class _FakeResponse:
    def __init__(self, text: str, status_code: int = 200) -> None:
        self.text = text
        self.status_code = status_code


class _SequentialHttpClient:
    def __init__(self, responses: list[_FakeResponse]) -> None:
        self._responses = responses
        self.calls: list[tuple[str, dict[str, str]]] = []

    def get(self, url: str, params: dict[str, str]) -> _FakeResponse:
        if not self._responses:
            raise RuntimeError("No more responses queued")
        self.calls.append((url, params))
        return self._responses.pop(0)


def test_mesh_builder_returns_ranked_terms() -> None:
    esearch_xml = (FIXTURES / "esearch_duchenne.xml").read_text(encoding="utf-8")
    esummary_xml = (FIXTURES / "esummary_68020388.xml").read_text(encoding="utf-8")
    http_client = _SequentialHttpClient([
        _FakeResponse(esearch_xml),
        _FakeResponse(esummary_xml),
    ])
    builder = NIHMeshBuilder(http_client=http_client, max_terms=4)

    result = builder("duchenne muscular dystrophy")

    assert result.source == "nih-esearch+esummary"
    assert result.mesh_terms == [
        "Muscular Dystrophy, Duchenne",
        "Duchenne Muscular Dystrophy",
        "Duchenne-Type Progressive Muscular Dystrophy",
        "Progressive Muscular Dystrophy, Duchenne Type",
    ]
    assert result.query_payload["esearch"]["primary_id"] == "68020388"
    assert result.query_payload["ranked_mesh_terms"][0]["term"] == "Muscular Dystrophy, Duchenne"
    assert http_client.calls[0][1]["term"] == "duchenne muscular dystrophy"
    assert http_client.calls[1][1]["id"] == "68020388"


def test_mesh_builder_handles_missing_ids() -> None:
    empty_esearch = """<?xml version='1.0' encoding='UTF-8'?><eSearchResult><Count>0</Count><IdList></IdList></eSearchResult>"""
    http_client = _SequentialHttpClient([
        _FakeResponse(empty_esearch),
    ])
    builder = NIHMeshBuilder(http_client=http_client)

    result = builder("condition")

    assert result.mesh_terms == []
    assert result.query_payload["esearch"]["ids"] == []


def test_mesh_builder_propagates_http_error() -> None:
    http_client = _SequentialHttpClient([
        _FakeResponse("fail", status_code=500),
    ])
    builder = NIHMeshBuilder(http_client=http_client)

    with pytest.raises(RuntimeError):
        builder("condition")
