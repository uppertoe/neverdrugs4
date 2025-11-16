from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pytest

from app.services.nih_http import dispatch_nih_request


class _FakeResponse:
    def __init__(self) -> None:
        self.status_code = 200
        self.text = "OK"


class _RecordingHttpClient:
    def __init__(self) -> None:
        self.calls: List[Tuple[str, str, Dict[str, Any] | None]] = []
        self.response = _FakeResponse()

    def get(self, url: str, params: Dict[str, Any] | None = None) -> str:
        self.calls.append(("get", url, params))
        return self.response

    def post(self, url: str, data: Dict[str, Any] | None = None) -> str:
        self.calls.append(("post", url, data))
        return self.response


def test_dispatch_nih_request_injects_email_for_get() -> None:
    http_client = _RecordingHttpClient()
    original_params = {"db": "mesh", "term": "condition"}

    response = dispatch_nih_request(
        http_client=http_client,
        method="get",
        base_url="https://example",
        endpoint="esearch.fcgi",
        params=original_params,
        contact_email="sample@example.com",
    )

    assert response is http_client.response
    assert original_params == {"db": "mesh", "term": "condition"}
    assert http_client.calls == [
        (
            "get",
            "https://example/esearch.fcgi",
            {"db": "mesh", "term": "condition", "email": "sample@example.com"},
        )
    ]


def test_dispatch_nih_request_injects_email_and_api_key_for_post() -> None:
    http_client = _RecordingHttpClient()

    response = dispatch_nih_request(
        http_client=http_client,
        method="post",
        base_url="https://example",
        endpoint="esummary.fcgi",
        data={"id": "123"},
        contact_email="sample@example.com",
        api_key="abc123",
    )

    assert response is http_client.response
    assert http_client.calls == [
        (
            "post",
            "https://example/esummary.fcgi",
            {"id": "123", "email": "sample@example.com", "api_key": "abc123"},
        )
    ]


def test_dispatch_nih_request_handles_missing_email_and_api_key() -> None:
    http_client = _RecordingHttpClient()

    dispatch_nih_request(
        http_client=http_client,
        method="get",
        base_url="https://example",
        endpoint="info.fcgi",
        params={"db": "mesh"},
    )

    assert http_client.calls == [
        (
            "get",
            "https://example/info.fcgi",
            {"db": "mesh"},
        )
    ]


def test_dispatch_nih_request_rejects_unknown_method() -> None:
    http_client = _RecordingHttpClient()

    with pytest.raises(ValueError):
        dispatch_nih_request(
            http_client=http_client,
            method="delete",
            base_url="https://example",
            endpoint="boom",
        )
