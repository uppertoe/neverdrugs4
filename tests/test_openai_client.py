from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.services.llm_batches import LLMRequestBatch, SnippetLLMEntry
from app.services.openai_client import LLMCompletionResult, OpenAIChatClient


class _StubCompletions:
    def __init__(self, response: object) -> None:
        self._response = response
        self.last_kwargs: dict[str, object] | None = None

    def create(self, **kwargs):  # type: ignore[no-untyped-def]
        self.last_kwargs = kwargs
        return self._response


class _StubOpenAIClient:
    def __init__(self, response: object) -> None:
        self.chat = SimpleNamespace(completions=_StubCompletions(response))


def _make_batch() -> LLMRequestBatch:
    snippet = SnippetLLMEntry(
        pmid="12345678",
        snippet_id=42,
        drug="test-drug",
        classification="risk",
        snippet_text="Example snippet text",
        snippet_score=3.2,
        cues=["example"],
        article_rank=1,
        article_score=5.5,
        citation_url="https://pubmed.ncbi.nlm.nih.gov/12345678/",
        article_title="Example Article",
        content_source="pubmed",
        token_estimate=120,
    )
    messages = [
        {"role": "system", "content": "System"},
        {"role": "user", "content": "User"},
    ]
    return LLMRequestBatch(messages=messages, snippets=[snippet], token_estimate=240)


def _make_response(content: object) -> object:
    usage = SimpleNamespace(prompt_tokens=100, completion_tokens=55, total_tokens=155)
    choice = SimpleNamespace(message=SimpleNamespace(content=content))
    return SimpleNamespace(id="resp-123", model="gpt-4o-mini", choices=[choice], usage=usage)


def test_run_batches_returns_result_with_stub_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    payload = {"condition": "Sample", "drugs": []}
    response = _make_response(json.dumps(payload))
    stub = _StubOpenAIClient(response)

    client = OpenAIChatClient(client=stub, max_retries=1)
    batch = _make_batch()

    results = client.run_batches([batch])

    assert len(results) == 1
    result = results[0]
    assert isinstance(result, LLMCompletionResult)
    assert result.model == "gpt-4o-mini"
    assert result.response_id == "resp-123"
    assert result.parsed_json() == payload

    assert stub.chat.completions.last_kwargs is not None
    kwargs = stub.chat.completions.last_kwargs
    assert kwargs["messages"] == batch.messages
    assert kwargs["response_format"] == {"type": "json_object"}


def test_client_requires_api_key_when_no_custom_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises((RuntimeError, ImportError)):
        OpenAIChatClient(max_retries=1)


def test_flatten_response_content_list(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    content = [
        {"text": "{\"condition\":"},
        {"text": " \"Sample\""},
        {"text": ", \"drugs\": []}"},
    ]
    response = _make_response(content)
    stub = _StubOpenAIClient(response)

    client = OpenAIChatClient(client=stub, max_retries=1)
    batch = _make_batch()

    result = client.run_batches([batch])[0]
    assert result.parsed_json() == {"condition": "Sample", "drugs": []}


def test_openai_fixture_has_expected_schema() -> None:
    fixture_path = Path("tests/fixtures/openai_duchenne_response.json")
    assert fixture_path.exists()

    records = json.loads(fixture_path.read_text(encoding="utf-8"))
    assert isinstance(records, list) and records

    record = records[0]
    assert {"batch_index", "model", "response_id", "usage", "content", "parsed"} <= record.keys()

    parsed = record["parsed"]
    assert isinstance(parsed, dict)
    assert parsed == json.loads(record["content"])

    assert parsed.get("condition") == "Duchenne muscular dystrophy"
    assert parsed.get("claims")

    claim_entry = parsed["claims"][0]
    assert claim_entry["claim_id"].startswith("risk:")
    assert claim_entry["classification"] in {"risk", "safety"}
    assert "succinylcholine" in claim_entry["drugs"]
    assert "depolarising neuromuscular blocker" in claim_entry["drug_classes"]
    assert claim_entry["supporting_evidence"]

    evidence = claim_entry["supporting_evidence"][0]
    assert isinstance(evidence["snippet_id"], str)
    assert evidence["pmid"] == "11111111"
    assert "malignant hyperthermia" in " ".join(evidence["key_points"]).lower()
