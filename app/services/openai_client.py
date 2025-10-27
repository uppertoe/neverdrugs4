from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from dotenv import load_dotenv

from app.services.llm_batches import LLMRequestBatch

load_dotenv()

try:  # pragma: no cover - import guard for optional error aliases
    import openai as _openai
except ImportError:  # pragma: no cover
    _openai = None

if _openai is not None:  # pragma: no cover - attribute detection at import time
    OpenAI = getattr(_openai, "OpenAI", None)
    APIError = getattr(_openai, "APIError", Exception)
    APIConnectionError = getattr(_openai, "APIConnectionError", APIError)
    APITimeoutError = getattr(_openai, "APITimeoutError", APIError)
    RateLimitError = getattr(_openai, "RateLimitError", APIError)
    ServiceUnavailableError = getattr(_openai, "ServiceUnavailableError", APIError)
else:
    OpenAI = None
    APIError = Exception
    APIConnectionError = APITimeoutError = RateLimitError = ServiceUnavailableError = Exception

DEFAULT_OPENAI_MODEL = "gpt-5-mini"
DEFAULT_OPENAI_TEMPERATURE: float | None = None
DEFAULT_OPENAI_MAX_RETRIES = 3
DEFAULT_BACKOFF_SECONDS = 2.0


@dataclass(slots=True)
class LLMCompletionResult:
    batch: LLMRequestBatch
    model: str
    response_id: str
    content: str
    usage: dict[str, int] | None

    def parsed_json(self) -> Any:
        text = self.content.strip()
        if not text:
            raise ValueError("LLM completion did not return any content")
        return json.loads(text)


class OpenAIChatClient:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = DEFAULT_OPENAI_MODEL,
    temperature: float | None = DEFAULT_OPENAI_TEMPERATURE,
        max_retries: int = DEFAULT_OPENAI_MAX_RETRIES,
    response_schema: dict[str, Any] | None = None,
        backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
        request_timeout: float | None = None,
        client: Any | None = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_retries = max(1, max_retries)
        self.backoff_seconds = max(0.0, backoff_seconds)
        self.request_timeout = request_timeout
        self.response_schema = response_schema or {
            "type": "json_schema",
            "name": "llm_claims",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "condition": {"type": "string"},
                    "claims": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "claim_id": {"type": "string"},
                                "classification": {
                                    "type": "string",
                                    "enum": ["risk", "safety", "uncertain"],
                                },
                                "drug_classes": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "drugs": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "summary": {"type": "string"},
                                "confidence": {
                                    "type": "string",
                                    "enum": ["low", "medium", "high"],
                                },
                                "supporting_evidence": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "snippet_id": {"type": "string"},
                                            "pmid": {"type": "string"},
                                            "article_title": {"type": "string"},
                                            "key_points": {
                                                "type": "array",
                                                "items": {"type": "string"},
                                            },
                                            "notes": {"type": "string"},
                                        },
                                        "required": [
                                            "snippet_id",
                                            "pmid",
                                            "article_title",
                                            "key_points",
                                            "notes",
                                        ],
                                        "additionalProperties": False,
                                    },
                                },
                            },
                            "required": [
                                "claim_id",
                                "classification",
                                "drug_classes",
                                "drugs",
                                "summary",
                                "confidence",
                                "supporting_evidence",
                            ],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["condition", "claims"],
                "additionalProperties": False,
            },
        }

        if client is not None:
            self._client = client
            return

        if OpenAI is None:  # pragma: no cover - raises when dependency missing
            version = "not installed" if _openai is None else getattr(_openai, "__version__", "unknown")
            raise ImportError(
                "The installed 'openai' package does not provide the OpenAI client. "
                "Install or upgrade to 'openai>=1.0'. "
                f"Detected version: {version}."
            )

        resolved_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_key:
            raise RuntimeError(
                "OpenAI API key not provided. Set OPENAI_API_KEY or pass api_key to OpenAIChatClient."
            )

        self._client = OpenAI(api_key=resolved_key, timeout=request_timeout)

    def run_batches(
        self,
        batches: Sequence[LLMRequestBatch],
        *,
        model: str | None = None,
    ) -> list[LLMCompletionResult]:
        results: list[LLMCompletionResult] = []
        target_model = model or self.model
        for batch in batches:
            results.append(self._invoke_batch(batch, target_model))
        return results

    def _invoke_batch(self, batch: LLMRequestBatch, model: str) -> LLMCompletionResult:
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                request_payload: dict[str, Any] = {
                    "model": model,
                    "input": _convert_messages_to_responses_input(batch.messages),
                    "text": {"format": self.response_schema},
                }
                if self.temperature is not None:
                    request_payload["temperature"] = self.temperature
                response = self._client.responses.create(**request_payload)
                content = _extract_content(response)
                usage = _extract_usage(response)
                response_id = getattr(response, "id", "")
                return LLMCompletionResult(
                    batch=batch,
                    model=model,
                    response_id=response_id,
                    content=content,
                    usage=usage,
                )
            except (  # pragma: no cover - exercised via integration
                APIError,
                APIConnectionError,
                APITimeoutError,
                RateLimitError,
                ServiceUnavailableError,
            ) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    raise RuntimeError(
                        "OpenAI chat completion failed after exhausting retries"
                    ) from exc
                sleep_seconds = self.backoff_seconds * attempt
                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)

        assert last_error is not None  # pragma: no cover
        raise RuntimeError("LLM invocation failed") from last_error


def _extract_content(response: Any) -> str:
    if hasattr(response, "output_text"):
        text_value = getattr(response, "output_text")
        if isinstance(text_value, str) and text_value.strip():
            return text_value

    outputs = getattr(response, "output", None) or getattr(response, "outputs", None)
    if outputs:
        texts: list[str] = []
        for item in _ensure_iterable(outputs):
            content = getattr(item, "content", None) or getattr(item, "contents", None)
            if not content:
                continue
            for part in _ensure_iterable(content):
                text = getattr(part, "text", None)
                if text is None and isinstance(part, dict):
                    text = part.get("text")
                if isinstance(text, str):
                    texts.append(text)
        if texts:
            return "".join(texts)

    choices = getattr(response, "choices", None) or []
    if choices:
        message = choices[0].message
        content = getattr(message, "content", "")
        return _flatten_content(content)

    return ""


def _flatten_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if "text" in item and isinstance(item["text"], str):
                    parts.append(item["text"])
                elif "content" in item and isinstance(item["content"], str):
                    parts.append(item["content"])
            elif isinstance(item, str):
                parts.append(item)
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content)


def _extract_usage(response: Any) -> dict[str, int] | None:
    usage = getattr(response, "usage", None)
    if not usage:
        return None

    if isinstance(usage, dict):
        prompt_tokens = usage.get("prompt_tokens") or usage.get("input_tokens")
        completion_tokens = usage.get("completion_tokens") or usage.get("output_tokens")
        total_tokens = usage.get("total_tokens")
    else:
        prompt_tokens = getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None) or getattr(usage, "output_tokens", None)
        total_tokens = getattr(usage, "total_tokens", None)

    data = {
        key: value
        for key, value in (
            ("prompt_tokens", prompt_tokens),
            ("completion_tokens", completion_tokens),
            ("total_tokens", total_tokens),
        )
        if isinstance(value, int)
    }
    return data or None


def _convert_messages_to_responses_input(messages: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        if isinstance(content, str):
            content_payload = [{"type": "input_text", "text": content}]
        elif isinstance(content, list):
            content_payload = []
            for item in content:
                if isinstance(item, dict) and "type" in item:
                    payload_item = dict(item)
                    if payload_item.get("type") == "text":
                        payload_item["type"] = "input_text"
                    content_payload.append(payload_item)
                elif isinstance(item, str):
                    content_payload.append({"type": "input_text", "text": item})
            if not content_payload:
                content_payload = [{"type": "input_text", "text": ""}]
        else:
            content_payload = [{"type": "input_text", "text": str(content)}]
        converted.append({"role": role, "content": content_payload})
    return converted


def _ensure_iterable(value: Any) -> Iterable[Any]:
    if isinstance(value, (list, tuple)):
        return value
    return (value,)
