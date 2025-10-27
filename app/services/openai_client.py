from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Sequence

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

DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_OPENAI_TEMPERATURE = 0.0
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
        temperature: float = DEFAULT_OPENAI_TEMPERATURE,
        max_retries: int = DEFAULT_OPENAI_MAX_RETRIES,
        response_format: dict[str, str] | None = None,
        backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
        request_timeout: float | None = None,
        client: Any | None = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_retries = max(1, max_retries)
        self.backoff_seconds = max(0.0, backoff_seconds)
        self.request_timeout = request_timeout
        self.response_format = response_format or {"type": "json_object"}

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
                response = self._client.chat.completions.create(
                    model=model,
                    messages=batch.messages,
                    temperature=self.temperature,
                    response_format=self.response_format,
                )
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
    choices = getattr(response, "choices", None) or []
    if not choices:
        return ""

    message = choices[0].message
    content = getattr(message, "content", "")
    return _flatten_content(content)


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
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        total_tokens = usage.get("total_tokens")
    else:
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
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
