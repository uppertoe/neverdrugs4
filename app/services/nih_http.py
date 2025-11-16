from __future__ import annotations

from typing import Any, Dict, Mapping, MutableMapping, Optional

import httpx


def _prepare_payload(
    payload: Optional[Mapping[str, Any]] | Optional[MutableMapping[str, Any]],
    *,
    contact_email: Optional[str],
    api_key: Optional[str],
) -> Dict[str, Any]:
    if payload is None:
        prepared: Dict[str, Any] = {}
    else:
        prepared = dict(payload)
    if contact_email:
        prepared.setdefault("email", contact_email)
    if api_key:
        prepared.setdefault("api_key", api_key)
    return prepared


def dispatch_nih_request(
    *,
    http_client: Optional[httpx.Client],
    method: str,
    base_url: str,
    endpoint: str,
    params: Optional[Mapping[str, Any]] = None,
    data: Optional[Mapping[str, Any]] = None,
    contact_email: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout_seconds: float = 5.0,
) -> httpx.Response:
    method_lower = method.lower()
    url = f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"

    if method_lower == "get":
        request_params = _prepare_payload(params, contact_email=contact_email, api_key=api_key)
        if http_client is not None:
            response = http_client.get(url, params=request_params)
        else:
            with httpx.Client(timeout=timeout_seconds) as client:
                response = client.get(url, params=request_params)
    elif method_lower == "post":
        request_data = _prepare_payload(data, contact_email=contact_email, api_key=api_key)
        if http_client is not None:
            response = http_client.post(url, data=request_data)
        else:
            with httpx.Client(timeout=timeout_seconds) as client:
                response = client.post(url, data=request_data)
    else:
        raise ValueError(f"Unsupported NIH request method: {method}")

    if response.status_code >= 400:
        raise RuntimeError(f"NIH request failed with status {response.status_code} for {endpoint}")

    return response
