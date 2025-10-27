from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from urllib.parse import quote

import httpx
import pytest

from app.services.search import compute_mesh_signature, normalize_condition

pytestmark = pytest.mark.integration

_BASE_URL = os.getenv("INTEGRATION_BASE_URL", "http://localhost:8000")
_ALEMBIC_RAN_MARKER = "/tmp/alembic_ran"


def _wait_for_service(path: str, *, timeout: float = 30.0) -> httpx.Client:
    deadline = time.time() + timeout
    client = httpx.Client(base_url=_BASE_URL, timeout=10.0)
    while time.time() < deadline:
        try:
            response = client.get(path)
            if response.status_code in {200, 400, 404, 500}:
                return client
        except httpx.RequestError:
            time.sleep(1.0)
    client.close()
    pytest.skip("HTTP service for integration tests is not reachable")


def test_resolve_requires_condition() -> None:
    client = _wait_for_service("/")
    try:
        response = client.post("/api/claims/resolve", json={})
    finally:
        client.close()
    assert response.status_code == 400
    payload = response.json()
    assert payload == {"detail": "Condition is required"}


def test_refresh_unknown_signature_returns_404() -> None:
    client = _wait_for_service("/api/claims/refresh/unknown-signature")
    try:
        response = client.get("/api/claims/refresh/unknown-signature")
    finally:
        client.close()
    assert response.status_code == 404
    assert response.json() == {"detail": "Refresh job not found"}


def _require_curl() -> None:
    if shutil.which("curl") is None:
        pytest.skip("curl executable not available")


def _curl_json(
    url: str,
    *,
    method: str = "GET",
    data: dict | None = None,
    timeout: float = 60.0,
) -> dict:
    command: list[str] = ["curl", "-sS", "--max-time", str(float(timeout))]
    verb = method.upper()
    if verb != "GET":
        command.extend(["-X", verb])
    if data is not None:
        body = json.dumps(data)
        command.extend(["-H", "Content-Type: application/json", "-d", body])
    command.append(url)

    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.fail(f"curl failed ({result.returncode}): {result.stderr.strip()}")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        pytest.fail(f"curl returned non-JSON response: {exc}\n{result.stdout}")


def test_resolve_and_poll_via_curl() -> None:
    _require_curl()
    client = _wait_for_service("/health")
    client.close()

    condition = os.getenv("INTEGRATION_TEST_CONDITION", "duchenne muscular dystrophy")
    expected_normalized = normalize_condition(condition)

    resolve_payload = _curl_json(
        f"{_BASE_URL}/api/claims/resolve",
        method="POST",
        data={"condition": condition, "force_refresh": True},
    )

    resolution = resolve_payload.get("resolution") or {}
    assert resolution.get("normalized_condition") == expected_normalized
    mesh_terms = resolution.get("mesh_terms") or []
    assert mesh_terms, "resolve response must include mesh terms"

    claim_set = resolve_payload.get("claim_set")
    job_info = resolve_payload.get("job")

    if claim_set is not None and job_info is None:
        assert claim_set.get("claims"), "cached claim set should contain claims"
        return

    assert isinstance(job_info, dict), "resolve expected to queue a background job"
    mesh_signature = compute_mesh_signature(mesh_terms)
    encoded_signature = quote(mesh_signature, safe="")

    timeout_seconds = float(os.getenv("INTEGRATION_POLL_TIMEOUT", "600"))
    deadline = time.time() + timeout_seconds
    final_payload: dict | None = None

    while time.time() < deadline:
        time.sleep(3.0)
        candidate = _curl_json(f"{_BASE_URL}/api/claims/refresh/{encoded_signature}")
        status = candidate.get("status")
        if status not in {"queued", "running"}:
            final_payload = candidate
            break

    if final_payload is None:
        pytest.fail("Timed out waiting for refresh job to finish")

    status = final_payload.get("status")
    assert status in {"completed", "no-batches", "no-responses"}, status

    final_resolution = final_payload.get("resolution") or {}
    assert final_resolution.get("normalized_condition") == expected_normalized
    assert final_resolution.get("mesh_terms")

    if status == "completed":
        claim_set_id = final_payload.get("claim_set_id")
        assert claim_set_id, "completed job should expose a claim_set_id"
        claim_set_payload = _curl_json(f"{_BASE_URL}/api/claims/{claim_set_id}")
        claims = claim_set_payload.get("claims") or []
        assert claims, "processed claim set should include at least one claim"
    elif status == "no-batches":
        details = (final_payload.get("progress") or {}).get("details", {})
        assert details.get("reason") == "no_llm_batches"

    second_payload = _curl_json(
        f"{_BASE_URL}/api/claims/resolve",
        method="POST",
        data={"condition": condition},
    )

    second_resolution = second_payload.get("resolution") or {}
    assert second_resolution.get("normalized_condition") == expected_normalized

    if status == "completed":
        cached_claim_set = second_payload.get("claim_set")
        assert cached_claim_set is not None, "cached resolve should return claim set"
        assert cached_claim_set.get("claims"), "cached claim set should include claims"
        assert second_payload.get("job") is None
        assert second_resolution.get("reused_cached") is True
    else:
        assert second_payload.get("claim_set") is None
        assert isinstance(second_payload.get("job"), dict), "expected follow-up job when cache empty"
