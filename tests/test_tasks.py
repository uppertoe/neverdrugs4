from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import select

from app.models import ClaimSetRefresh
from app.services.nih_pipeline import MeshTermsNotFoundError
from app.services.search import SearchResolution
from app.tasks import PROGRESS_DESCRIPTIONS, refresh_claims_for_condition


@pytest.fixture()
def seeded_refresh(session_factory):
    session = session_factory()
    refresh = ClaimSetRefresh(
        mesh_signature="central-core",
        job_id="test-task-id",
        status="queued",
    )
    session.add(refresh)
    session.commit()
    session.close()
    return refresh


def _settings_stub():
    return SimpleNamespace(
        search=SimpleNamespace(refresh_ttl_seconds=3600),
        article_selection=SimpleNamespace(
            base_full_text_articles=20,
            max_full_text_articles=50,
            pubmed_retmax=120,
        ),
    )


@patch("app.tasks.persist_processed_claims")
@patch("app.tasks.OpenAIChatClient")
@patch("app.tasks.build_llm_batches")
@patch("app.tasks.collect_pubmed_articles")
@patch("app.tasks.resolve_condition_via_nih")
@patch("app.tasks.create_session_factory")
@patch("app.tasks.load_settings", side_effect=_settings_stub)
@pytest.mark.usefixtures("seeded_refresh")
def test_refresh_claims_invokes_pubmed_collection(
    load_settings_mock,
    session_factory_mock,
    resolve_condition_mock,
    collect_articles_mock,
    build_batches_mock,
    openai_client_cls_mock,
    persist_claims_mock,
    session_factory,
):
    session_factory_mock.return_value = session_factory
    fresh_resolution = SearchResolution(
        normalized_condition="central core disease",
        mesh_terms=["Central Core Disease"],
        reused_cached=False,
        search_term_id=101,
    )
    resolve_condition_mock.return_value = fresh_resolution

    build_batches_mock.return_value = [SimpleNamespace()]
    client_instance = MagicMock()
    client_instance.run_batches.return_value = [SimpleNamespace()]
    openai_client_cls_mock.return_value = client_instance
    persist_claims_mock.return_value = SimpleNamespace(claims=[SimpleNamespace()])

    result = refresh_claims_for_condition(
        resolution_id=1,
        condition_label="Central Core Disease",
        normalized_condition="central core disease",
        mesh_terms=["Central Core Disease"],
        mesh_signature="central-core",
        job_id="test-task-id",
        session_factory=session_factory,
    )

    assert result == "completed"
    collect_articles_mock.assert_called_once()
    build_batches_mock.assert_called_once()
    assert collect_articles_mock.call_args[0][0] is fresh_resolution
    assert build_batches_mock.call_args.kwargs["search_term_id"] == fresh_resolution.search_term_id
    client_instance.run_batches.assert_called_once()
    run_args, run_kwargs = client_instance.run_batches.call_args
    assert run_args[0] == build_batches_mock.return_value
    assert callable(run_kwargs.get("progress_callback"))
    persist_claims_mock.assert_called_once()


@patch("app.tasks.OpenAIChatClient")
@patch("app.tasks.build_llm_batches", return_value=[])
@patch("app.tasks.collect_pubmed_articles")
@patch("app.tasks.resolve_condition_via_nih")
@patch("app.tasks.create_session_factory")
@patch("app.tasks.load_settings", side_effect=_settings_stub)
@pytest.mark.usefixtures("seeded_refresh")
def test_refresh_claims_returns_no_batches_after_collection(
    load_settings_mock,
    session_factory_mock,
    resolve_condition_mock,
    collect_articles_mock,
    build_batches_mock,
    openai_client_cls_mock,
    session_factory,
):
    session_factory_mock.return_value = session_factory
    fresh_resolution = SearchResolution(
        normalized_condition="central core disease",
        mesh_terms=["Central Core Disease"],
        reused_cached=False,
        search_term_id=202,
    )
    resolve_condition_mock.return_value = fresh_resolution

    result = refresh_claims_for_condition(
        resolution_id=2,
        condition_label="Central Core Disease",
        normalized_condition="central core disease",
        mesh_terms=["Central Core Disease"],
        mesh_signature="central-core",
        job_id="test-task-id",
        session_factory=session_factory,
    )

    assert result == "no-batches"
    collect_articles_mock.assert_called_once()
    build_batches_mock.assert_called_once()
    openai_client_cls_mock.assert_not_called()

    session = session_factory()
    try:
        refresh = session.execute(
            select(ClaimSetRefresh).where(ClaimSetRefresh.job_id == "test-task-id")
        ).scalar_one()
        assert refresh.status == "no-batches"
        assert refresh.progress_payload.get("reason") == "no_llm_batches"
        assert refresh.progress_payload.get("mesh_terms") == ["Central Core Disease"]
        assert refresh.progress_payload.get("normalized_condition") == "central core disease"
    finally:
        session.close()


@patch("app.tasks.persist_processed_claims")
@patch("app.tasks.OpenAIChatClient")
@patch("app.tasks.build_llm_batches")
@patch("app.tasks.collect_pubmed_articles")
@patch("app.tasks.resolve_condition_via_nih")
@patch("app.tasks.create_session_factory")
@patch("app.tasks.load_settings", side_effect=_settings_stub)
@pytest.mark.usefixtures("seeded_refresh")
def test_refresh_claims_records_empty_results(
    load_settings_mock,
    session_factory_mock,
    resolve_condition_mock,
    collect_articles_mock,
    build_batches_mock,
    openai_client_cls_mock,
    persist_claims_mock,
    session_factory,
):
    session_factory_mock.return_value = session_factory
    fresh_resolution = SearchResolution(
        normalized_condition="central core disease",
        mesh_terms=["Central Core Disease"],
        reused_cached=False,
        search_term_id=404,
    )
    resolve_condition_mock.return_value = fresh_resolution

    build_batches_mock.return_value = [SimpleNamespace()]
    client_instance = MagicMock()
    client_instance.run_batches.return_value = [SimpleNamespace()]
    openai_client_cls_mock.return_value = client_instance
    persist_claims_mock.return_value = SimpleNamespace(claims=[])

    result = refresh_claims_for_condition(
        resolution_id=4,
        condition_label="Central Core Disease",
        normalized_condition="central core disease",
        mesh_terms=["Central Core Disease"],
        mesh_signature="central-core",
        job_id="test-task-id",
        session_factory=session_factory,
    )

    assert result == "empty-results"

    session = session_factory()
    try:
        refresh = session.execute(
            select(ClaimSetRefresh).where(ClaimSetRefresh.job_id == "test-task-id")
        ).scalar_one()
        assert refresh.status == "empty-results"
        assert refresh.progress_state == "empty_results"
        assert refresh.progress_payload.get("claim_count") == 0
    finally:
        session.close()

    client_instance.run_batches.assert_called_once()
    _, run_kwargs = client_instance.run_batches.call_args
    assert callable(run_kwargs.get("progress_callback"))


@patch("app.tasks.persist_processed_claims")
@patch("app.tasks.OpenAIChatClient")
@patch("app.tasks.build_llm_batches")
@patch("app.tasks.collect_pubmed_articles")
@patch("app.tasks.resolve_condition_via_nih")
@patch("app.tasks.create_session_factory")
@patch("app.tasks.load_settings", side_effect=_settings_stub)
@patch("app.tasks._update_refresh_progress")
@pytest.mark.usefixtures("seeded_refresh")
def test_refresh_claims_reports_batch_progress(
    update_progress_mock,
    load_settings_mock,
    session_factory_mock,
    resolve_condition_mock,
    collect_articles_mock,
    build_batches_mock,
    openai_client_cls_mock,
    persist_claims_mock,
    session_factory,
):
    session_factory_mock.return_value = session_factory
    fresh_resolution = SearchResolution(
        normalized_condition="central core disease",
        mesh_terms=["Central Core Disease"],
        reused_cached=False,
        search_term_id=505,
    )
    resolve_condition_mock.return_value = fresh_resolution

    batches = [SimpleNamespace(id=index) for index in range(3)]
    build_batches_mock.return_value = batches

    client_instance = MagicMock()

    def _fake_run(batches_arg, progress_callback=None):
        if progress_callback is not None:
            progress_callback(1, len(batches_arg), batches_arg[0])
            progress_callback(len(batches_arg), len(batches_arg), batches_arg[-1])
        return [SimpleNamespace() for _ in batches_arg]

    client_instance.run_batches.side_effect = _fake_run
    openai_client_cls_mock.return_value = client_instance
    persist_claims_mock.return_value = SimpleNamespace(claims=[SimpleNamespace()])

    refresh_claims_for_condition(
        resolution_id=5,
        condition_label="Central Core Disease",
        normalized_condition="central core disease",
        mesh_terms=["Central Core Disease"],
        mesh_signature="central-core",
        job_id="test-task-id",
        session_factory=session_factory,
    )

    generating_calls = [
        call
        for call in update_progress_mock.call_args_list
        if call.kwargs.get("stage") == "generating_claims"
    ]

    assert len(generating_calls) >= 3
    initial_details = generating_calls[0].kwargs["details"]
    assert initial_details["batch_count"] == 3
    assert initial_details["batches_completed"] == 0
    assert initial_details["estimated_remaining_seconds"] == 3 * 150

    final_details = generating_calls[-1].kwargs["details"]
    assert final_details["batches_completed"] == 3
    assert final_details["estimated_remaining_seconds"] == 0

@patch("app.tasks.collect_pubmed_articles")
@patch("app.tasks.resolve_condition_via_nih")
@patch("app.tasks.create_session_factory")
@patch("app.tasks.load_settings", side_effect=_settings_stub)
@pytest.mark.usefixtures("seeded_refresh")
def test_refresh_claims_skips_when_mesh_terms_missing(
    load_settings_mock,
    session_factory_mock,
    resolve_condition_mock,
    collect_articles_mock,
    session_factory,
):
    session_factory_mock.return_value = session_factory
    resolve_condition_mock.side_effect = MeshTermsNotFoundError(
        normalized_condition="unknown", search_term_id=303, suggestions=["foo", "bar"]
    )

    result = refresh_claims_for_condition(
        resolution_id=3,
        condition_label="Unknown",
        normalized_condition="unknown",
        mesh_terms=["Unknown"],
        mesh_signature="unknown",
        job_id="test-task-id",
        session_factory=session_factory,
    )

    assert result == "skipped"
    collect_articles_mock.assert_not_called()

    session = session_factory()
    try:
        refresh = session.execute(
            select(ClaimSetRefresh).where(ClaimSetRefresh.job_id == "test-task-id")
        ).scalar_one()
        assert refresh.status == "skipped"
        assert refresh.progress_state == "skipped"
        assert refresh.progress_payload == {
            "description": PROGRESS_DESCRIPTIONS["skipped"],
            "reason": "missing_mesh_terms",
            "suggestions": ["foo", "bar"],
        }
    finally:
        session.close()
