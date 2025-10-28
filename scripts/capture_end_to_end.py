#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from dotenv import load_dotenv
from sqlalchemy.orm import Session, selectinload, sessionmaker

from app.database import Base, create_engine_for_url
from app.models import (
    ArticleArtefact,
    ProcessedClaim,
    ProcessedClaimSet,
    SearchArtefact,
)
from app.services.full_text import FullTextSelectionPolicy, NIHFullTextFetcher, collect_pubmed_articles
from app.services.llm_batches import LLMRequestBatch, build_llm_batches
from app.services.nih_pipeline import MeshTermsNotFoundError, resolve_condition_via_nih
from app.services.nih_pubmed import NIHPubMedSearcher
from app.services.openai_client import OpenAIChatClient
from app.services.processed_claims import persist_processed_claims
from app.services.search import compute_mesh_signature
from app.settings import load_settings


@contextmanager
def _record_stage(label: str, timings: dict[str, float]) -> Iterator[None]:
    print(f"-> {label}...", file=sys.stderr)
    start = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start
        timings[label] = duration
        print(f"   {label} completed in {duration:0.2f}s", file=sys.stderr)


def _serialise_batch(batch: LLMRequestBatch) -> dict[str, object]:
    return {
        "token_estimate": batch.token_estimate,
        "messages": batch.messages,
        "snippets": [
            {
                "pmid": entry.pmid,
                "snippet_id": entry.snippet_id,
                "drug": entry.drug,
                "classification": entry.classification,
                "snippet_text": entry.snippet_text,
                "snippet_score": entry.snippet_score,
                "cues": entry.cues,
                "article_rank": entry.article_rank,
                "article_score": entry.article_score,
                "citation_url": entry.citation_url,
                "article_title": entry.article_title,
                "content_source": entry.content_source,
                "token_estimate": entry.token_estimate,
            }
            for entry in batch.snippets
        ],
        "claim_groups": [
            {
                "group_key": group.group_key,
                "classification": group.classification,
                "drug_label": group.drug_label,
                "drug_terms": list(group.drug_terms),
                "drug_classes": list(group.drug_classes),
                "snippet_ids": list(group.snippet_ids),
                "pmids": list(group.pmids),
                "top_score": group.top_score,
            }
            for group in batch.claim_groups
        ],
    }


def _collect_snippet_snapshot(session: Session, *, search_term_id: int) -> list[dict[str, object]]:
    artefacts = (
        session.query(ArticleArtefact)
        .options(selectinload(ArticleArtefact.snippets))
        .filter(ArticleArtefact.search_term_id == search_term_id)
        .order_by(ArticleArtefact.rank)
        .all()
    )
    snapshot: list[dict[str, object]] = []
    for artefact in artefacts:
        for snippet in artefact.snippets:
            snapshot.append(
                {
                    "snippet_id": snippet.id,
                    "pmid": artefact.pmid,
                    "drug": snippet.drug,
                    "classification": snippet.classification,
                    "snippet_text": snippet.snippet_text,
                    "snippet_score": snippet.snippet_score,
                    "cues": snippet.cues,
                    "article_rank": artefact.rank,
                    "article_score": artefact.score,
                    "article_title": artefact.citation.get("title") if isinstance(artefact.citation, dict) else None,
                    "citation_url": artefact.citation.get("preferred_url") if isinstance(artefact.citation, dict) else None,
                    "content_source": artefact.content_source,
                }
            )
    return snapshot


def _collect_article_snapshot(session: Session, *, search_term_id: int) -> list[dict[str, object]]:
    artefacts = (
        session.query(ArticleArtefact)
        .filter(ArticleArtefact.search_term_id == search_term_id)
        .order_by(ArticleArtefact.rank)
        .all()
    )
    snapshot: list[dict[str, object]] = []
    for artefact in artefacts:
        citation = artefact.citation if isinstance(artefact.citation, dict) else {}
        snapshot.append(
            {
                "pmid": artefact.pmid,
                "rank": artefact.rank,
                "score": artefact.score,
                "content_source": artefact.content_source,
                "has_content": bool(artefact.content),
                "token_estimate": artefact.token_estimate,
                "preferred_url": citation.get("preferred_url"),
                "title": citation.get("title"),
            }
        )
    return snapshot


def _collect_processed_claims_snapshot(session: Session, mesh_signature: str | None) -> list[dict[str, object]]:
    if not mesh_signature:
        return []

    claim_set = (
        session.query(ProcessedClaimSet)
        .filter(ProcessedClaimSet.mesh_signature == mesh_signature)
        .options(
            selectinload(ProcessedClaimSet.claims)
            .selectinload(ProcessedClaim.evidence)
        )
        .options(selectinload(ProcessedClaimSet.claims).selectinload(ProcessedClaim.drug_links))
        .one_or_none()
    )

    if claim_set is None:
        return []

    claims_payload: list[dict[str, object]] = []
    for claim in claim_set.claims:
        claims_payload.append(
            {
                "claim_id": claim.claim_id,
                "classification": claim.classification,
                "summary": claim.summary,
                "confidence": claim.confidence,
                "drugs": list(claim.drugs),
                "drug_classes": list(claim.drug_classes),
                "source_claim_ids": list(claim.source_claim_ids),
                "supporting_evidence": [
                    {
                        "snippet_id": evidence.snippet_id,
                        "pmid": evidence.pmid,
                        "article_title": evidence.article_title,
                        "citation_url": evidence.citation_url,
                        "key_points": list(evidence.key_points),
                        "notes": evidence.notes,
                    }
                    for evidence in claim.evidence
                ],
                "drug_links": [
                    {"term": link.term, "term_kind": link.term_kind}
                    for link in claim.drug_links
                ],
            }
        )

    return claims_payload


def capture_end_to_end(
    *,
    search_term: str,
    output_path: Path,
    max_pubmed_results: int | None = None,
    max_full_text_articles: int | None = None,
) -> dict[str, object]:
    load_dotenv()
    engine = create_engine_for_url("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, future=True)

    session = SessionLocal()
    try:
        timings: dict[str, float] = {}
        started_at = time.perf_counter()
        settings = load_settings()
        article_defaults = settings.article_selection

        with _record_stage("resolve_condition_via_nih", timings):
            try:
                resolution = resolve_condition_via_nih(
                    search_term,
                    session=session,
                    refresh_ttl_seconds=settings.search.refresh_ttl_seconds,
                )
            except MeshTermsNotFoundError as exc:
                suggestion_text = ", ".join(exc.suggestions) if exc.suggestions else "(no suggestions available)"
                raise RuntimeError(
                    f"No MeSH terms found for '{search_term}'. Suggested alternatives: {suggestion_text}"
                ) from exc

        artefact = (
            session.query(SearchArtefact)
            .filter(SearchArtefact.search_term_id == resolution.search_term_id)
            .order_by(SearchArtefact.id.desc())
            .first()
        )

        resolved_pubmed_limit = max_pubmed_results or article_defaults.pubmed_retmax
        resolved_full_text_cap = max_full_text_articles or article_defaults.max_full_text_articles
        pubmed_searcher = NIHPubMedSearcher(retmax=resolved_pubmed_limit)
        selection_policy = FullTextSelectionPolicy(
            base_full_text=article_defaults.base_full_text_articles,
            max_full_text=resolved_full_text_cap,
            max_token_budget=article_defaults.full_text_token_budget,
            estimated_tokens_per_full_text=article_defaults.estimated_tokens_per_article,
        )
        full_text_fetcher = NIHFullTextFetcher()

        selected_mesh_terms = list(resolution.mesh_terms)
        with _record_stage("pubmed_search", timings):
            search_result = pubmed_searcher(
                selected_mesh_terms,
                additional_text_terms=[search_term, resolution.normalized_condition],
            )

        fallback_used = False
        if not search_result.articles:
            extra_terms = []
            if artefact is not None:
                esummary_terms = artefact.query_payload.get("esummary", {}).get("mesh_terms", [])
                for term in esummary_terms:
                    if term not in selected_mesh_terms and term not in extra_terms:
                        extra_terms.append(term)
            if extra_terms:
                selected_mesh_terms = selected_mesh_terms + extra_terms
                with _record_stage("pubmed_search_fallback", timings):
                    search_result = pubmed_searcher(
                        selected_mesh_terms,
                        additional_text_terms=[search_term, resolution.normalized_condition],
                    )
                fallback_used = True

        def _search(mesh_terms: list[str]):  # noqa: ARG001 - signature matches collector expectations
            return search_result

        with _record_stage("collect_pubmed_articles", timings):
            collect_pubmed_articles(
                resolution,
                session=session,
                pubmed_searcher=_search,
                full_text_fetcher=full_text_fetcher,
                selection_policy=selection_policy,
            )

        with _record_stage("build_llm_batches", timings):
            batches = build_llm_batches(
                session,
                search_term_id=resolution.search_term_id,
                condition_label=search_term,
                mesh_terms=resolution.mesh_terms,
            )

        client = OpenAIChatClient()
        with _record_stage("openai_run_batches", timings):
            responses = client.run_batches(batches)

        mesh_signature = compute_mesh_signature(resolution.mesh_terms)
        if responses and mesh_signature:
            with _record_stage("persist_processed_claims", timings):
                persist_processed_claims(
                    session,
                    search_term_id=resolution.search_term_id,
                    mesh_signature=mesh_signature,
                    condition_label=search_term,
                    llm_payloads=responses,
                )

        with _record_stage("collect_snapshot", timings):
            snippet_snapshot = _collect_snippet_snapshot(session, search_term_id=resolution.search_term_id)
            article_snapshot = _collect_article_snapshot(session, search_term_id=resolution.search_term_id)
            processed_snapshot = _collect_processed_claims_snapshot(session, mesh_signature)

        payload = {
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "search_term": search_term,
            "search_resolution": {
                "normalized_condition": resolution.normalized_condition,
                "mesh_terms": resolution.mesh_terms,
                "reused_cached": resolution.reused_cached,
                "search_term_id": resolution.search_term_id,
                "query_payload": artefact.query_payload if artefact is not None else None,
                "mesh_terms_used": selected_mesh_terms,
            },
            "pubmed_search": {
                "query": search_result.query,
                "pmids": search_result.pmids,
                "article_count": len(search_result.articles),
                "fallback_used": fallback_used,
            },
            "articles": article_snapshot,
            "snippets": snippet_snapshot,
            "batches": [_serialise_batch(batch) for batch in batches],
            "responses": [
                {
                    "model": result.model,
                    "response_id": result.response_id,
                    "content": result.content,
                    "parsed": result.parsed_json(),
                    "usage": result.usage,
                }
                for result in responses
            ],
            "processed_claims": processed_snapshot,
        }

        with _record_stage("write_output", timings):
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        timings["total_runtime"] = time.perf_counter() - started_at
        payload["timings"] = timings
        return payload
    finally:
        session.close()
        engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture an end-to-end pipeline run for a condition search term."
    )
    parser.add_argument(
        "search_term",
        nargs="?",
        help="Search term or condition to resolve (positional alternative to --condition)",
    )
    parser.add_argument(
        "--condition",
        dest="condition",
        help="Search term or condition to resolve",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tests/fixtures/e2e_capture.json"),
        help="Path to write the captured fixture JSON",
    )
    parser.add_argument(
        "--max-pubmed-results",
        type=int,
        default=None,
        help="Maximum PubMed results to request (defaults to settings)",
    )
    parser.add_argument(
        "--max-full-text-articles",
        type=int,
        default=None,
        help="Maximum number of articles to fetch in full (defaults to settings)",
    )
    parser.add_argument(
        "--show-timings",
        action="store_true",
        help="Print a breakdown of major stage durations",
    )
    args = parser.parse_args()

    search_term = args.condition or args.search_term
    if not search_term:
        parser.error("Expected a condition via positional SEARCH_TERM or --condition")

    payload = capture_end_to_end(
        search_term=search_term,
        output_path=args.output,
        max_pubmed_results=args.max_pubmed_results,
        max_full_text_articles=args.max_full_text_articles,
    )
    if args.show_timings:
        timings = payload.get("timings", {})
        if timings:
            print("Timing breakdown (seconds):", file=sys.stderr)
            for label, duration in sorted(timings.items(), key=lambda item: item[1], reverse=True):
                print(f"  {label:>26}: {duration:0.2f}", file=sys.stderr)
    print(
        "Captured end-to-end fixture with "
        f"{len(payload['articles'])} articles and {len(payload['snippets'])} snippets "
        f"to {args.output}"
    )


if __name__ == "__main__":
    main()
