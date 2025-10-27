from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import (
    ArticleSnippet,
    ProcessedClaim,
    ProcessedClaimDrugLink,
    ProcessedClaimEvidence,
    ProcessedClaimSet,
)

_CONFIDENCE_ORDER = {"low": 0, "medium": 1, "high": 2}


@dataclass(slots=True)
class _AggregatedEvidence:
    snippet_id: str
    pmid: str | None
    article_title: str | None
    key_points: list[str]
    notes: str | None


@dataclass(slots=True)
class _AggregatedClaim:
    claim_id: str
    classification: str
    summary: str
    confidence: str
    drugs: list[str]
    drug_classes: list[str]
    source_claim_ids: list[str] = field(default_factory=list)
    evidence: "OrderedDict[str, _AggregatedEvidence]" = field(default_factory=OrderedDict)


def persist_processed_claims(
    session: Session,
    *,
    search_term_id: int,
    mesh_signature: str,
    condition_label: str,
    llm_payloads: Sequence[object],
) -> ProcessedClaimSet:
    aggregated = _aggregate_claims(llm_payloads)

    claim_set = (
        session.execute(
            select(ProcessedClaimSet).where(ProcessedClaimSet.mesh_signature == mesh_signature)
        ).scalar_one_or_none()
    )

    if claim_set is None:
        claim_set = ProcessedClaimSet(
            mesh_signature=mesh_signature,
            condition_label=condition_label,
            last_search_term_id=search_term_id,
        )
        session.add(claim_set)
    else:
        claim_set.condition_label = condition_label
        claim_set.last_search_term_id = search_term_id
        if claim_set.claims:
            claim_set.claims.clear()
            session.flush()

    snippet_lookup = _load_snippets(session, aggregated)

    for aggregated_claim in aggregated.values():
        stored_claim = ProcessedClaim(
            claim_set=claim_set,
            claim_id=aggregated_claim.claim_id,
            classification=aggregated_claim.classification,
            summary=aggregated_claim.summary,
            confidence=aggregated_claim.confidence,
            drugs=list(aggregated_claim.drugs),
            drug_classes=list(aggregated_claim.drug_classes),
            source_claim_ids=list(aggregated_claim.source_claim_ids),
        )
        session.add(stored_claim)

        for evidence in aggregated_claim.evidence.values():
            snippet = snippet_lookup.get(evidence.snippet_id)
            pmid = evidence.pmid or (snippet.article.pmid if snippet is not None else None)
            article_title = evidence.article_title
            citation_url: str | None = None

            if snippet is not None:
                article = snippet.article
                if not article_title:
                    citation = article.citation if isinstance(article.citation, dict) else {}
                    article_title = citation.get("title")
                if citation_url is None and isinstance(article.citation, dict):
                    citation_url = article.citation.get("preferred_url")
                if citation_url is None:
                    citation_url = f"https://pubmed.ncbi.nlm.nih.gov/{article.pmid}/"
            elif pmid:
                citation_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

            processed_evidence = ProcessedClaimEvidence(
                claim=stored_claim,
                snippet_id=evidence.snippet_id,
                pmid=pmid or "",
                article_title=article_title,
                citation_url=citation_url,
                key_points=list(evidence.key_points),
                notes=evidence.notes,
            )
            session.add(processed_evidence)

        for drug in aggregated_claim.drugs:
            session.add(
                ProcessedClaimDrugLink(
                    claim=stored_claim,
                    term=drug,
                    term_kind="drug",
                )
            )
        for drug_class in aggregated_claim.drug_classes:
            session.add(
                ProcessedClaimDrugLink(
                    claim=stored_claim,
                    term=drug_class,
                    term_kind="drug_class",
                )
            )

    session.flush()
    return claim_set


def _aggregate_claims(llm_payloads: Sequence[object]) -> "OrderedDict[str, _AggregatedClaim]":
    aggregated: "OrderedDict[str, _AggregatedClaim]" = OrderedDict()

    for payload in llm_payloads:
        data = _coerce_payload(payload)
        if not data:
            continue
        claims = data.get("claims") or []
        for claim in claims:
            classification = (claim.get("classification") or "").strip().lower()
            if not classification:
                continue
            drugs = _unique_terms(claim.get("drugs") or [])
            drug_classes = _unique_terms(claim.get("drug_classes") or [])
            key = _build_claim_key(classification, drugs, drug_classes)

            confidence = (claim.get("confidence") or "low").strip().lower()
            summary = claim.get("summary") or ""
            claim_id = claim.get("claim_id") or key

            aggregated_claim = aggregated.get(key)
            if aggregated_claim is None:
                aggregated_claim = _AggregatedClaim(
                    claim_id=claim_id,
                    classification=classification,
                    summary=summary,
                    confidence=confidence,
                    drugs=drugs,
                    drug_classes=drug_classes,
                )
                aggregated[key] = aggregated_claim
            else:
                _update_claim_metadata(aggregated_claim, claim_id, summary, confidence)

            if claim_id not in aggregated_claim.source_claim_ids:
                aggregated_claim.source_claim_ids.append(claim_id)

            for evidence in claim.get("supporting_evidence") or []:
                snippet_id_raw = evidence.get("snippet_id")
                if snippet_id_raw is None:
                    continue
                snippet_id = str(snippet_id_raw)
                if snippet_id not in aggregated_claim.evidence:
                    aggregated_claim.evidence[snippet_id] = _AggregatedEvidence(
                        snippet_id=snippet_id,
                        pmid=_clean_str(evidence.get("pmid")),
                        article_title=_clean_str(evidence.get("article_title")),
                        key_points=[point for point in (evidence.get("key_points") or []) if point],
                        notes=_clean_str(evidence.get("notes")),
                    )

    return aggregated


def _coerce_payload(payload: object) -> dict:
    if payload is None:
        return {}
    if isinstance(payload, dict):
        return payload
    parsed = getattr(payload, "parsed_json", None)
    if callable(parsed):
        try:
            result = parsed()
        except Exception:  # noqa: BLE001 - defensive: wrapping external client objects
            return {}
        return result if isinstance(result, dict) else {}
    return {}


def _unique_terms(terms: Iterable[str]) -> list[str]:
    seen: OrderedDict[str, str] = OrderedDict()
    for term in terms:
        if term is None:
            continue
        cleaned = term.strip()
        if not cleaned:
            continue
        key = cleaned.lower()
        if key not in seen:
            seen[key] = cleaned
    return list(seen.values())


def _build_claim_key(classification: str, drugs: list[str], drug_classes: list[str]) -> str:
    drugs_key = ",".join(sorted(term.lower() for term in drugs))
    classes_key = ",".join(sorted(term.lower() for term in drug_classes))
    return f"{classification}|{drugs_key}|{classes_key}"


def _update_claim_metadata(
    aggregated_claim: _AggregatedClaim,
    new_claim_id: str,
    new_summary: str,
    new_confidence: str,
) -> None:
    current_rank = _CONFIDENCE_ORDER.get(aggregated_claim.confidence, -1)
    new_rank = _CONFIDENCE_ORDER.get(new_confidence, -1)

    if new_rank > current_rank:
        aggregated_claim.claim_id = new_claim_id
        aggregated_claim.summary = new_summary
        aggregated_claim.confidence = new_confidence


def _load_snippets(
    session: Session,
    aggregated: "OrderedDict[str, _AggregatedClaim]",
) -> dict[str, ArticleSnippet]:
    snippet_ids: set[str] = set()
    for claim in aggregated.values():
        snippet_ids.update(claim.evidence.keys())

    numeric_ids = [int(sid) for sid in snippet_ids if sid.isdigit()]
    if not numeric_ids:
        return {}

    rows = session.execute(
        select(ArticleSnippet).where(ArticleSnippet.id.in_(numeric_ids))
    ).scalars()
    return {str(snippet.id): snippet for snippet in rows}


def _clean_str(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned or None
    return str(value)
