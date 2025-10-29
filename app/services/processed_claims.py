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
    severe_reaction_flag: bool = False
    severe_reaction_terms: list[str] = field(default_factory=list)


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
            severe_reaction_flag=aggregated_claim.severe_reaction_flag,
            severe_reaction_terms=list(aggregated_claim.severe_reaction_terms),
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
            drugs, drug_classes = _normalise_claim_terms(
                claim.get("drugs"),
                claim.get("drug_classes"),
            )
            key = _build_claim_key(classification, drugs, drug_classes)

            confidence = (claim.get("confidence") or "low").strip().lower()
            summary = claim.get("summary") or ""
            claim_id = claim.get("claim_id") or key
            severe_flag, severe_terms = _parse_severe_reaction(claim.get("severe_reaction"))

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

            _merge_severe_reaction(aggregated_claim, severe_flag, severe_terms)

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

    return _reduce_redundant_claims(aggregated)


def _parse_severe_reaction(value: object) -> tuple[bool, list[str]]:
    if value is None:
        return False, []
    if isinstance(value, bool):
        return bool(value), []
    if not isinstance(value, dict):
        return False, []

    raw_flag = value.get("flag")
    terms_source = value.get("terms") or []
    cleaned_terms: list[str] = []
    for term in terms_source:
        cleaned = _clean_str(term)
        if cleaned:
            cleaned_terms.append(cleaned)
    unique_terms = list(dict.fromkeys(cleaned_terms))
    flag = bool(raw_flag) or bool(unique_terms)
    return flag, unique_terms


def _merge_severe_reaction(
    aggregated_claim: _AggregatedClaim,
    flag: bool,
    terms: Sequence[str],
) -> None:
    if flag:
        aggregated_claim.severe_reaction_flag = True
    if not terms:
        return
    existing = aggregated_claim.severe_reaction_terms
    seen = {term.lower(): None for term in existing}
    for term in terms:
        key = term.lower()
        if key not in seen:
            existing.append(term)
            seen[key] = None


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
        key = _normalize_term_key(cleaned)
        if key not in seen:
            seen[key] = cleaned
    return list(seen.values())


def _build_claim_key(classification: str, drugs: list[str], drug_classes: list[str]) -> str:
    drugs_key = ",".join(sorted(_normalize_term_key(term) for term in drugs))
    classes_key = ",".join(sorted(_normalize_term_key(term) for term in drug_classes))
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


def _normalise_claim_terms(
    drugs: Iterable[str] | None,
    drug_classes: Iterable[str] | None,
) -> tuple[list[str], list[str]]:
    raw_drugs = drugs or []
    raw_classes = drug_classes or []
    class_terms: list[str] = []
    drug_terms: list[str] = []

    for term in raw_drugs:
        cleaned = _clean_str(term)
        if not cleaned:
            continue
        if _is_generic_class_term(cleaned):
            class_terms.append(cleaned)
        else:
            drug_terms.append(cleaned)

    for term in raw_classes:
        cleaned = _clean_str(term)
        if cleaned:
            class_terms.append(cleaned)

    return _unique_terms(drug_terms), _unique_terms(class_terms)


def _normalize_term_key(term: str) -> str:
    value = term.strip().lower()
    if not value:
        return ""
    value = value.replace("/", " ")
    value = value.replace("-", " ")
    value = value.replace("_", " ")
    value = value.replace("anaesthetic", "anesthetic")
    value = value.replace("anaesthetics", "anesthetics")
    value = value.replace("  ", " ")
    value = " ".join(value.split())
    if value.endswith("s") and len(value) > 4 and " " in value and not value.endswith(("ss", "us")):
        value = value[:-1]
    return value


_GENERIC_SINGLE_WORDS = {
    "agent",
    "agents",
    "class",
    "drug",
    "drugs",
    "therapy",
    "treatment",
    "use",
    "usage",
    "volatile",
    "neuromuscular",
    "anesthetic",
    "anesthetics",
}

_GENERIC_KEYWORDS = {
    "agent",
    "agents",
    "class",
    "classes",
    "therapy",
    "therapies",
    "treatment",
    "treatments",
    "blocker",
    "blockers",
    "blocking",
    "relaxant",
    "relaxants",
    "anesthetic",
    "anesthetics",
    "anaesthetic",
    "anaesthetics",
    "inhalational",
    "volatile",
    "modulator",
    "modulators",
    "myorelaxant",
    "myorelaxants",
}

_GENERIC_PHRASES = {
    "mh therapy",
    "mh treatment",
    "mh trigger",
    "mh triggering agent",
    "generic class",
    "generic-class",
}


def _is_generic_class_term(term: str) -> bool:
    slug = _normalize_term_key(term)
    if not slug:
        return True
    if slug in _GENERIC_PHRASES:
        return True
    tokens = slug.split()
    if len(tokens) == 1:
        return slug in _GENERIC_SINGLE_WORDS
    return any(token in _GENERIC_KEYWORDS for token in tokens)


_TOKEN_NORMALISER = {
    "blocking": "block",
    "blocker": "block",
    "blockers": "block",
    "relaxants": "relaxant",
    "anesthetics": "anesthetic",
    "anaesthetic": "anesthetic",
    "anaesthetics": "anesthetic",
}

_GENERIC_TOKEN_SKIP = {
    "agent",
    "agents",
    "class",
    "classes",
    "therapy",
    "therapies",
    "treatment",
    "treatments",
    "drug",
    "drugs",
    "use",
    "usage",
}


def _normalize_token(token: str) -> str:
    if not token:
        return ""
    value = token.lower().strip()
    if not value:
        return ""
    value = value.replace("anaesthet", "anesthet")
    value = _TOKEN_NORMALISER.get(value, value)
    if value.endswith("s") and len(value) > 4 and not value.endswith("ss"):
        value = value[:-1]
    value = _TOKEN_NORMALISER.get(value, value)
    return value


def _claim_has_specific_drug(claim: _AggregatedClaim) -> bool:
    for term in claim.drugs:
        if not _is_generic_class_term(term):
            return True
    return False


def _claim_term_tokens(claim: _AggregatedClaim) -> set[str]:
    tokens: set[str] = set()
    for term in list(claim.drug_classes) + list(claim.drugs):
        slug = _normalize_term_key(term)
        if slug:
            tokens.add(slug)
        for raw in slug.split():
            token = _normalize_token(raw)
            if token and token not in _GENERIC_TOKEN_SKIP:
                tokens.add(token)
    return tokens


def _reduce_redundant_claims(
    aggregated: "OrderedDict[str, _AggregatedClaim]",
) -> "OrderedDict[str, _AggregatedClaim]":
    if not aggregated:
        return aggregated

    grouped: dict[str, list[tuple[str, _AggregatedClaim]]] = {}
    for key, claim in aggregated.items():
        grouped.setdefault(claim.classification, []).append((key, claim))

    keys_to_remove: set[str] = set()

    for items in grouped.values():
        specific_tokens: list[set[str]] = []
        for key, claim in items:
            if _claim_has_specific_drug(claim):
                tokens = _claim_term_tokens(claim)
                if tokens:
                    specific_tokens.append(tokens)

        if not specific_tokens:
            continue

        combined_specific_tokens = set().union(*specific_tokens)
        if not combined_specific_tokens:
            continue

        for key, claim in items:
            if key in keys_to_remove:
                continue
            if _claim_has_specific_drug(claim):
                continue
            claim_tokens = _claim_term_tokens(claim)
            if not claim_tokens:
                keys_to_remove.add(key)
            elif claim_tokens & combined_specific_tokens:
                keys_to_remove.add(key)

    if not keys_to_remove:
        return aggregated

    return OrderedDict((key, claim) for key, claim in aggregated.items() if key not in keys_to_remove)
