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
_CLAIM_TYPES = {"risk", "safety", "uncertain", "nuanced"}


class InvalidClaimPayload(ValueError):
    """Raised when the LLM payload violates the enforced claim schema."""


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


@dataclass(frozen=True)
class _DrugCatalogEntry:
    drug_id: str
    name: str
    classes: tuple[str, ...]


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
        drug_catalog, claim_to_drugs = _build_drug_catalog(data.get("drugs"))
        declared_claim_ids = set(claim_to_drugs.keys())
        claims = data.get("claims") or []
        for claim in claims:
            if not isinstance(claim, dict):
                raise InvalidClaimPayload("LLM claim payload contains a non-object claim entry")

            claim_id_raw = _clean_str(claim.get("id"))
            if not claim_id_raw:
                raise InvalidClaimPayload("LLM claim payload missing required field 'id'")
            claim_id = claim_id_raw

            classification_raw = _clean_str(claim.get("type"))
            if not classification_raw:
                raise InvalidClaimPayload("LLM claim payload missing required field 'type'")
            classification = classification_raw.lower()
            if classification not in _CLAIM_TYPES:
                raise InvalidClaimPayload(
                    f"LLM claim payload has invalid type '{classification_raw}'"
                )

            summary = _clean_str(claim.get("summary"))
            if not summary:
                raise InvalidClaimPayload("LLM claim payload missing required field 'summary'")

            confidence_raw = _clean_str(claim.get("confidence"))
            if not confidence_raw:
                raise InvalidClaimPayload("LLM claim payload missing required field 'confidence'")
            confidence = confidence_raw.lower()
            if confidence not in _CONFIDENCE_ORDER:
                raise InvalidClaimPayload(
                    f"LLM claim payload has invalid confidence '{confidence_raw}'"
                )

            if "drugs" not in claim or claim.get("drugs") is None:
                raise InvalidClaimPayload("LLM claim payload missing required field 'drugs'")
            drugs, drug_classes, canonical_drug_ids = _normalise_claim_terms(
                claim.get("drugs"),
                claim.get("drug_classes"),
                drug_catalog=drug_catalog,
            )
            if not drugs:
                raise InvalidClaimPayload("LLM claim payload did not resolve any drug terms")

            if claim_id not in claim_to_drugs:
                raise InvalidClaimPayload(
                    f"Claim '{claim_id}' is not declared in the top-level drugs claims list"
                )

            declared_drugs = claim_to_drugs[claim_id]
            for canonical_id in canonical_drug_ids:
                if canonical_id not in declared_drugs:
                    raise InvalidClaimPayload(
                        f"Claim '{claim_id}' references drug '{canonical_id}' without declaration"
                    )

            declared_claim_ids.discard(claim_id)
            key = _build_claim_key(classification, drugs, drug_classes)

            reaction_payload = claim.get("idiosyncratic_reaction")
            if reaction_payload is None:
                raise InvalidClaimPayload(
                    "LLM claim payload missing required field 'idiosyncratic_reaction'"
                )
            if not isinstance(reaction_payload, dict):
                raise InvalidClaimPayload(
                    "LLM claim payload 'idiosyncratic_reaction' must be an object"
                )
            severe_flag, severe_terms = _parse_severe_reaction(reaction_payload)

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
                if not isinstance(evidence, dict):
                    raise InvalidClaimPayload("LLM claim payload supporting_evidence must be objects")

                snippet_id_raw = _clean_str(evidence.get("snippet_id"))
                if not snippet_id_raw:
                    raise InvalidClaimPayload(
                        "LLM claim payload supporting_evidence missing 'snippet_id'"
                    )
                snippet_id = snippet_id_raw

                pmid = _clean_str(evidence.get("pmid"))
                if not pmid:
                    raise InvalidClaimPayload(
                        "LLM claim payload supporting_evidence missing 'pmid'"
                    )

                article_title = _clean_str(evidence.get("article_title"))
                if not article_title:
                    raise InvalidClaimPayload(
                        "LLM claim payload supporting_evidence missing 'article_title'"
                    )

                raw_key_points = evidence.get("key_points")
                if isinstance(raw_key_points, (str, bytes)):
                    raise InvalidClaimPayload(
                        "LLM claim payload supporting_evidence.key_points must be an array of strings"
                    )
                try:
                    key_points_iter = iter(raw_key_points or [])
                except TypeError as exc:
                    raise InvalidClaimPayload(
                        "LLM claim payload supporting_evidence.key_points must be an array of strings"
                    ) from exc

                key_points = [point for point in key_points_iter if isinstance(point, str) and point]
                if not key_points:
                    raise InvalidClaimPayload(
                        "LLM claim payload supporting_evidence.key_points must contain at least one entry"
                    )

                if snippet_id not in aggregated_claim.evidence:
                    aggregated_claim.evidence[snippet_id] = _AggregatedEvidence(
                        snippet_id=snippet_id,
                        pmid=pmid,
                        article_title=article_title,
                        key_points=key_points,
                        notes=_clean_str(evidence.get("notes")),
                    )

            if "articles" not in claim or claim.get("articles") is None:
                raise InvalidClaimPayload("LLM claim payload missing required field 'articles'")
            linked_articles = _normalise_article_ids(claim.get("articles"))
            if not linked_articles:
                raise InvalidClaimPayload(
                    "LLM claim payload did not include any linked articles"
                )
            for article_id, pmid in linked_articles:
                if article_id not in aggregated_claim.evidence:
                    aggregated_claim.evidence[article_id] = _AggregatedEvidence(
                        snippet_id=article_id,
                        pmid=pmid,
                        article_title=None,
                        key_points=[],
                        notes=None,
                    )

        if declared_claim_ids:
            missing = sorted(declared_claim_ids)
            raise InvalidClaimPayload(
                "Drug metadata declares claims without payload entries: " + ", ".join(missing)
            )

    return _reduce_redundant_claims(aggregated)


def _parse_severe_reaction(value: dict) -> tuple[bool, list[str]]:
    if "flag" not in value:
        raise InvalidClaimPayload("LLM claim payload idiosyncratic_reaction missing 'flag'")
    flag_value = value["flag"]
    if not isinstance(flag_value, bool):
        raise InvalidClaimPayload("LLM claim payload idiosyncratic_reaction.flag must be boolean")

    if "descriptors" not in value:
        raise InvalidClaimPayload(
            "LLM claim payload idiosyncratic_reaction.descriptors missing"
        )
    descriptors = value["descriptors"]
    if isinstance(descriptors, (str, bytes)):
        raise InvalidClaimPayload(
            "LLM claim payload idiosyncratic_reaction.descriptors must be an array of strings"
        )
    try:
        iterator = iter(descriptors)
    except TypeError as exc:  # noqa: PERF203 - defensive validation
        raise InvalidClaimPayload(
            "LLM claim payload idiosyncratic_reaction.descriptors must be an array of strings"
        ) from exc

    cleaned_terms: list[str] = []
    for term in iterator:
        cleaned = _clean_str(term)
        if cleaned:
            cleaned_terms.append(cleaned)
    unique_terms = list(dict.fromkeys(cleaned_terms))
    return flag_value, unique_terms


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
        raise InvalidClaimPayload("LLM payload must be a JSON object")
    if isinstance(payload, dict):
        return payload
    parsed = getattr(payload, "parsed_json", None)
    if callable(parsed):
        try:
            result = parsed()
        except Exception as exc:  # noqa: BLE001 - wrap external client exceptions
            raise InvalidClaimPayload("LLM payload parsed_json() failed") from exc
        if isinstance(result, dict):
            return result
        raise InvalidClaimPayload("LLM payload parsed_json() must return a JSON object")
    raise InvalidClaimPayload("LLM payload must be a JSON object")


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
    *,
    drug_catalog: dict[str, _DrugCatalogEntry] | None = None,
) -> tuple[list[str], list[str], list[str]]:
    if isinstance(drugs, (str, bytes)):
        raise InvalidClaimPayload("LLM claim payload claim.drugs must be an array of canonical ids")
    try:
        raw_drugs = list(drugs or [])
    except TypeError as exc:
        raise InvalidClaimPayload(
            "LLM claim payload claim.drugs must be an array of canonical ids"
        ) from exc

    if isinstance(drug_classes, (str, bytes)):
        raise InvalidClaimPayload(
            "LLM claim payload claim.drug_classes must be an array of strings"
        )
    try:
        raw_classes = list(drug_classes or [])
    except TypeError as exc:
        raise InvalidClaimPayload(
            "LLM claim payload claim.drug_classes must be an array of strings"
        ) from exc

    class_terms: list[str] = []
    drug_terms: list[str] = []
    canonical_ids: list[str] = []

    for term in raw_drugs:
        cleaned = _clean_str(term)
        if not cleaned:
            raise InvalidClaimPayload(
                "LLM claim payload claim.drugs contains an empty identifier"
            )
        if not cleaned.startswith("drug:"):
            raise InvalidClaimPayload(
                "LLM claim payload claim.drugs must use canonical 'drug:' identifiers"
            )
        if drug_catalog is None:
            raise InvalidClaimPayload("LLM payload missing required field 'drugs'")
        entry = drug_catalog.get(cleaned)
        if entry is None:
            raise InvalidClaimPayload(f"Claim references unknown drug id '{cleaned}'")
        canonical_ids.append(entry.drug_id)
        if entry.name:
            drug_terms.append(entry.name)
        class_terms.extend(entry.classes)

    for term in raw_classes:
        cleaned = _clean_str(term)
        if cleaned:
            class_terms.append(cleaned)

    unique_canonical = list(dict.fromkeys(canonical_ids))
    return _unique_terms(drug_terms), _unique_terms(class_terms), unique_canonical


def _build_drug_catalog(entries: object) -> tuple[dict[str, _DrugCatalogEntry], dict[str, set[str]]]:
    catalog: dict[str, _DrugCatalogEntry] = {}
    claim_to_drugs: dict[str, set[str]] = {}
    if entries is None:
        raise InvalidClaimPayload("LLM payload missing required field 'drugs'")

    if isinstance(entries, (str, bytes)):
        raise InvalidClaimPayload("LLM payload drugs must be an array of objects")

    try:
        entries_list = list(entries)
    except TypeError as exc:
        raise InvalidClaimPayload("LLM payload drugs must be an array of objects") from exc

    if not entries_list:
        raise InvalidClaimPayload("Drugs array must contain at least one entry")

    for item in entries_list:
        if not isinstance(item, dict):
            raise InvalidClaimPayload("LLM payload drugs must contain objects")
        raw_id = _clean_str(item.get("id"))
        if not raw_id or not raw_id.startswith("drug:"):
            raise InvalidClaimPayload("LLM payload drugs entries must include canonical 'drug:' ids")

        name = _clean_str(item.get("name"))
        if not name:
            raise InvalidClaimPayload("LLM payload drugs entries must include non-empty 'name'")

        claims_field = item.get("claims")
        if claims_field is None:
            raise InvalidClaimPayload("LLM payload drugs entries must include 'claims' arrays")
        if isinstance(claims_field, (str, bytes)):
            raise InvalidClaimPayload("LLM payload drugs claims must be an array of claim ids")
        try:
            claim_iter = list(claims_field)
        except TypeError as exc:
            raise InvalidClaimPayload("LLM payload drugs claims must be an array of claim ids") from exc

        for claim_id in claim_iter:
            cleaned_claim = _clean_str(claim_id)
            if not cleaned_claim or not cleaned_claim.startswith("claim:"):
                raise InvalidClaimPayload(
                    "LLM payload drugs claims must contain canonical 'claim:' identifiers"
                )
            claim_to_drugs.setdefault(cleaned_claim, set()).add(raw_id)

        class_values: list[str] = []
        for class_term in item.get("classifications") or []:
            cleaned_class = _clean_str(class_term)
            if cleaned_class:
                class_values.append(cleaned_class)

        entry = _DrugCatalogEntry(drug_id=raw_id, name=name, classes=tuple(class_values))
        catalog[raw_id] = entry

    

    return catalog, claim_to_drugs


def _normalise_article_ids(values: object) -> list[tuple[str, str | None]]:
    if not values:
        return []

    if isinstance(values, (str, bytes)):
        raise InvalidClaimPayload("LLM claim payload articles must be an array of strings")

    try:
        iterator = iter(values)
    except TypeError as exc:
        raise InvalidClaimPayload("LLM claim payload articles must be an array of strings") from exc

    normalised: list[tuple[str, str | None]] = []
    for value in iterator:
        token = _clean_str(value)
        if not token:
            raise InvalidClaimPayload("LLM claim payload articles contains an empty identifier")
        if not token.startswith("article:"):
            raise InvalidClaimPayload("LLM claim payload articles must use 'article:' prefix")
        suffix = token[len("article:") :]
        if not suffix or not suffix.isdigit():
            raise InvalidClaimPayload(
                "LLM claim payload articles must use 'article:' followed by numeric PMID"
            )
        normalised.append((token, suffix))

    seen: set[str] = set()
    deduped: list[tuple[str, str | None]] = []
    for article_id, pmid in normalised:
        if article_id in seen:
            continue
        seen.add(article_id)
        deduped.append((article_id, pmid))
    return deduped


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
