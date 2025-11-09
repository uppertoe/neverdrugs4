from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime
from dataclasses import asdict, dataclass, field
from typing import Iterable, Literal, Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import ArticleArtefact, ArticleSnippet
from app.services.drug_classes import resolve_drug_group
from app.services.query_terms import DRUG_TEXT_TERMS
from app.services.snippet_candidates import (
    RegexSnippetCandidateFinder,
    SnippetCandidateFinder,
    SnippetSpan,
)
from app.services.snippet_pruning import WindowedCandidate, prune_window_overlaps
from app.services.snippet_scoring import (
    SnippetScoringConfig,
    score_snippet_with_config,
)
from app.services.snippet_tags import DEFAULT_RISK_CUES, DEFAULT_SAFETY_CUES, NEGATING_RISK_PATTERNS, Tag
from app.services.snippet_tagger import RuleBasedSnippetTagger, SnippetTagger
THERAPY_ROLE_PATTERNS: dict[str, dict[str, tuple[str, ...]]] = {
    "mh-therapy": {
        "condition_terms": ("malignant hyperthermia", "mh"),
        "keywords": (
            "treat",
            "treated",
            "treating",
            "treatment",
            "therapy",
            "therapeutic",
            "manage",
            "managed",
            "managing",
            "management",
            "administer",
            "administered",
            "administering",
            "administration",
            "give",
            "given",
            "giving",
            "dose",
            "dosing",
            "bolus",
            "reversal",
            "reverse",
            "reverses",
            "reversed",
            "responded",
            "response",
            "mitigates",
            "mitigated",
            "mitigate",
            "ameliorates",
            "ameliorated",
            "ameliorate",
            "rescue",
            "only effective",
            "first-line",
            "should be available",
            "required",
            "requires",
            "requirement",
            "must have",
            "availability",
            "prompt",
            "immediate",
            "loading",
            "infusion",
            "stocked",
        ),
        "exclusions": (
            "contraindicated",
            "contraindication",
            "should not",
            "do not use",
            "avoid",
            "toxicity",
            "toxic",
            "hepatotoxic",
            "hepatotoxicity",
            "adverse event",
            "adverse events",
            "serious adverse",
            "black box",
            "risk of hepatotoxicity",
        ),
    }
}
DEFAULT_WINDOW_CHARS = 600
_MIN_SNIPPET_CHARS = 60


def _parse_publication_year(raw: str | None) -> int | None:
    if not raw:
        return None
    match = re.search(r"(19|20|21)\d{2}", raw)
    if match is None:
        return None
    year = int(match.group(0))
    current_year = datetime.now(UTC).year + 1
    if 1800 <= year <= current_year:
        return year
    return None


def _normalize_study_types(values: Sequence[str] | None) -> tuple[str, ...]:
    if not values:
        return ()
    cleaned = [value.strip() for value in values if value]
    if not cleaned:
        return ()
    ordered = dict.fromkeys(cleaned)
    return tuple(ordered.keys())


@dataclass(slots=True)
class SnippetCandidate:
    pmid: str
    drug: str
    classification: Literal["risk", "safety"]
    snippet_text: str
    article_rank: int
    article_score: float
    preferred_url: str
    pmc_ref_count: int
    snippet_score: float
    cues: list[str]
    tags: list[Tag] = field(default_factory=list)
    publication_year: int | None = None
    study_types: tuple[str, ...] = field(default_factory=tuple)
    cohort_size: int | None = None


@dataclass(slots=True)
class SnippetResult:
    candidate: SnippetCandidate
    span: SnippetSpan
    metadata: dict[str, object] = field(default_factory=dict)


class ArticleSnippetExtractor:
    def __init__(
        self,
        *,
        drug_terms: Iterable[str] = DRUG_TEXT_TERMS,
        risk_cues: Sequence[str] = DEFAULT_RISK_CUES,
        safety_cues: Sequence[str] = DEFAULT_SAFETY_CUES,
        window_chars: int = DEFAULT_WINDOW_CHARS,
        min_snippet_chars: int = _MIN_SNIPPET_CHARS,
        tagger: SnippetTagger | None = None,
        candidate_finder: SnippetCandidateFinder | None = None,
        scoring_config: SnippetScoringConfig | None = None,
    ) -> None:
        self.drug_terms = tuple(sorted({term.lower() for term in drug_terms if term}))
        self.risk_cues = tuple(cue.lower() for cue in risk_cues)
        self.safety_cues = tuple(cue.lower() for cue in safety_cues)
        self.window_chars = max(100, window_chars)
        self.candidate_finder = candidate_finder or RegexSnippetCandidateFinder(
            drug_terms=self.drug_terms,
            window_chars=self.window_chars,
        )
        self.tagger = tagger or RuleBasedSnippetTagger(
            risk_cues=self.risk_cues,
            safety_cues=self.safety_cues,
        )
        self.min_snippet_chars = max(30, min_snippet_chars)
        self.scoring_config = scoring_config
        self.therapy_role_patterns = {
            role: {
                "condition_terms": tuple(
                    term.lower() for term in config.get("condition_terms", ())
                ),
                "keywords": tuple(keyword.lower() for keyword in config.get("keywords", ())),
                "exclusions": tuple(
                    exclusion.lower() for exclusion in config.get("exclusions", ())
                ),
            }
            for role, config in THERAPY_ROLE_PATTERNS.items()
        }

    def extract_snippets(
        self,
        *,
        article_text: str,
        pmid: str,
        condition_terms: Sequence[str],
        article_rank: int,
        article_score: float,
        preferred_url: str,
        pmc_ref_count: int,
        publication_date: str | None = None,
        publication_types: Sequence[str] | None = None,
        cohort_size: int | None = None,
    ) -> list[SnippetCandidate]:
        results = self.extract_snippet_results(
            article_text=article_text,
            pmid=pmid,
            condition_terms=condition_terms,
            article_rank=article_rank,
            article_score=article_score,
            preferred_url=preferred_url,
            pmc_ref_count=pmc_ref_count,
            publication_date=publication_date,
            publication_types=publication_types,
            cohort_size=cohort_size,
        )
        return [result.candidate for result in results]

    def extract_snippet_results(
        self,
        *,
        article_text: str,
        pmid: str,
        condition_terms: Sequence[str],
        article_rank: int,
        article_score: float,
        preferred_url: str,
        pmc_ref_count: int,
        publication_date: str | None = None,
        publication_types: Sequence[str] | None = None,
        cohort_size: int | None = None,
    ) -> list[SnippetResult]:
        if not article_text:
            return []

        normalized_text = _normalize_whitespace(article_text)
        lower_text = normalized_text.lower()
        condition_aliases = [term.lower() for term in condition_terms if term]
        if not condition_aliases:
            return []

        # Only enforce condition matching when the article actually references one of the aliases.
        require_condition = any(alias in lower_text for alias in condition_aliases)
        publication_year = _parse_publication_year(publication_date)
        normalized_study_types = _normalize_study_types(publication_types)
        cohort_size_value = cohort_size if isinstance(cohort_size, int) and cohort_size > 0 else None

        windowed_candidates: list[WindowedCandidate] = []

        for drug in self.drug_terms:
            drug_group = resolve_drug_group(drug)
            for span in self.candidate_finder.find_candidates(
                article_text=normalized_text,
                normalized_text=normalized_text,
                lower_text=lower_text,
                drug=drug,
            ):
                snippet = span.text
                if len(snippet) < self.min_snippet_chars:
                    continue
                snippet_lower = snippet.lower()
                tags = self.tagger.tag_snippet(
                    snippet,
                    drug=drug,
                    condition_terms=condition_terms,
                )
                snippet_matches_condition = any(
                    alias in snippet_lower for alias in condition_aliases
                )
                classification, cues = self._classify(
                    snippet_lower, drug, drug_group.roles
                )
                inferred_condition = False
                if classification is None:
                    if snippet_matches_condition:
                        classification = "risk"
                        cues = ("condition-match",)
                        inferred_condition = True
                    else:
                        continue

                severe_tags = [tag.label for tag in tags if tag.kind == "severe_reaction"]
                therapy_roles = [tag.label for tag in tags if tag.kind == "therapy_role"]
                mechanism_alerts = [tag.label for tag in tags if tag.kind == "mechanism_alert"]

                snippet_score = score_snippet_with_config(
                    article_score=article_score,
                    pmc_ref_count=pmc_ref_count,
                    classification=classification,
                    cue_count=len(cues),
                    condition_match=(
                        snippet_matches_condition or not require_condition or inferred_condition
                    ),
                    config=self.scoring_config,
                    study_types=normalized_study_types,
                    publication_year=publication_year,
                    cohort_size=cohort_size_value,
                )

                metadata = {
                    "condition_matched": snippet_matches_condition,
                    "condition_inferred": inferred_condition,
                    "require_condition": require_condition,
                    "drug_roles": drug_group.roles,
                }
                if publication_year is not None:
                    metadata["publication_year"] = publication_year
                if normalized_study_types:
                    metadata["study_types"] = normalized_study_types
                if cohort_size_value is not None:
                    metadata["cohort_size"] = cohort_size_value
                if severe_tags:
                    metadata["severe_reaction_flag"] = True
                    metadata["severe_reaction_terms"] = tuple(sorted({tag for tag in severe_tags}))
                if therapy_roles:
                    metadata["therapy_roles"] = tuple(sorted({role for role in therapy_roles}))
                if mechanism_alerts:
                    metadata["mechanism_alerts"] = tuple(sorted({alert for alert in mechanism_alerts}))

                candidate_obj = SnippetCandidate(
                    pmid=pmid,
                    drug=drug,
                    classification=classification,
                    snippet_text=snippet,
                    article_rank=article_rank,
                    article_score=article_score,
                    preferred_url=preferred_url,
                    pmc_ref_count=pmc_ref_count,
                    snippet_score=snippet_score,
                    cues=list(cues),
                    tags=list(tags),
                    publication_year=publication_year,
                    study_types=normalized_study_types,
                    cohort_size=cohort_size_value,
                )
                key = (drug, snippet_lower)
                windowed_candidates.append(
                    WindowedCandidate(
                        candidate=candidate_obj,
                        span=span,
                        key=key,
                        metadata=metadata,
                    )
                )

        pruned = prune_window_overlaps(windowed_candidates)
        results = [
            SnippetResult(
                candidate=entry.candidate,
                span=entry.span,
                metadata=dict(entry.metadata or {}),
            )
            for entry in pruned
        ]
        results.sort(
            key=lambda item: (
                item.candidate.article_rank,
                -item.candidate.snippet_score,
            )
        )
        return results

    def _classify(
        self, snippet_lower: str, drug: str, drug_roles: tuple[str, ...]
    ) -> tuple[str | None, tuple[str, ...]]:
        risk_hits = tuple(
            cue
            for cue in self.risk_cues
            if cue in snippet_lower and not _is_negated_risk_phrase(snippet_lower, cue)
        )
        safety_hits = tuple(cue for cue in self.safety_cues if cue in snippet_lower)

        alt_phrase = f"alternative to {drug}"
        if alt_phrase in snippet_lower and alt_phrase not in risk_hits:
            risk_hits = risk_hits + (alt_phrase,)

        if not risk_hits and not safety_hits:
            base_classification: str | None = None
            cues: tuple[str, ...] = ()
        elif risk_hits and not safety_hits:
            base_classification = "risk"
            cues = risk_hits
        elif safety_hits and not risk_hits:
            base_classification = "safety"
            cues = safety_hits
        else:
            base_classification = "risk"
            cues = risk_hits + safety_hits

        override = self._apply_therapy_override(
            snippet_lower=snippet_lower,
            drug=drug,
            drug_roles=drug_roles,
            base_classification=base_classification,
            safety_hits=safety_hits,
        )
        if override is not None:
            return override

        return base_classification, cues

    def _apply_therapy_override(
        self,
        *,
        snippet_lower: str,
        drug: str,
        drug_roles: tuple[str, ...],
        base_classification: str | None,
        safety_hits: tuple[str, ...],
    ) -> tuple[str, tuple[str, ...]] | None:
        for role in drug_roles:
            config = self.therapy_role_patterns.get(role)
            if not config:
                continue

            if config.get("condition_terms") and not any(
                term in snippet_lower for term in config["condition_terms"]
            ):
                continue

            if config.get("exclusions") and any(
                exclusion in snippet_lower for exclusion in config["exclusions"]
            ):
                continue

            if not self._matches_therapy_role(snippet_lower, drug, config.get("keywords", ())):
                continue

            cue_label = f"therapy-role:{role}"
            cues = tuple(dict.fromkeys((*safety_hits, cue_label)))
            if not cues:
                cues = (cue_label,)
            target_classification = "safety" if base_classification != "safety" else "safety"
            return target_classification, cues

        return None

    @staticmethod
    def _matches_therapy_role(
        snippet_lower: str,
        drug: str,
        keywords: tuple[str, ...],
        *,
        radius: int = 80,
    ) -> bool:
        if not keywords:
            return False

        start = 0
        while True:
            idx = snippet_lower.find(drug, start)
            if idx == -1:
                break
            window_start = max(0, idx - radius)
            window_end = idx + len(drug) + radius
            window = snippet_lower[window_start:window_end]
            if any(keyword in window for keyword in keywords):
                return True
            start = idx + len(drug)
        return False

def _normalize_whitespace(value: str) -> str:
    return " ".join(value.split())


def _is_negated_risk_phrase(snippet_lower: str, cue: str) -> bool:
    for pattern in NEGATING_RISK_PATTERNS:
        formatted = pattern.format(cue)
        if formatted in snippet_lower:
            return True
    return False


def persist_snippet_candidates(
    session: Session,
    *,
    article_artefacts: Sequence[ArticleArtefact],
    snippet_candidates: Sequence[SnippetCandidate],
) -> list[ArticleSnippet]:
    if not snippet_candidates:
        return []

    artefact_by_pmid = {artefact.pmid: artefact for artefact in article_artefacts}
    persisted: list[ArticleSnippet] = []

    for candidate in snippet_candidates:
        artefact = artefact_by_pmid.get(candidate.pmid)
        if artefact is None:
            continue

        record = _serialize_candidate(candidate)
        snippet_hash = record["snippet_hash"]
        stmt = select(ArticleSnippet).where(
            ArticleSnippet.article_artefact_id == artefact.id,
            ArticleSnippet.snippet_hash == snippet_hash,
        )
        existing = session.execute(stmt).scalar_one_or_none()

        if existing is None:
            snippet = ArticleSnippet(
                article_artefact_id=artefact.id,
                **record,
            )
            session.add(snippet)
            persisted.append(snippet)
            continue

        existing.drug = record["drug"]
        existing.classification = record["classification"]
        existing.snippet_text = record["snippet_text"]
        existing.snippet_score = record["snippet_score"]
        existing.cues = record["cues"]
        existing.tags = record["tags"]
        persisted.append(existing)

    session.flush()
    return persisted


def _compute_snippet_hash(snippet_text: str) -> str:
    normalized = _normalize_whitespace(snippet_text).lower().encode("utf-8")
    return hashlib.sha1(normalized).hexdigest()


def _serialize_tags(tags: Sequence[Tag] | None) -> list[dict[str, object]]:
    if not tags:
        return []

    serialized: list[dict[str, object]] = []
    for tag in tags:
        if isinstance(tag, Tag):
            serialized.append(asdict(tag))
        elif isinstance(tag, dict):
            serialized.append(dict(tag))
        else:
            msg = f"Cannot serialize tag of type {type(tag)!r}"
            raise TypeError(msg)
    return serialized


def _serialize_candidate(candidate: SnippetCandidate) -> dict[str, object]:
    return {
        "snippet_hash": _compute_snippet_hash(candidate.snippet_text),
        "drug": candidate.drug,
        "classification": candidate.classification,
        "snippet_text": candidate.snippet_text,
        "snippet_score": candidate.snippet_score,
        "cues": list(candidate.cues),
        "tags": _serialize_tags(candidate.tags),
    }
