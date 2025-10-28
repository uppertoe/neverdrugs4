from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Iterable, Literal, Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import ArticleArtefact, ArticleSnippet
from app.services.drug_classes import resolve_drug_group
from app.services.query_terms import DRUG_TEXT_TERMS

_NEGATING_RISK_PATTERNS: tuple[str, ...] = (
    "no {}",
    "not {}",
    "without {}",
    "absence of {}",
    "does not cause {}",
    "did not cause {}",
    "doesn't cause {}",
    "didn't cause {}",
    "not associated with {}",
    "no evidence of {}",
)

DEFAULT_RISK_CUES: tuple[str, ...] = (
    "adverse",
    "adverse event",
    "adverse events",
    "arrhythmia",
    "arrhythmias",
    "avoid",
    "avoided",
    "cardiac arrest",
    "complication",
    "complications",
    "contraindicated",
    "contraindication",
    "contraindications",
    "caution",
    "danger",
    "deterioration",
    "do not use",
    "exacerbate",
    "exacerbated",
    "fatal",
    "harmful",
    "hazard",
    "hyperkalemia",
    "hyperkalaemia",
    "life-threatening",
    "malignant hyperthermia",
    "mh",
    "precaution",
    "precipitate",
    "risk",
    "risk of",
    "rhabdomyolysis",
    "serious adverse",
    "should not",
    "side effect",
    "side effects",
    "toxic",
    "toxicity",
    "trigger",
    "triggered",
    "triggering",
    "unsafe",
    "worsened",
)
DEFAULT_SAFETY_CUES: tuple[str, ...] = (
    "acceptable safety",
    "beneficial",
    "did not cause",
    "effective",
    "generally safe",
    "no adverse event",
    "no adverse events",
    "no complications",
    "no major complications",
    "no reported complications",
    "no significant adverse",
    "recommended",
    "safe",
    "safe option",
    "safely",
    "safety",
    "successfully",
    "tolerated well",
    "well tolerated",
    "well-tolerated",
    "without complications",
    "ameliorated",
    "ameliorate",
    "prevent",
    "preventative",
    "indicated",
    "efficacious",
    "efficacy",
)
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


@dataclass(slots=True)
class _SnippetWindow:
    text: str
    left: int
    right: int


class ArticleSnippetExtractor:
    def __init__(
        self,
        *,
        drug_terms: Iterable[str] = DRUG_TEXT_TERMS,
        risk_cues: Sequence[str] = DEFAULT_RISK_CUES,
        safety_cues: Sequence[str] = DEFAULT_SAFETY_CUES,
        window_chars: int = DEFAULT_WINDOW_CHARS,
    ) -> None:
        self.drug_terms = tuple(sorted({term.lower() for term in drug_terms if term}))
        self.risk_cues = tuple(cue.lower() for cue in risk_cues)
        self.safety_cues = tuple(cue.lower() for cue in safety_cues)
        self.window_chars = max(100, window_chars)
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
    ) -> list[SnippetCandidate]:
        if not article_text:
            return []

        normalized_text = _normalize_whitespace(article_text)
        lower_text = normalized_text.lower()
        condition_aliases = [term.lower() for term in condition_terms if term]
        if not condition_aliases:
            return []

        # Only enforce condition matching when the article actually references one of the aliases.
        require_condition = any(alias in lower_text for alias in condition_aliases)

        seen_keys: set[tuple[str, str]] = set()
        occupied_windows: list[dict[str, object]] = []
        candidates: list[SnippetCandidate] = []

        for drug in self.drug_terms:
            pattern = re.compile(rf"\b{re.escape(drug)}\b")
            drug_group = resolve_drug_group(drug)
            for match in pattern.finditer(lower_text):
                window = self._build_window(normalized_text, match.start(), match.end())
                snippet = window.text
                left_bound = window.left
                right_bound = window.right
                if len(snippet) < _MIN_SNIPPET_CHARS:
                    continue
                snippet_lower = snippet.lower()
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

                key = (drug, snippet_lower)
                overlapping = _find_overlapping_window(occupied_windows, left_bound, right_bound)
                if overlapping is not None and overlapping.get("drug") != drug:
                    overlapping = None
                if overlapping is None and key in seen_keys:
                    continue

                snippet_score = self._score_snippet(
                    article_score=article_score,
                    pmc_ref_count=pmc_ref_count,
                    classification=classification,
                    cue_count=len(cues),
                    condition_match=(
                        snippet_matches_condition or not require_condition or inferred_condition
                    ),
                )

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
                )

                if overlapping is not None:
                    if snippet_score <= overlapping["score"]:
                        continue
                    old_key = overlapping["key"]
                    if isinstance(old_key, tuple):
                        seen_keys.discard(old_key)
                    index = int(overlapping["index"])
                    candidates[index] = candidate_obj
                    overlapping.update(
                        {
                            "left": left_bound,
                            "right": right_bound,
                            "score": snippet_score,
                            "key": key,
                            "drug": drug,
                        }
                    )
                    seen_keys.add(key)
                    continue

                seen_keys.add(key)
                index = len(candidates)
                candidates.append(candidate_obj)
                occupied_windows.append(
                    {
                        "left": left_bound,
                        "right": right_bound,
                        "score": snippet_score,
                        "index": index,
                        "key": key,
                        "drug": drug,
                    }
                )

        candidates.sort(key=lambda item: (item.article_rank, -item.snippet_score))
        return candidates

    def _build_window(self, text: str, start: int, end: int) -> _SnippetWindow:
        left = max(0, start - self.window_chars)
        right = min(len(text), end + self.window_chars)
        snippet = text[left:right]
        snippet = snippet.strip()

        # Attempt to trim to sentence boundaries for readability
        period_left = snippet.find(". ")
        if period_left != -1 and period_left < self.window_chars // 2:
            snippet = snippet[period_left + 2 :]
        period_right = snippet.rfind(". ")
        if period_right != -1 and len(snippet) - period_right > self.window_chars // 2:
            snippet = snippet[: period_right + 1]

        snippet = snippet.strip()
        adjusted_left = text.find(snippet, left, right) if snippet else left
        if adjusted_left == -1:
            adjusted_left = left
        adjusted_right = adjusted_left + len(snippet)
        return _SnippetWindow(text=snippet, left=adjusted_left, right=adjusted_right)

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

    def _score_snippet(
        self,
        *,
        article_score: float,
        pmc_ref_count: int,
        classification: Literal["risk", "safety"],
        cue_count: int,
        condition_match: bool,
    ) -> float:
        score = article_score
        score += min(pmc_ref_count / 40.0, 2.0)
        score += 0.5 if classification == "risk" else 0.3
        score += 0.1 * cue_count
        score += 0.4 if condition_match else -0.2
        return round(score, 4)


def select_top_snippets(
    candidates: Sequence[SnippetCandidate],
    *,
    base_quota: int = 3,
    max_quota: int = 8,
) -> list[SnippetCandidate]:
    if not candidates:
        return []

    grouped: dict[str, list[SnippetCandidate]] = {}
    for candidate in candidates:
        grouped.setdefault(candidate.pmid, []).append(candidate)

    selected: list[SnippetCandidate] = []
    for pmid, items in grouped.items():
        items_sorted = sorted(items, key=lambda s: s.snippet_score, reverse=True)
        quota = _compute_quota(items_sorted[0], base_quota=base_quota, max_quota=max_quota)
        selected.extend(items_sorted[:quota])

    # Order by article rank, then snippet score descending
    selected.sort(key=lambda s: (s.article_rank, -s.snippet_score))
    return selected


def _compute_quota(
    candidate: SnippetCandidate,
    *,
    base_quota: int,
    max_quota: int,
) -> int:
    quota = base_quota
    if candidate.pmc_ref_count >= 10:
        quota += 1
    if candidate.pmc_ref_count >= 30:
        quota += 1
    if candidate.article_score >= 4.0:
        quota += 1
    return min(max_quota, quota)


def _normalize_whitespace(value: str) -> str:
    return " ".join(value.split())


def _ranges_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return not (a_end <= b_start or b_end <= a_start)


def _find_overlapping_window(
    windows: Sequence[dict[str, object]],
    left: int,
    right: int,
) -> dict[str, object] | None:
    for window in windows:
        existing_left = int(window.get("left", 0))
        existing_right = int(window.get("right", 0))
        if _ranges_overlap(left, right, existing_left, existing_right):
            return window
    return None


def _is_negated_risk_phrase(snippet_lower: str, cue: str) -> bool:
    for pattern in _NEGATING_RISK_PATTERNS:
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

        snippet_hash = _compute_snippet_hash(candidate.snippet_text)
        stmt = select(ArticleSnippet).where(
            ArticleSnippet.article_artefact_id == artefact.id,
            ArticleSnippet.snippet_hash == snippet_hash,
        )
        existing = session.execute(stmt).scalar_one_or_none()

        if existing is None:
            snippet = ArticleSnippet(
                article_artefact_id=artefact.id,
                snippet_hash=snippet_hash,
                drug=candidate.drug,
                classification=candidate.classification,
                snippet_text=candidate.snippet_text,
                snippet_score=candidate.snippet_score,
                cues=list(candidate.cues),
            )
            session.add(snippet)
            persisted.append(snippet)
            continue

        existing.drug = candidate.drug
        existing.classification = candidate.classification
        existing.snippet_text = candidate.snippet_text
        existing.snippet_score = candidate.snippet_score
        existing.cues = list(candidate.cues)
        persisted.append(existing)

    session.flush()
    return persisted


def _compute_snippet_hash(snippet_text: str) -> str:
    normalized = _normalize_whitespace(snippet_text).lower().encode("utf-8")
    return hashlib.sha1(normalized).hexdigest()
