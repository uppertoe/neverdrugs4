from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Generic, List, Protocol, Sequence, Tuple, TypeVar, cast

from app.services.drug_classes import DrugGroup, resolve_drug_group


class SnippetLike(Protocol):
    pmid: str
    snippet_id: int | None
    drug: str
    classification: str
    snippet_text: str
    snippet_score: float


SnippetT = TypeVar("SnippetT", bound=SnippetLike)


@dataclass(slots=True)
class ClaimEvidenceGroup(Generic[SnippetT]):
    group_key: str
    classification: str
    drug_label: str
    drug_terms: Tuple[str, ...]
    drug_classes: Tuple[str, ...]
    snippets: List[SnippetT]
    top_score: float

    @property
    def snippet_ids(self) -> Tuple[int | None, ...]:
        return tuple(snippet.snippet_id for snippet in self.snippets)

    @property
    def pmids(self) -> Tuple[str, ...]:
        return tuple(sorted({snippet.pmid for snippet in self.snippets}))


def group_snippets_for_claims(
    snippets: Sequence[SnippetT],
    *,
    normalise_drug: Callable[[str], DrugGroup] = resolve_drug_group,
) -> List[ClaimEvidenceGroup]:
    if not snippets:
        return []

    buckets: Dict[Tuple[str, str], Dict[str, object]] = {}

    for snippet in snippets:
        drug_name = getattr(snippet, "drug", "").strip()
        classification = getattr(snippet, "classification", "").strip().lower()
        if not drug_name or classification not in {"risk", "safety"}:
            # Skip snippets that do not provide core metadata
            continue

        drug_group = normalise_drug(drug_name)
        key = (classification, drug_group.key)

        if key not in buckets:
            buckets[key] = {
                "classification": classification,
                "drug_label": drug_group.label,
                "drug_terms": {drug_name},
                "drug_classes": set(drug_group.classes),
                "drug_roles": set(drug_group.roles),
                "snippets": [snippet],
                "top_score": getattr(snippet, "snippet_score", 0.0),
            }
            continue

        bucket = buckets[key]
        bucket["drug_terms"].add(drug_name)
        bucket["drug_classes"].update(drug_group.classes)
        bucket["drug_roles"].update(drug_group.roles)
        bucket["snippets"].append(snippet)
        score = getattr(snippet, "snippet_score", 0.0)
        if score > bucket["top_score"]:
            bucket["top_score"] = score

    groups: List[ClaimEvidenceGroup] = []
    for (classification, group_key), payload in buckets.items():
        snippets_list = cast(List[SnippetT], payload["snippets"])
        snippets_list.sort(key=lambda item: getattr(item, "snippet_score", 0.0), reverse=True)
        drug_terms = tuple(sorted(cast(set[str], payload["drug_terms"]), key=lambda term: term.lower()))
        class_set = cast(set[str], payload["drug_classes"])
        roles_set = cast(set[str], payload["drug_roles"])
        drug_classes = tuple(sorted(class_set | roles_set))
        groups.append(
            ClaimEvidenceGroup(
                group_key=f"{classification}:{group_key}",
                classification=classification,
                drug_label=payload["drug_label"],
                drug_terms=drug_terms,
                drug_classes=drug_classes,
                snippets=snippets_list,
                top_score=payload["top_score"],
            )
        )

    groups.sort(key=lambda group: (-group.top_score, group.classification, group.drug_label.lower()))
    return groups
