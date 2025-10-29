from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence, TYPE_CHECKING

from app.services.snippet_candidates import SnippetSpan
from app.services.snippet_scoring import (
    SnippetQuotaConfig,
    compute_quota_with_config,
)

if TYPE_CHECKING:
    from app.services.snippets import SnippetCandidate


@dataclass(slots=True)
class WindowedCandidate:
    candidate: "SnippetCandidate"
    span: SnippetSpan
    key: tuple[str, str]
    metadata: dict[str, object] | None = None


def prune_window_overlaps(
    candidates: Sequence[WindowedCandidate],
) -> list[WindowedCandidate]:
    if not candidates:
        return []

    windows: list[dict[str, object]] = []
    seen_keys: set[tuple[str, str]] = set()
    pruned: list[WindowedCandidate] = []

    for entry in candidates:
        left_bound = entry.span.left
        right_bound = entry.span.right
        overlapping = _find_overlapping_window(windows, left_bound, right_bound)
        if overlapping is not None and overlapping.get("drug") != entry.candidate.drug:
            overlapping = None

        key = entry.key
        if overlapping is None and key in seen_keys:
            continue

        score = entry.candidate.snippet_score
        if overlapping is not None:
            if score <= overlapping["score"]:
                continue
            old_key = overlapping["key"]
            if isinstance(old_key, tuple):
                seen_keys.discard(old_key)
            index = int(overlapping["index"])
            pruned[index] = entry
            overlapping.update(
                {
                    "left": left_bound,
                    "right": right_bound,
                    "score": score,
                    "key": key,
                    "drug": entry.candidate.drug,
                }
            )
            seen_keys.add(key)
            continue

        seen_keys.add(key)
        index = len(pruned)
        pruned.append(entry)
        windows.append(
            {
                "left": left_bound,
                "right": right_bound,
                "score": score,
                "index": index,
                "key": key,
                "drug": entry.candidate.drug,
            }
        )

    return pruned


def apply_article_quotas(
    candidates: Sequence["SnippetCandidate"],
    *,
    base_quota: int = 3,
    max_quota: int = 8,
    quota_config: SnippetQuotaConfig | None = None,
) -> list["SnippetCandidate"]:
    if not candidates:
        return []

    grouped: dict[str, list["SnippetCandidate"]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate.pmid].append(candidate)

    selected: list["SnippetCandidate"] = []
    for pmid, items in grouped.items():
        items_sorted = sorted(items, key=lambda item: item.snippet_score, reverse=True)
        quota = compute_quota_with_config(
            items_sorted[0],
            base_quota=base_quota,
            max_quota=max_quota,
            config=quota_config,
        )
        selected.extend(items_sorted[:quota])

    selected.sort(key=lambda item: (item.article_rank, -item.snippet_score))
    return selected


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


def _ranges_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return not (a_end <= b_start or b_end <= a_start)


__all__ = [
    "WindowedCandidate",
    "apply_article_quotas",
    "prune_window_overlaps",
]
