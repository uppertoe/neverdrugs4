from __future__ import annotations

from typing import Iterable, Protocol, Sequence, runtime_checkable

from app.services.snippet_tags import (
    DEFAULT_RISK_CUES,
    DEFAULT_SAFETY_CUES,
    IGNORED_SEVERE_REACTION_CUES,
    MECHANISM_ALERT_CUES,
    SEVERE_REACTION_ALWAYS_CUES,
    SEVERE_REACTION_CONDITIONAL_CUES,
    SEVERE_REACTION_QUALIFIERS,
    THERAPY_ROLE_CUES,
    Tag,
)


@runtime_checkable
class SnippetTagger(Protocol):
    def tag_snippet(
        self,
        snippet_text: str,
        *,
        drug: str,
        condition_terms: Sequence[str],
    ) -> list[Tag]:
        """Return structured tags for the supplied snippet."""


class RuleBasedSnippetTagger:
    def __init__(
        self,
        *,
        risk_cues: Iterable[str] | None = None,
        safety_cues: Iterable[str] | None = None,
        severe_reaction_always_cues: Iterable[str] | None = None,
        severe_reaction_conditional_cues: Iterable[str] | None = None,
        severe_reaction_qualifiers: Iterable[str] | None = None,
        ignored_severe_reaction_cues: Iterable[str] | None = None,
        therapy_role_cues: dict[str, Iterable[str]] | None = None,
        mechanism_alert_cues: dict[str, Iterable[str]] | None = None,
    ) -> None:
        self._risk_cues = tuple(cue.lower() for cue in (risk_cues or DEFAULT_RISK_CUES))
        self._safety_cues = tuple(cue.lower() for cue in (safety_cues or DEFAULT_SAFETY_CUES))
        self._severe_reaction_always = tuple(
            cue.lower() for cue in (severe_reaction_always_cues or SEVERE_REACTION_ALWAYS_CUES)
        )
        self._severe_reaction_conditional = tuple(
            cue.lower()
            for cue in (severe_reaction_conditional_cues or SEVERE_REACTION_CONDITIONAL_CUES)
        )
        self._severe_reaction_qualifiers = tuple(
            qualifier.lower()
            for qualifier in (severe_reaction_qualifiers or SEVERE_REACTION_QUALIFIERS)
        )
        self._ignored_severe_reaction_cues = tuple(
            cue.lower()
            for cue in (ignored_severe_reaction_cues or IGNORED_SEVERE_REACTION_CUES)
        )
        role_cues = therapy_role_cues or THERAPY_ROLE_CUES
        self._therapy_role_cues = {
            role: tuple(cue.lower() for cue in cues)
            for role, cues in role_cues.items()
        }
        mechanism_cues = mechanism_alert_cues or MECHANISM_ALERT_CUES
        self._mechanism_alert_cues = {
            label: tuple(cue.lower() for cue in cues)
            for label, cues in mechanism_cues.items()
        }

    def tag_snippet(
        self,
        snippet_text: str,
        *,
        drug: str,
        condition_terms: Sequence[str],
    ) -> list[Tag]:
        del drug
        del condition_terms
        normalized = snippet_text.lower()
        tags: list[Tag] = []

        for cue in self._risk_cues:
            if cue and cue in normalized:
                tags.append(Tag(kind="risk", label=cue, confidence=1.0, source="rule"))

        for cue in self._safety_cues:
            if cue and cue in normalized:
                tags.append(Tag(kind="safety", label=cue, confidence=1.0, source="rule"))

        def _append_severe(label: str) -> None:
            if not any(tag.kind == "severe_reaction" and tag.label == label for tag in tags):
                tags.append(Tag(kind="severe_reaction", label=label, confidence=1.0, source="rule"))

        for cue in self._severe_reaction_always:
            if cue and cue not in self._ignored_severe_reaction_cues and cue in normalized:
                _append_severe(cue)

        qualifiers_present = any(qualifier in normalized for qualifier in self._severe_reaction_qualifiers)
        if qualifiers_present:
            for cue in self._severe_reaction_conditional:
                if cue and cue not in self._ignored_severe_reaction_cues and cue in normalized:
                    _append_severe(cue)

        for role, cues in self._therapy_role_cues.items():
            if any(cue and cue in normalized for cue in cues):
                if not any(tag.kind == "therapy_role" and tag.label == role for tag in tags):
                    tags.append(Tag(kind="therapy_role", label=role, confidence=0.9, source="rule"))

        for label, cues in self._mechanism_alert_cues.items():
            if any(cue and cue in normalized for cue in cues):
                if not any(tag.kind == "mechanism_alert" and tag.label == label for tag in tags):
                    tags.append(Tag(kind="mechanism_alert", label=label, confidence=0.9, source="rule"))

        return tags


__all__ = ["SnippetTagger", "RuleBasedSnippetTagger"]
