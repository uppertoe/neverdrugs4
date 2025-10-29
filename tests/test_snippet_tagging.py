from __future__ import annotations

from app.services.snippet_tags import Tag
from app.services.snippet_tagger import RuleBasedSnippetTagger


def test_rule_based_tagger_detects_risk_cue() -> None:
    tagger = RuleBasedSnippetTagger()
    snippet = "Administration of succinylcholine resulted in a serious adverse event with arrhythmia."

    tags = tagger.tag_snippet(
        snippet,
        drug="succinylcholine",
        condition_terms=("malignant hyperthermia",),
    )

    assert Tag(kind="risk", label="adverse event", confidence=1.0, source="rule") in tags


def test_rule_based_tagger_detects_safety_cue() -> None:
    tagger = RuleBasedSnippetTagger()
    snippet = "The propofol infusion was well tolerated without complications for these patients."

    tags = tagger.tag_snippet(
        snippet,
        drug="propofol",
        condition_terms=("dravet syndrome",),
    )

    assert Tag(kind="safety", label="well tolerated", confidence=1.0, source="rule") in tags
