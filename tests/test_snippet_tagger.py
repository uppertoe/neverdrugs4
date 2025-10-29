from __future__ import annotations

from app.services.snippet_tagger import RuleBasedSnippetTagger


def test_rule_based_tagger_tags_severe_reactions() -> None:
    tagger = RuleBasedSnippetTagger()
    snippet = (
        "Following induction, the patient developed ventricular tachycardia and cardiac arrest "
        "requiring prolonged resuscitation."
    )

    tags = tagger.tag_snippet(snippet, drug="propofol", condition_terms=("dravet syndrome",))

    severe_labels = {tag.label for tag in tags if tag.kind == "severe_reaction"}
    assert "ventricular tachycardia" in severe_labels or "cardiac arrest" in severe_labels


def test_rule_based_tagger_ignores_expected_effects_for_severe_tags() -> None:
    tagger = RuleBasedSnippetTagger()
    snippet = "Mild respiratory depression was observed after fentanyl administration."

    tags = tagger.tag_snippet(snippet, drug="fentanyl", condition_terms=("dravet syndrome",))

    assert not any(tag.kind == "severe_reaction" for tag in tags)


def test_rule_based_tagger_tags_therapy_role() -> None:
    tagger = RuleBasedSnippetTagger()
    snippet = "Ketamine served as rescue therapy when benzodiazepines failed."

    tags = tagger.tag_snippet(snippet, drug="ketamine", condition_terms=("status epilepticus",))

    assert any(tag.kind == "therapy_role" and tag.label == "rescue" for tag in tags)


def test_rule_based_tagger_tags_mechanism_alert() -> None:
    tagger = RuleBasedSnippetTagger()
    snippet = "Prolonged paralysis occurred due to pseudocholinesterase deficiency after succinylcholine."

    tags = tagger.tag_snippet(snippet, drug="succinylcholine", condition_terms=("mh",))

    assert any(tag.kind == "mechanism_alert" and tag.label == "pseudocholinesterase deficiency" for tag in tags)


def test_rule_based_tagger_requires_qualifier_for_conditional_severe_cue() -> None:
    tagger = RuleBasedSnippetTagger()
    snippet = "A brief seizure occurred post-induction but was expected."

    tags = tagger.tag_snippet(snippet, drug="ketamine", condition_terms=("dravet",))

    assert not any(tag.kind == "severe_reaction" for tag in tags)


def test_rule_based_tagger_tags_conditional_severe_with_qualifier() -> None:
    tagger = RuleBasedSnippetTagger()
    snippet = "An unexpected seizure and convulsions followed the induction dose."

    tags = tagger.tag_snippet(snippet, drug="propofol", condition_terms=("dravet",))

    severe_labels = {tag.label for tag in tags if tag.kind == "severe_reaction"}
    assert "seizure" in severe_labels or "convulsions" in severe_labels
