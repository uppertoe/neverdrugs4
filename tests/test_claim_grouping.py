from __future__ import annotations

from app.services.claims import ClaimEvidenceGroup, group_snippets_for_claims
from app.services.llm_batches import SnippetLLMEntry


def _entry(
    *,
    snippet_id: int,
    pmid: str,
    drug: str,
    classification: str,
    score: float,
    text: str,
) -> SnippetLLMEntry:
    return SnippetLLMEntry(
        pmid=pmid,
        snippet_id=snippet_id,
        drug=drug,
        classification=classification,
        snippet_text=text,
        snippet_score=score,
        cues=[],
        article_rank=1,
        article_score=score,
        citation_url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        article_title="title",
        content_source="pmc-full-text",
        token_estimate=120,
    )


def test_group_snippets_clusters_by_drug_and_classification() -> None:
    snippets = [
        _entry(
            snippet_id=1,
            pmid="111",
            drug="succinylcholine",
            classification="risk",
            score=5.0,
            text="Evidence for malignant hyperthermia risk",
        ),
        _entry(
            snippet_id=2,
            pmid="222",
            drug="succinylcholine",
            classification="risk",
            score=4.2,
            text="Additional caution noted",
        ),
        _entry(
            snippet_id=3,
            pmid="333",
            drug="propofol",
            classification="safety",
            score=3.5,
            text="Well tolerated case report",
        ),
    ]

    groups = group_snippets_for_claims(snippets)

    assert len(groups) == 2
    assert all(isinstance(group, ClaimEvidenceGroup) for group in groups)

    risk_group = next(group for group in groups if group.classification == "risk")
    assert risk_group.drug_label == "depolarising neuromuscular blockers"
    assert set(risk_group.drug_terms) == {"succinylcholine"}
    assert set(risk_group.drug_classes) == {"depolarising neuromuscular blocker"}
    assert {snippet.snippet_id for snippet in risk_group.snippets} == {1, 2}
    assert risk_group.top_score == 5.0

    safety_group = next(group for group in groups if group.classification == "safety")
    assert safety_group.drug_label == "propofol"
    assert safety_group.drug_classes == ()
    assert {snippet.snippet_id for snippet in safety_group.snippets} == {3}


def test_group_snippets_maps_related_drugs_to_shared_label() -> None:
    snippets = [
        _entry(
            snippet_id=10,
            pmid="444",
            drug="Sevoflurane",
            classification="risk",
            score=4.8,
            text="Sevoflurane triggered malignant hyperthermia",
        ),
        _entry(
            snippet_id=11,
            pmid="555",
            drug="desflurane",
            classification="risk",
            score=4.1,
            text="Desflurane implicated in MH episode",
        ),
    ]

    groups = group_snippets_for_claims(snippets)

    assert len(groups) == 1
    group = groups[0]
    assert group.drug_label == "volatile anesthetics"
    assert set(group.drug_terms) == {"Sevoflurane", "desflurane"}
    assert set(group.drug_classes) == {"volatile anesthetic"}
    assert group.top_score == 4.8


def test_group_snippets_separates_depolarising_and_non_depolarising_blockers() -> None:
    snippets = [
        _entry(
            snippet_id=20,
            pmid="666",
            drug="succinylcholine",
            classification="risk",
            score=5.1,
            text="Succinylcholine triggered malignant hyperthermia in the patient.",
        ),
        _entry(
            snippet_id=21,
            pmid="777",
            drug="rocuronium",
            classification="safety",
            score=3.9,
            text="Rocuronium was used as a safe non-depolarising alternative without complications.",
        ),
    ]

    groups = group_snippets_for_claims(snippets)

    assert len(groups) == 2
    depolarising = next(group for group in groups if "depolarising" in group.drug_label)
    non_depolarising = next(group for group in groups if "non-depolarising" in group.drug_label)

    assert depolarising.drug_label == "depolarising neuromuscular blockers"
    assert set(depolarising.drug_terms) == {"succinylcholine"}
    assert set(depolarising.drug_classes) == {"depolarising neuromuscular blocker"}

    assert non_depolarising.drug_label == "non-depolarising neuromuscular blockers"
    assert set(non_depolarising.drug_terms) == {"rocuronium"}
    assert set(non_depolarising.drug_classes) == {"non-depolarising neuromuscular blocker"}


def test_group_snippets_flags_generic_class_groups() -> None:
    generic = _entry(
        snippet_id=30,
        pmid="999001",
        drug="muscle relaxants",
        classification="risk",
        score=4.0,
        text="Muscle relaxants are associated with malignant hyperthermia in susceptible patients.",
    )
    specific = _entry(
        snippet_id=31,
        pmid="999002",
        drug="succinylcholine",
        classification="risk",
        score=5.2,
        text="Succinylcholine triggered malignant hyperthermia in the reported case.",
    )

    groups = group_snippets_for_claims([generic, specific])

    generic_group = next(group for group in groups if group.drug_label == "neuromuscular blocking agents")
    specific_group = next(group for group in groups if "succinylcholine" in group.drug_terms)

    assert "generic-class" in generic_group.drug_classes
    assert generic_group.top_score < specific_group.top_score
