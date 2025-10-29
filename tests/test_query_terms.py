from __future__ import annotations

import pytest

from app.services import query_terms
from app.services.query_terms import ConditionTermExpansion


def test_build_nih_search_query_includes_condition_mesh_terms() -> None:
    condition_terms = ["Muscular Dystrophy, Duchenne", "Becker Muscular Dystrophy"]

    query = query_terms.build_nih_search_query(condition_terms)

    assert "\"Muscular Dystrophy, Duchenne\"[mesh]" in query
    assert "\"Becker Muscular Dystrophy\"[mesh]" in query
    assert any(f"\"{term}\"[mesh]" in query for term in query_terms.ANESTHESIA_MESH_TERMS)
    assert any(f"\"{term}\"[tiab]" in query for term in query_terms.ANESTHESIA_TEXT_TERMS)
    assert any(f"\"{term}\"[mesh]" in query for term in query_terms.DRUG_MESH_TERMS)
    assert any(f"\"{term}\"[tiab]" in query for term in query_terms.DRUG_QUERY_TEXT_TERMS)
    assert any(f"\"{term}\"[mesh]" in query for term in query_terms.CLINICAL_CONTEXT_MESH_TERMS)
    assert any(f"\"{term}\"[tiab]" in query for term in query_terms.CLINICAL_CONTEXT_TEXT_TERMS)
    assert any(f"\"{term}\"[pt]" in query for term in query_terms.FOCUSED_PUBLICATION_TYPES)
    assert "\"Humans\"[mesh]" in query
    assert query.count("AND") >= 3
    assert query.count("OR") >= 4


def test_build_nih_search_query_requires_condition_terms() -> None:
    with pytest.raises(ValueError):
        query_terms.build_nih_search_query([])


def test_build_nih_search_query_applies_condition_expander() -> None:
    def _expand(term: str) -> ConditionTermExpansion | None:
        if term.lower() != "dravet syndrome":
            return None
        return ConditionTermExpansion(
            mesh_terms=("Epilepsies, Myoclonic",),
            alias_terms=("Dravet Syndrome", "SCN1A"),
        )

    query = query_terms.build_nih_search_query(["Dravet Syndrome"], term_expander=_expand)

    assert '"Epilepsies, Myoclonic"[mesh]' in query
    assert '"Dravet Syndrome"[tiab]' in query
    assert '"SCN1A"[tiab]' in query


def test_drug_mesh_terms_cover_drug_trolley_inventory() -> None:
    trolley_to_mesh = {
        "propofol": {"Anesthetics, Intravenous"},
        "atracurium": {"Neuromuscular Blocking Agents"},
        "rocuronium": {"Neuromuscular Blocking Agents"},
        "neostigmine": {"Cholinesterase Inhibitors"},
        "sugammadex": {"Cyclodextrins"},
        "atropine": {"Parasympatholytics"},
        "adrenaline": {"Adrenergic Agonists"},
        "noradrenaline": {"Adrenergic Agonists"},
        "calcium gluconate": {"Calcium Compounds"},
        "parecoxib": {"Cyclooxygenase 2 Inhibitors"},
        "morphine": {"Analgesics, Opioid"},
        "fentanyl": {"Analgesics, Opioid"},
        "remifentanil": {"Analgesics, Opioid"},
        "cefazolin": {"Cephalosporins"},
        "dexamethasone": {"Adrenal Cortex Hormones"},
        "ondansetron": {"Antiemetics", "Serotonin Antagonists"},
        "metoclopramide": {"Antiemetics"},
        "hyoscine": {"Parasympatholytics"},
        "glucagon": {"Glucagon"},
    "hartmann's solution": {"Ringer's Lactate"},
        "sodium chloride": {"Sodium Chloride", "Electrolyte Solutions"},
        "hydralazine": {"Vasodilator Agents"},
        "gtn": {"Nitrates"},
        "sodium nitroprusside": {"Vasodilator Agents"},
        "levetiracetam": {"Anticonvulsants"},
        "amiodarone": {"Anti-Arrhythmia Agents"},
        "metoprolol": {"Adrenergic beta-Antagonists"},
        "adenosine": {"Adenosine"},
        "lignocaine": {"Anesthetics, Local"},
    "dantrolene": {"Muscle Relaxants, Central"},
    }

    mesh_terms = set(query_terms.DRUG_MESH_TERMS)
    for agent, expected_mesh in trolley_to_mesh.items():
        assert mesh_terms.intersection(expected_mesh), f"{agent} missing mesh coverage"
