from __future__ import annotations

import pytest

from app.services import query_terms


def test_build_nih_search_query_includes_condition_mesh_terms() -> None:
    condition_terms = ["Muscular Dystrophy, Duchenne", "Becker Muscular Dystrophy"]

    query = query_terms.build_nih_search_query(condition_terms)

    assert "\"Muscular Dystrophy, Duchenne\"[mesh]" in query
    assert "\"Becker Muscular Dystrophy\"[mesh]" in query
    assert any(f"\"{term}\"[mesh]" in query for term in query_terms.ANESTHESIA_MESH_TERMS)
    assert any(f"\"{term}\"[tiab]" in query for term in query_terms.ANESTHESIA_TEXT_TERMS)
    assert any(f"\"{term}\"[mesh]" in query for term in query_terms.DRUG_MESH_TERMS)
    assert any(f"\"{term}\"[tiab]" in query for term in query_terms.DRUG_TEXT_TERMS)
    assert query.count("AND") >= 1
    assert query.count("OR") >= 4


def test_build_nih_search_query_requires_condition_terms() -> None:
    with pytest.raises(ValueError):
        query_terms.build_nih_search_query([])
