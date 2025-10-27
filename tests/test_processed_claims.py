from __future__ import annotations

from app.models import (
    ArticleArtefact,
    ArticleSnippet,
    ProcessedClaim,
    ProcessedClaimDrugLink,
    ProcessedClaimSet,
    SearchArtefact,
    SearchTerm,
)
from app.services.processed_claims import persist_processed_claims


def _seed_search_term(session, mesh_signature: str) -> tuple[SearchTerm, SearchArtefact]:
    term = SearchTerm(canonical="king denborough syndrome")
    session.add(term)
    session.flush()

    artefact = SearchArtefact(
        search_term_id=term.id,
        query_payload={"mesh": ["King Denborough syndrome"]},
        mesh_terms=["King Denborough syndrome"],
        mesh_signature=mesh_signature,
        ttl_policy_seconds=86400,
    )
    session.add(artefact)
    session.flush()
    return term, artefact


def _seed_article_with_snippets(session, term: SearchTerm, *, pmid: str) -> tuple[ArticleArtefact, ArticleSnippet, ArticleSnippet]:
    artefact = ArticleArtefact(
        search_term_id=term.id,
        pmid=pmid,
        rank=1,
        score=5.0,
        citation={
            "pmid": pmid,
            "preferred_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            "title": "Safety considerations for neuromuscular blockade",
        },
        content=None,
        content_source="pubmed-abstract",
        token_estimate=200,
    )
    session.add(artefact)
    session.flush()

    snippet_one = ArticleSnippet(
        article_artefact_id=artefact.id,
        snippet_hash=f"{pmid}-s1",
        drug="succinylcholine",
        classification="risk",
        snippet_text="Evidence of malignant hyperthermia risk",
        snippet_score=5.5,
        cues=["malignant hyperthermia"],
    )
    snippet_two = ArticleSnippet(
        article_artefact_id=artefact.id,
        snippet_hash=f"{pmid}-s2",
        drug="succinylcholine",
        classification="risk",
        snippet_text="Second observation of risk",
        snippet_score=4.2,
        cues=["risk"],
    )
    session.add_all([snippet_one, snippet_two])
    session.flush()
    return artefact, snippet_one, snippet_two


def test_persist_processed_claims_merges_and_indexes(session) -> None:
    mesh_signature = "king-denborough|anesthesia"
    term, _ = _seed_search_term(session, mesh_signature)
    article, snippet_one, snippet_two = _seed_article_with_snippets(session, term, pmid="11111111")

    payloads = [
        {
            "condition": "King Denborough",
            "claims": [
                {
                    "claim_id": "risk:succinylcholine",
                    "classification": "risk",
                    "drug_classes": ["depolarising neuromuscular blocker"],
                    "drugs": ["succinylcholine"],
                    "summary": "Succinylcholine may trigger malignant hyperthermia.",
                    "confidence": "medium",
                    "supporting_evidence": [
                        {
                            "snippet_id": str(snippet_one.id),
                            "pmid": "11111111",
                            "article_title": "Safety considerations for neuromuscular blockade",
                            "key_points": [
                                "Review links depolarising blockers with malignant hyperthermia episodes."
                            ],
                            "notes": "Classification is risk based on explicit trigger statement.",
                        }
                    ],
                }
            ],
        },
        {
            "condition": "King Denborough",
            "claims": [
                {
                    "claim_id": "risk:succinylcholine-duplicate",
                    "classification": "risk",
                    "drug_classes": ["depolarising neuromuscular blocker"],
                    "drugs": ["succinylcholine"],
                    "summary": "Evidence from multiple reports indicates succinylcholine precipitates crises.",
                    "confidence": "high",
                    "supporting_evidence": [
                        {
                            "snippet_id": str(snippet_two.id),
                            "pmid": "11111111",
                            "article_title": "Safety considerations for neuromuscular blockade",
                            "key_points": [
                                "Case summaries describe malignant hyperthermia after succinylcholine exposure."
                            ],
                            "notes": "Reinforces trigger relationship with higher confidence.",
                        }
                    ],
                }
            ],
        },
    ]

    claim_set = persist_processed_claims(
        session,
        search_term_id=term.id,
        mesh_signature=mesh_signature,
        condition_label="King Denborough syndrome",
        llm_payloads=payloads,
    )

    session.flush()

    stored_sets = session.query(ProcessedClaimSet).all()
    assert len(stored_sets) == 1
    stored_set = stored_sets[0]
    assert stored_set.id == claim_set.id
    assert stored_set.mesh_signature == mesh_signature
    assert stored_set.condition_label == "King Denborough syndrome"

    assert len(stored_set.claims) == 1
    stored_claim = stored_set.claims[0]
    assert stored_claim.classification == "risk"
    assert stored_claim.confidence == "high"
    assert stored_claim.summary == "Evidence from multiple reports indicates succinylcholine precipitates crises."
    assert stored_claim.drugs == ["succinylcholine"]
    assert stored_claim.drug_classes == ["depolarising neuromuscular blocker"]
    assert set(stored_claim.source_claim_ids) == {
        "risk:succinylcholine",
        "risk:succinylcholine-duplicate",
    }

    evidence_entries = stored_claim.evidence
    assert len(evidence_entries) == 2
    evidence_snippet_ids = {entry.snippet_id for entry in evidence_entries}
    assert evidence_snippet_ids == {str(snippet_one.id), str(snippet_two.id)}
    expected_url = article.citation["preferred_url"]
    assert all(entry.citation_url == expected_url for entry in evidence_entries)

    links = session.query(ProcessedClaimDrugLink).all()
    terms = {(link.term, link.term_kind) for link in links}
    assert terms == {
        ("succinylcholine", "drug"),
        ("depolarising neuromuscular blocker", "drug_class"),
    }
    assert all(link.claim_id == stored_claim.id for link in links)


def test_persist_processed_claims_replaces_existing_set(session) -> None:
    mesh_signature = "king-denborough|anesthesia"
    term, _ = _seed_search_term(session, mesh_signature)

    # Pre-create an old processed set with a different claim to ensure replacement occurs.
    old_set = ProcessedClaimSet(mesh_signature=mesh_signature, condition_label="Old label")
    old_claim = ProcessedClaim(
        claim_id="old",
        classification="risk",
        summary="Outdated",
        confidence="low",
        drugs=["old"],
        drug_classes=[],
        source_claim_ids=["old"],
    )
    old_set.claims.append(old_claim)
    session.add(old_set)
    session.flush()

    payloads = [
        {
            "condition": "King Denborough",
            "claims": [
                {
                    "claim_id": "risk:new",
                    "classification": "risk",
                    "drug_classes": ["volatile anesthetic"],
                    "drugs": ["sevoflurane"],
                    "summary": "Volatile anesthetics can trigger malignant hyperthermia.",
                    "confidence": "medium",
                    "supporting_evidence": [],
                }
            ],
        }
    ]

    claim_set = persist_processed_claims(
        session,
        search_term_id=term.id,
        mesh_signature=mesh_signature,
        condition_label="King Denborough syndrome",
        llm_payloads=payloads,
    )

    session.flush()

    refreshed_set = session.get(ProcessedClaimSet, claim_set.id)
    assert refreshed_set is not None
    assert refreshed_set.condition_label == "King Denborough syndrome"
    assert len(refreshed_set.claims) == 1
    assert refreshed_set.claims[0].claim_id == "risk:new"
    # Old claim should be removed.
    assert session.query(ProcessedClaim).filter(ProcessedClaim.claim_id == "old").count() == 0