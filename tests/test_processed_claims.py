from __future__ import annotations

import pytest

from app.models import (
    ArticleArtefact,
    ArticleSnippet,
    ProcessedClaim,
    ProcessedClaimDrugLink,
    ProcessedClaimSet,
    SearchArtefact,
    SearchTerm,
)
from app.services.processed_claims import InvalidClaimPayload, persist_processed_claims


def _seed_search_term(session, mesh_signature: str) -> tuple[SearchTerm, SearchArtefact]:
    term = SearchTerm(canonical="king denborough syndrome")
    session.add(term)
    session.flush()

    artefact = SearchArtefact(
        search_term_id=term.id,
        query_payload={"mesh": ["King Denborough syndrome"]},
        mesh_terms=["King Denborough syndrome"],
        mesh_signature=mesh_signature,
        result_signature=mesh_signature,
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
        tags=[],
    )
    snippet_two = ArticleSnippet(
        article_artefact_id=artefact.id,
        snippet_hash=f"{pmid}-s2",
        drug="succinylcholine",
        classification="risk",
        snippet_text="Second observation of risk",
        snippet_score=4.2,
        cues=["risk"],
        tags=[],
    )
    session.add_all([snippet_one, snippet_two])
    session.flush()
    return artefact, snippet_one, snippet_two


def _persist_payloads(
    session,
    *,
    mesh_signature: str,
    term: SearchTerm,
    payloads: list[dict],
) -> ProcessedClaimSet:
    return persist_processed_claims(
        session,
        search_term_id=term.id,
        mesh_signature=mesh_signature,
        condition_label="King Denborough syndrome",
        llm_payloads=payloads,
    )


def _single_claim_payload(
    *,
    article: ArticleArtefact,
    snippet: ArticleSnippet,
    claim_id: str = "claim:succinylcholine-risk",
    confidence: str = "medium",
    summary: str = "Succinylcholine may trigger malignant hyperthermia.",
    descriptors: list[str] | None = None,
    drug_id: str = "drug:succinylcholine",
    drug_name: str = "Succinylcholine",
    top_level_classes: list[str] | None = None,
    claim_drug_classes: list[str] | None = None,
) -> dict:
    descriptors = descriptors or ["anaphylaxis"]
    top_level_classes = top_level_classes or ["depolarising neuromuscular blocker"]
    claim_drug_classes = claim_drug_classes or ["neuromuscular blocker"]
    return {
        "condition": "King Denborough",
        "drugs": [
            {
                "id": drug_id,
                "name": drug_name,
                "classifications": list(top_level_classes),
                "claims": [claim_id],
            }
        ],
        "claims": [
            {
                "id": claim_id,
                "type": "risk",
                "summary": summary,
                "confidence": confidence,
                "drugs": [drug_id],
                "drug_classes": list(claim_drug_classes),
                "idiosyncratic_reaction": {
                    "flag": True,
                    "descriptors": descriptors,
                },
                "articles": [f"article:{article.pmid}"],
                "supporting_evidence": [
                    {
                        "snippet_id": str(snippet.id),
                        "pmid": article.pmid,
                        "article_title": article.citation["title"],
                        "key_points": [
                            "Review links depolarising blockers with malignant hyperthermia episodes."
                        ],
                        "notes": "Classification is risk based on explicit trigger statement.",
                    }
                ],
            }
        ],
    }


def test_persist_processed_claims_merges_and_indexes(session) -> None:
    mesh_signature = "king-denborough|anesthesia"
    term, _ = _seed_search_term(session, mesh_signature)
    article, snippet_one, snippet_two = _seed_article_with_snippets(session, term, pmid="11111111")

    first_payload = _single_claim_payload(article=article, snippet=snippet_one)
    second_payload = _single_claim_payload(
        article=article,
        snippet=snippet_two,
        claim_id="claim:succinylcholine-risk-duplicate",
        confidence="high",
        summary="Evidence from multiple reports indicates succinylcholine precipitates crises.",
        descriptors=["Cardiac arrest"],
    )

    claim_set = _persist_payloads(
        session,
        mesh_signature=mesh_signature,
        term=term,
        payloads=[first_payload, second_payload],
    )

    stored_set = session.get(ProcessedClaimSet, claim_set.id)
    assert stored_set is not None
    assert stored_set.mesh_signature == mesh_signature
    assert stored_set.condition_label == "King Denborough syndrome"

    assert len(stored_set.claims) == 1
    stored_claim = stored_set.claims[0]
    assert stored_claim.claim_id == "claim:succinylcholine-risk-duplicate"
    assert stored_claim.classification == "risk"
    assert stored_claim.confidence == "high"
    assert stored_claim.summary == "Evidence from multiple reports indicates succinylcholine precipitates crises."
    assert stored_claim.drugs == ["Succinylcholine"]
    assert stored_claim.drug_classes == ["depolarising neuromuscular blocker", "neuromuscular blocker"]
    assert stored_claim.severe_reaction_flag is True
    assert set(stored_claim.severe_reaction_terms) == {"anaphylaxis", "Cardiac arrest"}
    assert set(stored_claim.source_claim_ids) == {
        "claim:succinylcholine-risk",
        "claim:succinylcholine-risk-duplicate",
    }

    evidence_ids = {entry.snippet_id for entry in stored_claim.evidence}
    assert evidence_ids == {str(snippet_one.id), str(snippet_two.id), "article:11111111"}
    for entry in stored_claim.evidence:
        if entry.snippet_id.isdigit():
            assert entry.citation_url == article.citation["preferred_url"]
        else:
            assert entry.citation_url == "https://pubmed.ncbi.nlm.nih.gov/11111111/"

    links = session.query(ProcessedClaimDrugLink).order_by(ProcessedClaimDrugLink.id).all()
    terms = {(link.term, link.term_kind) for link in links}
    assert terms == {
        ("Succinylcholine", "drug"),
        ("depolarising neuromuscular blocker", "drug_class"),
        ("neuromuscular blocker", "drug_class"),
    }
    assert {link.claim_id for link in links} == {stored_claim.id}


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
        severe_reaction_flag=False,
        severe_reaction_terms=[],
    )
    old_set.claims.append(old_claim)
    session.add(old_set)
    session.flush()

    article, snippet_one, _ = _seed_article_with_snippets(session, term, pmid="22222222")
    payload = _single_claim_payload(
        article=article,
        snippet=snippet_one,
        claim_id="claim:new",
        confidence="medium",
        summary="Volatile anesthetics can trigger malignant hyperthermia.",
        drug_id="drug:sevoflurane",
        drug_name="Sevoflurane",
        top_level_classes=["volatile anesthetic"],
        claim_drug_classes=["inhalational anesthetic"],
    )

    claim_set = _persist_payloads(
        session,
        mesh_signature=mesh_signature,
        term=term,
        payloads=[payload],
    )

    session.flush()

    refreshed_set = session.get(ProcessedClaimSet, claim_set.id)
    assert refreshed_set is not None
    assert refreshed_set.condition_label == "King Denborough syndrome"
    assert len(refreshed_set.claims) == 1
    assert refreshed_set.claims[0].claim_id == "claim:new"
    # Old claim should be removed.
    assert session.query(ProcessedClaim).filter(ProcessedClaim.claim_id == "old").count() == 0


def test_persist_processed_claims_rejects_empty_drug_array(session) -> None:
    mesh_signature = "king-denborough|anesthesia"
    term, _ = _seed_search_term(session, mesh_signature)
    article, snippet_one, _ = _seed_article_with_snippets(session, term, pmid="33333333")

    payload = _single_claim_payload(article=article, snippet=snippet_one)
    payload["drugs"] = []

    with pytest.raises(InvalidClaimPayload, match="Drugs array must contain at least one entry"):
        _persist_payloads(session, mesh_signature=mesh_signature, term=term, payloads=[payload])


def test_persist_processed_claims_rejects_missing_claim_id(session) -> None:
    mesh_signature = "king-denborough|anesthesia"
    term, _ = _seed_search_term(session, mesh_signature)
    article, snippet_one, _ = _seed_article_with_snippets(session, term, pmid="44444444")

    payload = _single_claim_payload(article=article, snippet=snippet_one)
    claim = payload["claims"][0]
    claim.pop("id")

    with pytest.raises(InvalidClaimPayload, match="missing required field 'id'"):
        _persist_payloads(session, mesh_signature=mesh_signature, term=term, payloads=[payload])


def test_persist_processed_claims_rejects_missing_type(session) -> None:
    mesh_signature = "king-denborough|anesthesia"
    term, _ = _seed_search_term(session, mesh_signature)
    article, snippet_one, _ = _seed_article_with_snippets(session, term, pmid="45454545")

    payload = _single_claim_payload(article=article, snippet=snippet_one)
    payload["claims"][0].pop("type")

    with pytest.raises(InvalidClaimPayload, match="missing required field 'type'"):
        _persist_payloads(session, mesh_signature=mesh_signature, term=term, payloads=[payload])


def test_persist_processed_claims_rejects_invalid_type(session) -> None:
    mesh_signature = "king-denborough|anesthesia"
    term, _ = _seed_search_term(session, mesh_signature)
    article, snippet_one, _ = _seed_article_with_snippets(session, term, pmid="45454546")

    payload = _single_claim_payload(article=article, snippet=snippet_one)
    payload["claims"][0]["type"] = "legacy-risk"

    with pytest.raises(
        InvalidClaimPayload,
        match="LLM claim payload has invalid type 'legacy-risk'",
    ):
        _persist_payloads(session, mesh_signature=mesh_signature, term=term, payloads=[payload])


def test_persist_processed_claims_rejects_missing_summary(session) -> None:
    mesh_signature = "king-denborough|anesthesia"
    term, _ = _seed_search_term(session, mesh_signature)
    article, snippet_one, _ = _seed_article_with_snippets(session, term, pmid="45454547")

    payload = _single_claim_payload(article=article, snippet=snippet_one)
    payload["claims"][0].pop("summary")

    with pytest.raises(InvalidClaimPayload, match="missing required field 'summary'"):
        _persist_payloads(session, mesh_signature=mesh_signature, term=term, payloads=[payload])


def test_persist_processed_claims_rejects_missing_confidence(session) -> None:
    mesh_signature = "king-denborough|anesthesia"
    term, _ = _seed_search_term(session, mesh_signature)
    article, snippet_one, _ = _seed_article_with_snippets(session, term, pmid="45454548")

    payload = _single_claim_payload(article=article, snippet=snippet_one)
    payload["claims"][0].pop("confidence")

    with pytest.raises(InvalidClaimPayload, match="missing required field 'confidence'"):
        _persist_payloads(session, mesh_signature=mesh_signature, term=term, payloads=[payload])


def test_persist_processed_claims_rejects_invalid_confidence(session) -> None:
    mesh_signature = "king-denborough|anesthesia"
    term, _ = _seed_search_term(session, mesh_signature)
    article, snippet_one, _ = _seed_article_with_snippets(session, term, pmid="45454549")

    payload = _single_claim_payload(article=article, snippet=snippet_one)
    payload["claims"][0]["confidence"] = "uncertain"

    with pytest.raises(
        InvalidClaimPayload,
        match="LLM claim payload has invalid confidence 'uncertain'",
    ):
        _persist_payloads(session, mesh_signature=mesh_signature, term=term, payloads=[payload])


def test_persist_processed_claims_rejects_malformed_reaction(session) -> None:
    mesh_signature = "king-denborough|anesthesia"
    term, _ = _seed_search_term(session, mesh_signature)
    article, snippet_one, _ = _seed_article_with_snippets(session, term, pmid="55555555")

    payload = _single_claim_payload(article=article, snippet=snippet_one)
    payload["claims"][0]["idiosyncratic_reaction"]["flag"] = "yes"

    with pytest.raises(
        InvalidClaimPayload,
        match="idiosyncratic_reaction.flag must be boolean",
    ):
        _persist_payloads(session, mesh_signature=mesh_signature, term=term, payloads=[payload])


def test_persist_processed_claims_rejects_missing_reaction_flag(session) -> None:
    mesh_signature = "king-denborough|anesthesia"
    term, _ = _seed_search_term(session, mesh_signature)
    article, snippet_one, _ = _seed_article_with_snippets(session, term, pmid="56565651")

    payload = _single_claim_payload(article=article, snippet=snippet_one)
    payload["claims"][0]["idiosyncratic_reaction"].pop("flag")

    with pytest.raises(
        InvalidClaimPayload,
        match="idiosyncratic_reaction missing 'flag'",
    ):
        _persist_payloads(session, mesh_signature=mesh_signature, term=term, payloads=[payload])


def test_persist_processed_claims_rejects_missing_reaction_descriptors(session) -> None:
    mesh_signature = "king-denborough|anesthesia"
    term, _ = _seed_search_term(session, mesh_signature)
    article, snippet_one, _ = _seed_article_with_snippets(session, term, pmid="56565652")

    payload = _single_claim_payload(article=article, snippet=snippet_one)
    payload["claims"][0]["idiosyncratic_reaction"].pop("descriptors")

    with pytest.raises(
        InvalidClaimPayload,
        match="idiosyncratic_reaction.descriptors missing",
    ):
        _persist_payloads(session, mesh_signature=mesh_signature, term=term, payloads=[payload])


def test_persist_processed_claims_rejects_scalar_reaction_descriptors(session) -> None:
    mesh_signature = "king-denborough|anesthesia"
    term, _ = _seed_search_term(session, mesh_signature)
    article, snippet_one, _ = _seed_article_with_snippets(session, term, pmid="56565653")

    payload = _single_claim_payload(article=article, snippet=snippet_one)
    payload["claims"][0]["idiosyncratic_reaction"]["descriptors"] = "anaphylaxis"

    with pytest.raises(
        InvalidClaimPayload,
        match="idiosyncratic_reaction.descriptors must be an array of strings",
    ):
        _persist_payloads(session, mesh_signature=mesh_signature, term=term, payloads=[payload])


def test_persist_processed_claims_rejects_invalid_supporting_evidence(session) -> None:
    mesh_signature = "king-denborough|anesthesia"
    term, _ = _seed_search_term(session, mesh_signature)
    article, snippet_one, _ = _seed_article_with_snippets(session, term, pmid="66666666")

    payload = _single_claim_payload(article=article, snippet=snippet_one)
    payload["claims"][0]["supporting_evidence"][0]["key_points"] = []

    with pytest.raises(
        InvalidClaimPayload,
        match="supporting_evidence.key_points must contain at least one entry",
    ):
        _persist_payloads(session, mesh_signature=mesh_signature, term=term, payloads=[payload])


def test_persist_processed_claims_rejects_nonobject_supporting_evidence(session) -> None:
    mesh_signature = "king-denborough|anesthesia"
    term, _ = _seed_search_term(session, mesh_signature)
    article, snippet_one, _ = _seed_article_with_snippets(session, term, pmid="67676761")

    payload = _single_claim_payload(article=article, snippet=snippet_one)
    payload["claims"][0]["supporting_evidence"] = ["not-an-object"]

    with pytest.raises(
        InvalidClaimPayload,
        match="supporting_evidence must be objects",
    ):
        _persist_payloads(session, mesh_signature=mesh_signature, term=term, payloads=[payload])


def test_persist_processed_claims_rejects_supporting_evidence_missing_snippet(session) -> None:
    mesh_signature = "king-denborough|anesthesia"
    term, _ = _seed_search_term(session, mesh_signature)
    article, snippet_one, _ = _seed_article_with_snippets(session, term, pmid="67676762")

    payload = _single_claim_payload(article=article, snippet=snippet_one)
    payload["claims"][0]["supporting_evidence"][0].pop("snippet_id")

    with pytest.raises(
        InvalidClaimPayload,
        match="supporting_evidence missing 'snippet_id'",
    ):
        _persist_payloads(session, mesh_signature=mesh_signature, term=term, payloads=[payload])


def test_persist_processed_claims_rejects_supporting_evidence_missing_pmid(session) -> None:
    mesh_signature = "king-denborough|anesthesia"
    term, _ = _seed_search_term(session, mesh_signature)
    article, snippet_one, _ = _seed_article_with_snippets(session, term, pmid="67676763")

    payload = _single_claim_payload(article=article, snippet=snippet_one)
    payload["claims"][0]["supporting_evidence"][0].pop("pmid")

    with pytest.raises(
        InvalidClaimPayload,
        match="supporting_evidence missing 'pmid'",
    ):
        _persist_payloads(session, mesh_signature=mesh_signature, term=term, payloads=[payload])


def test_persist_processed_claims_rejects_supporting_evidence_missing_title(session) -> None:
    mesh_signature = "king-denborough|anesthesia"
    term, _ = _seed_search_term(session, mesh_signature)
    article, snippet_one, _ = _seed_article_with_snippets(session, term, pmid="67676764")

    payload = _single_claim_payload(article=article, snippet=snippet_one)
    payload["claims"][0]["supporting_evidence"][0].pop("article_title")

    with pytest.raises(
        InvalidClaimPayload,
        match="supporting_evidence missing 'article_title'",
    ):
        _persist_payloads(session, mesh_signature=mesh_signature, term=term, payloads=[payload])


def test_persist_processed_claims_rejects_supporting_evidence_scalar_key_points(session) -> None:
    mesh_signature = "king-denborough|anesthesia"
    term, _ = _seed_search_term(session, mesh_signature)
    article, snippet_one, _ = _seed_article_with_snippets(session, term, pmid="67676765")

    payload = _single_claim_payload(article=article, snippet=snippet_one)
    payload["claims"][0]["supporting_evidence"][0]["key_points"] = "not-an-array"

    with pytest.raises(
        InvalidClaimPayload,
        match="supporting_evidence.key_points must be an array of strings",
    ):
        _persist_payloads(session, mesh_signature=mesh_signature, term=term, payloads=[payload])


def test_persist_processed_claims_rejects_noncanonical_articles(session) -> None:
    mesh_signature = "king-denborough|anesthesia"
    term, _ = _seed_search_term(session, mesh_signature)
    article, snippet_one, _ = _seed_article_with_snippets(session, term, pmid="77777777")

    payload = _single_claim_payload(article=article, snippet=snippet_one)
    payload["claims"][0]["articles"] = ["77777777"]

    with pytest.raises(
        InvalidClaimPayload,
        match="articles must use 'article:' prefix",
    ):
        _persist_payloads(session, mesh_signature=mesh_signature, term=term, payloads=[payload])


def test_persist_processed_claims_rejects_scalar_articles(session) -> None:
    mesh_signature = "king-denborough|anesthesia"
    term, _ = _seed_search_term(session, mesh_signature)
    article, snippet_one, _ = _seed_article_with_snippets(session, term, pmid="78787871")

    payload = _single_claim_payload(article=article, snippet=snippet_one)
    payload["claims"][0]["articles"] = "article:78787871"

    with pytest.raises(
        InvalidClaimPayload,
        match="articles must be an array of strings",
    ):
        _persist_payloads(session, mesh_signature=mesh_signature, term=term, payloads=[payload])


def test_persist_processed_claims_rejects_empty_article_identifier(session) -> None:
    mesh_signature = "king-denborough|anesthesia"
    term, _ = _seed_search_term(session, mesh_signature)
    article, snippet_one, _ = _seed_article_with_snippets(session, term, pmid="78787872")

    payload = _single_claim_payload(article=article, snippet=snippet_one)
    payload["claims"][0]["articles"] = [""]

    with pytest.raises(
        InvalidClaimPayload,
        match="articles contains an empty identifier",
    ):
        _persist_payloads(session, mesh_signature=mesh_signature, term=term, payloads=[payload])


def test_persist_processed_claims_rejects_articles_with_non_numeric_suffix(session) -> None:
    mesh_signature = "king-denborough|anesthesia"
    term, _ = _seed_search_term(session, mesh_signature)
    article, snippet_one, _ = _seed_article_with_snippets(session, term, pmid="78787873")

    payload = _single_claim_payload(article=article, snippet=snippet_one)
    payload["claims"][0]["articles"] = ["article:not-a-number"]

    with pytest.raises(
        InvalidClaimPayload,
        match="articles must use 'article:' followed by numeric PMID",
    ):
        _persist_payloads(session, mesh_signature=mesh_signature, term=term, payloads=[payload])


def test_persist_processed_claims_rejects_unknown_drug_reference(session) -> None:
    mesh_signature = "king-denborough|anesthesia"
    term, _ = _seed_search_term(session, mesh_signature)
    article, snippet_one, _ = _seed_article_with_snippets(session, term, pmid="88888888")

    payload = _single_claim_payload(article=article, snippet=snippet_one)
    payload["claims"][0]["drugs"] = ["drug:unknown"]

    with pytest.raises(
        InvalidClaimPayload,
        match="Claim references unknown drug id 'drug:unknown'",
    ):
        _persist_payloads(session, mesh_signature=mesh_signature, term=term, payloads=[payload])


def test_persist_processed_claims_rejects_scalar_claim_drugs(session) -> None:
    mesh_signature = "king-denborough|anesthesia"
    term, _ = _seed_search_term(session, mesh_signature)
    article, snippet_one, _ = _seed_article_with_snippets(session, term, pmid="89898981")

    payload = _single_claim_payload(article=article, snippet=snippet_one)
    payload["claims"][0]["drugs"] = "drug:succinylcholine"

    with pytest.raises(
        InvalidClaimPayload,
        match="claim.drugs must be an array of canonical ids",
    ):
        _persist_payloads(session, mesh_signature=mesh_signature, term=term, payloads=[payload])


def test_persist_processed_claims_rejects_empty_claim_drug_identifier(session) -> None:
    mesh_signature = "king-denborough|anesthesia"
    term, _ = _seed_search_term(session, mesh_signature)
    article, snippet_one, _ = _seed_article_with_snippets(session, term, pmid="89898982")

    payload = _single_claim_payload(article=article, snippet=snippet_one)
    payload["claims"][0]["drugs"] = [""]

    with pytest.raises(
        InvalidClaimPayload,
        match="claim.drugs contains an empty identifier",
    ):
        _persist_payloads(session, mesh_signature=mesh_signature, term=term, payloads=[payload])


def test_persist_processed_claims_rejects_noncanonical_claim_drug_identifier(session) -> None:
    mesh_signature = "king-denborough|anesthesia"
    term, _ = _seed_search_term(session, mesh_signature)
    article, snippet_one, _ = _seed_article_with_snippets(session, term, pmid="89898983")

    payload = _single_claim_payload(article=article, snippet=snippet_one)
    payload["claims"][0]["drugs"] = ["succinylcholine"]

    with pytest.raises(
        InvalidClaimPayload,
        match="claim.drugs must use canonical 'drug:' identifiers",
    ):
        _persist_payloads(session, mesh_signature=mesh_signature, term=term, payloads=[payload])


def test_persist_processed_claims_rejects_scalar_claim_drug_classes(session) -> None:
    mesh_signature = "king-denborough|anesthesia"
    term, _ = _seed_search_term(session, mesh_signature)
    article, snippet_one, _ = _seed_article_with_snippets(session, term, pmid="89898984")

    payload = _single_claim_payload(article=article, snippet=snippet_one)
    payload["claims"][0]["drug_classes"] = "not-a-list"

    with pytest.raises(
        InvalidClaimPayload,
        match="claim.drug_classes must be an array of strings",
    ):
        _persist_payloads(session, mesh_signature=mesh_signature, term=term, payloads=[payload])


def test_persist_processed_claims_rejects_claim_without_declaration(session) -> None:
    mesh_signature = "king-denborough|anesthesia"
    term, _ = _seed_search_term(session, mesh_signature)
    article, snippet_one, _ = _seed_article_with_snippets(session, term, pmid="99999999")

    payload = _single_claim_payload(article=article, snippet=snippet_one)
    payload["drugs"][0]["claims"] = []

    with pytest.raises(
        InvalidClaimPayload,
        match="is not declared in the top-level drugs claims list",
    ):
        _persist_payloads(session, mesh_signature=mesh_signature, term=term, payloads=[payload])


def test_persist_processed_claims_rejects_dangling_declared_claim(session) -> None:
    mesh_signature = "king-denborough|anesthesia"
    term, _ = _seed_search_term(session, mesh_signature)
    article, snippet_one, _ = _seed_article_with_snippets(session, term, pmid="10101010")

    payload = _single_claim_payload(article=article, snippet=snippet_one)
    payload["claims"] = []

    with pytest.raises(
        InvalidClaimPayload,
        match="Drug metadata declares claims without payload entries",
    ):
        _persist_payloads(session, mesh_signature=mesh_signature, term=term, payloads=[payload])


def test_persist_processed_claims_rejects_claim_referencing_undeclared_drug(session) -> None:
    mesh_signature = "king-denborough|anesthesia"
    term, _ = _seed_search_term(session, mesh_signature)
    article, snippet_one, _ = _seed_article_with_snippets(session, term, pmid="10101011")

    payload = _single_claim_payload(article=article, snippet=snippet_one)
    payload["claims"][0]["drugs"] = ["drug:succinylcholine", "drug:sevoflurane"]
    payload["drugs"][0]["claims"].append(payload["claims"][0]["id"])
    payload["drugs"].append(
        {
            "id": "drug:sevoflurane",
            "name": "Sevoflurane",
            "classifications": ["volatile anesthetic"],
            "claims": [],
        }
    )

    with pytest.raises(
        InvalidClaimPayload,
        match="references drug 'drug:sevoflurane' without declaration",
    ):
        _persist_payloads(session, mesh_signature=mesh_signature, term=term, payloads=[payload])


def test_persist_processed_claims_rejects_scalar_drug_catalog_entries(session) -> None:
    mesh_signature = "king-denborough|anesthesia"
    term, _ = _seed_search_term(session, mesh_signature)
    article, snippet_one, _ = _seed_article_with_snippets(session, term, pmid="20202021")

    payload = _single_claim_payload(article=article, snippet=snippet_one)
    payload["drugs"] = "not-a-list"

    with pytest.raises(
        InvalidClaimPayload,
        match="LLM payload drugs must be an array of objects",
    ):
        _persist_payloads(session, mesh_signature=mesh_signature, term=term, payloads=[payload])


def test_persist_processed_claims_rejects_nonobject_drug_entry(session) -> None:
    mesh_signature = "king-denborough|anesthesia"
    term, _ = _seed_search_term(session, mesh_signature)
    article, snippet_one, _ = _seed_article_with_snippets(session, term, pmid="20202022")

    payload = _single_claim_payload(article=article, snippet=snippet_one)
    payload["drugs"] = ["not-an-object"]

    with pytest.raises(
        InvalidClaimPayload,
        match="LLM payload drugs must contain objects",
    ):
        _persist_payloads(session, mesh_signature=mesh_signature, term=term, payloads=[payload])


def test_persist_processed_claims_rejects_drug_entry_missing_id(session) -> None:
    mesh_signature = "king-denborough|anesthesia"
    term, _ = _seed_search_term(session, mesh_signature)
    article, snippet_one, _ = _seed_article_with_snippets(session, term, pmid="20202023")

    payload = _single_claim_payload(article=article, snippet=snippet_one)
    payload["drugs"][0].pop("id")

    with pytest.raises(
        InvalidClaimPayload,
        match="drugs entries must include canonical 'drug:' ids",
    ):
        _persist_payloads(session, mesh_signature=mesh_signature, term=term, payloads=[payload])


def test_persist_processed_claims_rejects_drug_entry_noncanonical_id(session) -> None:
    mesh_signature = "king-denborough|anesthesia"
    term, _ = _seed_search_term(session, mesh_signature)
    article, snippet_one, _ = _seed_article_with_snippets(session, term, pmid="20202024")

    payload = _single_claim_payload(article=article, snippet=snippet_one)
    payload["drugs"][0]["id"] = "succinylcholine"

    with pytest.raises(
        InvalidClaimPayload,
        match="drugs entries must include canonical 'drug:' ids",
    ):
        _persist_payloads(session, mesh_signature=mesh_signature, term=term, payloads=[payload])


def test_persist_processed_claims_rejects_drug_entry_missing_name(session) -> None:
    mesh_signature = "king-denborough|anesthesia"
    term, _ = _seed_search_term(session, mesh_signature)
    article, snippet_one, _ = _seed_article_with_snippets(session, term, pmid="20202025")

    payload = _single_claim_payload(article=article, snippet=snippet_one)
    payload["drugs"][0].pop("name")

    with pytest.raises(
        InvalidClaimPayload,
        match="drugs entries must include non-empty 'name'",
    ):
        _persist_payloads(session, mesh_signature=mesh_signature, term=term, payloads=[payload])


def test_persist_processed_claims_rejects_drug_entry_empty_name(session) -> None:
    mesh_signature = "king-denborough|anesthesia"
    term, _ = _seed_search_term(session, mesh_signature)
    article, snippet_one, _ = _seed_article_with_snippets(session, term, pmid="20202026")

    payload = _single_claim_payload(article=article, snippet=snippet_one)
    payload["drugs"][0]["name"] = ""

    with pytest.raises(
        InvalidClaimPayload,
        match="drugs entries must include non-empty 'name'",
    ):
        _persist_payloads(session, mesh_signature=mesh_signature, term=term, payloads=[payload])


def test_persist_processed_claims_rejects_drug_entry_missing_claims(session) -> None:
    mesh_signature = "king-denborough|anesthesia"
    term, _ = _seed_search_term(session, mesh_signature)
    article, snippet_one, _ = _seed_article_with_snippets(session, term, pmid="20202027")

    payload = _single_claim_payload(article=article, snippet=snippet_one)
    payload["drugs"][0].pop("claims")

    with pytest.raises(
        InvalidClaimPayload,
        match="drugs entries must include 'claims' arrays",
    ):
        _persist_payloads(session, mesh_signature=mesh_signature, term=term, payloads=[payload])


def test_persist_processed_claims_rejects_drug_entry_scalar_claims(session) -> None:
    mesh_signature = "king-denborough|anesthesia"
    term, _ = _seed_search_term(session, mesh_signature)
    article, snippet_one, _ = _seed_article_with_snippets(session, term, pmid="20202028")

    payload = _single_claim_payload(article=article, snippet=snippet_one)
    payload["drugs"][0]["claims"] = "claim:succinylcholine-risk"

    with pytest.raises(
        InvalidClaimPayload,
        match="drugs claims must be an array of claim ids",
    ):
        _persist_payloads(session, mesh_signature=mesh_signature, term=term, payloads=[payload])


def test_persist_processed_claims_rejects_drug_entry_noncanonical_claim_id(session) -> None:
    mesh_signature = "king-denborough|anesthesia"
    term, _ = _seed_search_term(session, mesh_signature)
    article, snippet_one, _ = _seed_article_with_snippets(session, term, pmid="20202029")

    payload = _single_claim_payload(article=article, snippet=snippet_one)
    payload["drugs"][0]["claims"] = ["legacy-claim"]

    with pytest.raises(
        InvalidClaimPayload,
        match="drugs claims must contain canonical 'claim:' identifiers",
    ):
        _persist_payloads(session, mesh_signature=mesh_signature, term=term, payloads=[payload])


def test_persist_processed_claims_rejects_missing_top_level_drugs(session) -> None:
    mesh_signature = "king-denborough|anesthesia"
    term, _ = _seed_search_term(session, mesh_signature)
    article, snippet_one, _ = _seed_article_with_snippets(session, term, pmid="21212121")

    payload = _single_claim_payload(article=article, snippet=snippet_one)
    payload.pop("drugs")

    with pytest.raises(
        InvalidClaimPayload,
        match="LLM payload missing required field 'drugs'",
    ):
        _persist_payloads(session, mesh_signature=mesh_signature, term=term, payloads=[payload])


def test_persist_processed_claims_rejects_top_level_drugs_none(session) -> None:
    mesh_signature = "king-denborough|anesthesia"
    term, _ = _seed_search_term(session, mesh_signature)
    article, snippet_one, _ = _seed_article_with_snippets(session, term, pmid="21212122")

    payload = _single_claim_payload(article=article, snippet=snippet_one)
    payload["drugs"] = None

    with pytest.raises(
        InvalidClaimPayload,
        match="LLM payload missing required field 'drugs'",
    ):
        _persist_payloads(session, mesh_signature=mesh_signature, term=term, payloads=[payload])


def test_persist_processed_claims_rejects_none_payload(session) -> None:
    mesh_signature = "king-denborough|anesthesia"
    term, _ = _seed_search_term(session, mesh_signature)

    with pytest.raises(
        InvalidClaimPayload,
        match="LLM payload must be a JSON object",
    ):
        _persist_payloads(session, mesh_signature=mesh_signature, term=term, payloads=[None])


def test_persist_processed_claims_rejects_non_mapping_payload(session) -> None:
    mesh_signature = "king-denborough|anesthesia"
    term, _ = _seed_search_term(session, mesh_signature)

    with pytest.raises(
        InvalidClaimPayload,
        match="LLM payload must be a JSON object",
    ):
        _persist_payloads(session, mesh_signature=mesh_signature, term=term, payloads=[[1, 2, 3]])


class _ParsedJsonRaises:
    def parsed_json(self) -> dict:
        raise RuntimeError("boom")


class _ParsedJsonReturnsNonMapping:
    def parsed_json(self) -> list[str]:
        return ["not", "a", "dict"]


def test_persist_processed_claims_rejects_parsed_json_exception(session) -> None:
    mesh_signature = "king-denborough|anesthesia"
    term, _ = _seed_search_term(session, mesh_signature)

    with pytest.raises(
        InvalidClaimPayload,
        match="LLM payload parsed_json\(\) failed",
    ):
        _persist_payloads(
            session,
            mesh_signature=mesh_signature,
            term=term,
            payloads=[_ParsedJsonRaises()],
        )


def test_persist_processed_claims_rejects_parsed_json_non_mapping(session) -> None:
    mesh_signature = "king-denborough|anesthesia"
    term, _ = _seed_search_term(session, mesh_signature)

    with pytest.raises(
        InvalidClaimPayload,
        match="LLM payload parsed_json\(\) must return a JSON object",
    ):
        _persist_payloads(
            session,
            mesh_signature=mesh_signature,
            term=term,
            payloads=[_ParsedJsonReturnsNonMapping()],
        )


def test_persist_processed_claims_rejects_non_object_claim_entry(session) -> None:
    mesh_signature = "king-denborough|anesthesia"
    term, _ = _seed_search_term(session, mesh_signature)
    article, snippet_one, _ = _seed_article_with_snippets(session, term, pmid="30303030")

    payload = _single_claim_payload(article=article, snippet=snippet_one)
    payload["claims"] = ["not-a-claim"]

    with pytest.raises(
        InvalidClaimPayload,
        match="contains a non-object claim entry",
    ):
        _persist_payloads(session, mesh_signature=mesh_signature, term=term, payloads=[payload])


def test_persist_processed_claims_rejects_claim_missing_drugs_field(session) -> None:
    mesh_signature = "king-denborough|anesthesia"
    term, _ = _seed_search_term(session, mesh_signature)
    article, snippet_one, _ = _seed_article_with_snippets(session, term, pmid="31313131")

    payload = _single_claim_payload(article=article, snippet=snippet_one)
    payload["claims"][0].pop("drugs")

    with pytest.raises(
        InvalidClaimPayload,
        match="LLM claim payload missing required field 'drugs'",
    ):
        _persist_payloads(session, mesh_signature=mesh_signature, term=term, payloads=[payload])


def test_persist_processed_claims_rejects_claim_none_drugs_field(session) -> None:
    mesh_signature = "king-denborough|anesthesia"
    term, _ = _seed_search_term(session, mesh_signature)
    article, snippet_one, _ = _seed_article_with_snippets(session, term, pmid="31313132")

    payload = _single_claim_payload(article=article, snippet=snippet_one)
    payload["claims"][0]["drugs"] = None

    with pytest.raises(
        InvalidClaimPayload,
        match="LLM claim payload missing required field 'drugs'",
    ):
        _persist_payloads(session, mesh_signature=mesh_signature, term=term, payloads=[payload])


def test_persist_processed_claims_rejects_claim_with_no_resolved_drugs(session) -> None:
    mesh_signature = "king-denborough|anesthesia"
    term, _ = _seed_search_term(session, mesh_signature)
    article, snippet_one, _ = _seed_article_with_snippets(session, term, pmid="31313133")

    payload = _single_claim_payload(article=article, snippet=snippet_one)
    payload["claims"][0]["drugs"] = []

    with pytest.raises(
        InvalidClaimPayload,
        match="did not resolve any drug terms",
    ):
        _persist_payloads(session, mesh_signature=mesh_signature, term=term, payloads=[payload])


def test_persist_processed_claims_rejects_scalar_reaction_payload(session) -> None:
    mesh_signature = "king-denborough|anesthesia"
    term, _ = _seed_search_term(session, mesh_signature)
    article, snippet_one, _ = _seed_article_with_snippets(session, term, pmid="32323232")

    payload = _single_claim_payload(article=article, snippet=snippet_one)
    payload["claims"][0]["idiosyncratic_reaction"] = "oops"

    with pytest.raises(
        InvalidClaimPayload,
        match="idiosyncratic_reaction' must be an object",
    ):
        _persist_payloads(session, mesh_signature=mesh_signature, term=term, payloads=[payload])


def test_persist_processed_claims_rejects_claim_missing_articles_field(session) -> None:
    mesh_signature = "king-denborough|anesthesia"
    term, _ = _seed_search_term(session, mesh_signature)
    article, snippet_one, _ = _seed_article_with_snippets(session, term, pmid="33333333")

    payload = _single_claim_payload(article=article, snippet=snippet_one)
    payload["claims"][0].pop("articles")

    with pytest.raises(
        InvalidClaimPayload,
        match="LLM claim payload missing required field 'articles'",
    ):
        _persist_payloads(session, mesh_signature=mesh_signature, term=term, payloads=[payload])


def test_persist_processed_claims_rejects_claim_with_empty_articles_list(session) -> None:
    mesh_signature = "king-denborough|anesthesia"
    term, _ = _seed_search_term(session, mesh_signature)
    article, snippet_one, _ = _seed_article_with_snippets(session, term, pmid="33333334")

    payload = _single_claim_payload(article=article, snippet=snippet_one)
    payload["claims"][0]["articles"] = []

    with pytest.raises(
        InvalidClaimPayload,
        match="did not include any linked articles",
    ):
        _persist_payloads(session, mesh_signature=mesh_signature, term=term, payloads=[payload])