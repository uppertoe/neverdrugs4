from __future__ import annotations

import pytest

from app.models import (
    ArticleArtefact,
    ArticleSnippet,
    ProcessedClaim,
    ProcessedClaimDrugLink,
    ProcessedClaimSet,
    ProcessedClaimSetVersion,
    SearchArtefact,
    SearchTerm,
)
from app.services.processed_claims import (
    InvalidClaimPayload,
    _AggregatedClaim,
    _DrugCatalogEntry,
    _compute_claim_group_id,
    _normalise_claim_terms,
    persist_processed_claims,
)


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

    assert len(stored_set.versions) == 1
    active_version = stored_set.versions[0]
    assert active_version.status == "active"
    assert active_version.version_number == 1

    assert len(active_version.claims) == 1
    stored_claim = active_version.claims[0]
    assert stored_claim.claim_id == "claim:succinylcholine-risk-duplicate"
    assert stored_claim.classification == "risk"
    assert stored_claim.confidence == "high"
    assert stored_claim.summary == "Evidence from multiple reports indicates succinylcholine precipitates crises."
    assert stored_claim.drugs == ["Succinylcholine"]
    assert stored_claim.drug_classes == ["depolarising neuromuscular blocker"]
    assert stored_claim.severe_reaction_flag is True
    assert set(stored_claim.severe_reaction_terms) == {"anaphylaxis", "Cardiac arrest"}
    assert set(stored_claim.source_claim_ids) == {
        "claim:succinylcholine-risk",
        "claim:succinylcholine-risk-duplicate",
    }
    assert stored_claim.canonical_hash
    assert stored_claim.claim_group_id
    assert stored_claim.up_votes == 0
    assert stored_claim.down_votes == 0

    evidence_ids = {entry.snippet_id for entry in stored_claim.evidence}
    assert evidence_ids == {str(snippet_one.id), str(snippet_two.id)}
    for entry in stored_claim.evidence:
        assert entry.citation_url == article.citation["preferred_url"]

    links = session.query(ProcessedClaimDrugLink).order_by(ProcessedClaimDrugLink.id).all()
    terms = {(link.term, link.term_kind) for link in links}
    assert terms == {
        ("Succinylcholine", "drug"),
        ("depolarising neuromuscular blocker", "drug_class"),
    }
    assert {link.claim_id for link in links} == {stored_claim.id}


def test_persist_processed_claims_includes_article_only_reference_when_needed(session) -> None:
    mesh_signature = "king-denborough|anesthesia"
    term, _ = _seed_search_term(session, mesh_signature)
    article, snippet_one, _ = _seed_article_with_snippets(session, term, pmid="11111111")

    payload = _single_claim_payload(article=article, snippet=snippet_one)
    payload["claims"][0]["articles"].append("article:22222222")

    claim_set = _persist_payloads(
        session,
        mesh_signature=mesh_signature,
        term=term,
        payloads=[payload],
    )

    stored_set = session.get(ProcessedClaimSet, claim_set.id)
    assert stored_set is not None
    active_version = stored_set.get_active_version()
    assert active_version is not None
    stored_claim = active_version.claims[0]

    evidence_by_id = {entry.snippet_id: entry for entry in stored_claim.evidence}
    assert set(evidence_by_id) == {str(snippet_one.id), "article:22222222"}
    assert evidence_by_id[str(snippet_one.id)].citation_url == article.citation["preferred_url"]
    assert evidence_by_id["article:22222222"].citation_url == "https://pubmed.ncbi.nlm.nih.gov/22222222/"


def test_persist_processed_claims_creates_new_version_and_supersedes_previous(session) -> None:
    mesh_signature = "king-denborough|anesthesia"
    term, _ = _seed_search_term(session, mesh_signature)

    old_set = ProcessedClaimSet(
        mesh_signature=mesh_signature,
        condition_label="Old label",
        last_search_term_id=term.id,
    )
    old_version = ProcessedClaimSetVersion(
        claim_set=old_set,
        version_number=1,
        status="active",
        pipeline_metadata={"source": "test"},
    )
    old_claim = ProcessedClaim(
        claim_set=old_set,
        claim_set_version=old_version,
        claim_id="old",
        classification="risk",
        summary="Outdated",
        confidence="low",
        canonical_hash="hash-old",
        claim_group_id="group-old",
        drugs=["old"],
        drug_classes=[],
        source_claim_ids=["old"],
        severe_reaction_flag=False,
        severe_reaction_terms=[],
        up_votes=3,
        down_votes=1,
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

    assert {version.version_number for version in refreshed_set.versions} == {1, 2}
    active_version = next(version for version in refreshed_set.versions if version.status == "active")
    superseded_version = next(version for version in refreshed_set.versions if version.status == "superseded")

    assert active_version.version_number == 2
    assert active_version.claims[0].claim_id == "claim:new"
    assert superseded_version.version_number == 1
    assert superseded_version.claims[0].claim_id == "old"
    assert superseded_version.claims[0].up_votes == 3
    assert superseded_version.claims[0].down_votes == 1


def test_persist_processed_claims_carries_vote_totals(session) -> None:
    mesh_signature = "king-denborough|anesthesia"
    term, _ = _seed_search_term(session, mesh_signature)

    claim_set = ProcessedClaimSet(
        mesh_signature=mesh_signature,
        condition_label="Initial",
        last_search_term_id=term.id,
    )
    first_version = ProcessedClaimSetVersion(
        claim_set=claim_set,
        version_number=1,
        status="active",
        pipeline_metadata={"step": "seed"},
    )
    legacy_group_id = _compute_claim_group_id(
        _AggregatedClaim(
            claim_id="legacy-claim",
            classification="risk",
            summary="Legacy summary",
            confidence="medium",
            drugs=["Succinylcholine"],
            drug_classes=[
                "Depolarising neuromuscular blocker",
            ],
            severe_reaction_flag=True,
            severe_reaction_terms=["Malignant hyperthermia"],
        )
    )

    legacy_claim = ProcessedClaim(
        claim_set=claim_set,
        claim_set_version=first_version,
        claim_id="legacy-claim",
        classification="risk",
        summary="Legacy summary",
        confidence="medium",
        canonical_hash="legacy-hash",
        claim_group_id=legacy_group_id,
        drugs=["Succinylcholine"],
        drug_classes=[
            "Depolarising neuromuscular blocker",
        ],
        source_claim_ids=["legacy-claim"],
        severe_reaction_flag=True,
        severe_reaction_terms=["Malignant hyperthermia"],
        up_votes=5,
        down_votes=2,
    )
    session.add_all([claim_set, first_version, legacy_claim])
    session.flush()

    article, snippet_one, _ = _seed_article_with_snippets(session, term, pmid="77777777")
    payload = _single_claim_payload(
        article=article,
        snippet=snippet_one,
        summary="New perspective on succinylcholine risk.",
        descriptors=["Malignant hyperthermia"],
    )

    refreshed_set = _persist_payloads(
        session,
        mesh_signature=mesh_signature,
        term=term,
        payloads=[payload],
    )

    active_version = next(version for version in refreshed_set.versions if version.status == "active")
    superseded_version = next(version for version in refreshed_set.versions if version.status == "superseded")

    new_claim = active_version.claims[0]
    old_claim = superseded_version.claims[0]

    assert new_claim.claim_group_id == old_claim.claim_group_id
    assert new_claim.canonical_hash != old_claim.canonical_hash
    assert new_claim.up_votes == 5
    assert new_claim.down_votes == 2


def test_persist_processed_claims_warns_on_empty_drug_array(session) -> None:
    mesh_signature = "king-denborough|anesthesia"
    term, _ = _seed_search_term(session, mesh_signature)
    article, snippet_one, _ = _seed_article_with_snippets(session, term, pmid="33333333")

    payload = _single_claim_payload(article=article, snippet=snippet_one)
    payload["drugs"] = []

    claim_set = _persist_payloads(session, mesh_signature=mesh_signature, term=term, payloads=[payload])

    active_version = claim_set.get_active_version()
    assert active_version is not None
    warnings = active_version.pipeline_metadata.get("warnings", [])
    codes = {entry.get("code") for entry in warnings}
    assert "empty-drug-catalog" in codes
    assert "claim-unknown-drug-identifier" in codes


def test_persist_processed_claims_warns_when_drug_catalog_missing(session) -> None:
    mesh_signature = "king-denborough|anesthesia"
    term, _ = _seed_search_term(session, mesh_signature)
    article, snippet_one, _ = _seed_article_with_snippets(session, term, pmid="33333334")

    payload = _single_claim_payload(article=article, snippet=snippet_one)
    payload.pop("drugs")

    claim_set = _persist_payloads(
        session,
        mesh_signature=mesh_signature,
        term=term,
        payloads=[payload],
    )

    active_version = claim_set.get_active_version()
    assert active_version is not None
    stored_claim = active_version.claims[0]
    assert stored_claim.drugs == ["Succinylcholine"]

    metadata = active_version.pipeline_metadata
    warnings = metadata.get("warnings", [])
    codes = {entry.get("code") for entry in warnings}
    assert {"missing-drug-catalog", "claim-unknown-drug-identifier"}.issubset(codes)


def test_persist_processed_claims_rejects_missing_claim_id(session) -> None:
    mesh_signature = "king-denborough|anesthesia"
    term, _ = _seed_search_term(session, mesh_signature)
    article, snippet_one, _ = _seed_article_with_snippets(session, term, pmid="44444444")

    payload = _single_claim_payload(article=article, snippet=snippet_one)
    claim = payload["claims"][0]
    claim.pop("id")

    with pytest.raises(InvalidClaimPayload, match="missing required field 'id'"):
        _persist_payloads(session, mesh_signature=mesh_signature, term=term, payloads=[payload])


def test_normalise_claim_terms_drops_generic_neuromuscular_class() -> None:
    catalog = {
        "drug:succinylcholine": _DrugCatalogEntry(
            drug_id="drug:succinylcholine",
            name="Succinylcholine",
            classes=("depolarising neuromuscular blocker",),
        )
    }

    drugs, classes, canonical = _normalise_claim_terms(
        ["drug:succinylcholine"],
        ["neuromuscular blocker"],
        drug_catalog=catalog,
    )

    assert drugs == ["Succinylcholine"]
    assert classes == ["depolarising neuromuscular blocker"]
    assert canonical == ["drug:succinylcholine"]


def test_normalise_claim_terms_canonicalises_benzylisoquinolinium_label() -> None:
    catalog = {
        "drug:atracurium": _DrugCatalogEntry(
            drug_id="drug:atracurium",
            name="Atracurium",
            classes=(
                "neuromuscular blocker (non-depolarising)",
                "neuromuscular blocker (benzylisoquinolinium)",
            ),
        )
    }

    drugs, classes, canonical = _normalise_claim_terms(
        ["drug:atracurium"],
        ["benzylisoquinolinium neuromuscular blocker"],
        drug_catalog=catalog,
    )

    assert drugs == ["Atracurium"]
    assert classes == [
        "neuromuscular blocker (non-depolarising)",
        "neuromuscular blocker (benzylisoquinolinium)",
    ]
    assert canonical == ["drug:atracurium"]


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


def test_persist_processed_claims_rejects_noniterable_reaction_descriptors(session) -> None:
    mesh_signature = "king-denborough|anesthesia"
    term, _ = _seed_search_term(session, mesh_signature)
    article, snippet_one, _ = _seed_article_with_snippets(session, term, pmid="56565654")

    payload = _single_claim_payload(article=article, snippet=snippet_one)
    payload["claims"][0]["idiosyncratic_reaction"]["descriptors"] = 123

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


def test_persist_processed_claims_rejects_supporting_evidence_noniterable_key_points(session) -> None:
    mesh_signature = "king-denborough|anesthesia"
    term, _ = _seed_search_term(session, mesh_signature)
    article, snippet_one, _ = _seed_article_with_snippets(session, term, pmid="67676766")

    payload = _single_claim_payload(article=article, snippet=snippet_one)
    payload["claims"][0]["supporting_evidence"][0]["key_points"] = 123

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


def test_persist_processed_claims_rejects_noniterable_articles(session) -> None:
    mesh_signature = "king-denborough|anesthesia"
    term, _ = _seed_search_term(session, mesh_signature)
    article, snippet_one, _ = _seed_article_with_snippets(session, term, pmid="78787875")

    payload = _single_claim_payload(article=article, snippet=snippet_one)
    payload["claims"][0]["articles"] = 123

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


def test_persist_processed_claims_warns_on_unknown_drug_reference(session) -> None:
    mesh_signature = "king-denborough|anesthesia"
    term, _ = _seed_search_term(session, mesh_signature)
    article, snippet_one, _ = _seed_article_with_snippets(session, term, pmid="88888888")

    payload = _single_claim_payload(article=article, snippet=snippet_one)
    payload["claims"][0]["drugs"] = ["drug:unknown"]

    claim_set = _persist_payloads(session, mesh_signature=mesh_signature, term=term, payloads=[payload])

    active_version = claim_set.get_active_version()
    assert active_version is not None
    stored_claim = active_version.claims[0]
    assert stored_claim.drugs == ["Unknown"]

    codes = {entry.get("code") for entry in active_version.pipeline_metadata.get("warnings", [])}
    assert "claim-unknown-drug-identifier" in codes


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


def test_persist_processed_claims_warns_when_claim_missing_declaration(session) -> None:
    mesh_signature = "king-denborough|anesthesia"
    term, _ = _seed_search_term(session, mesh_signature)
    article, snippet_one, _ = _seed_article_with_snippets(session, term, pmid="99999999")

    payload = _single_claim_payload(article=article, snippet=snippet_one)
    payload["drugs"][0]["claims"] = []

    result = _persist_payloads(session, mesh_signature=mesh_signature, term=term, payloads=[payload])

    active_version = next(version for version in result.versions if version.status == "active")
    warnings = active_version.pipeline_metadata.get("schema_warnings", [])
    claim_id = payload["claims"][0]["id"]

    assert any(warning["code"] == "claim-missing-drug-declaration" and warning["claim_id"] == claim_id for warning in warnings)
    assert any(
        warning["code"] == "claim-missing-drug-reference"
        and warning["claim_id"] == claim_id
        and warning["drug_id"] == payload["drugs"][0]["id"]
        for warning in warnings
    )
    assert len(active_version.claims) == 1


def test_persist_processed_claims_warns_on_dangling_declared_claim(session) -> None:
    mesh_signature = "king-denborough|anesthesia"
    term, _ = _seed_search_term(session, mesh_signature)
    article, snippet_one, _ = _seed_article_with_snippets(session, term, pmid="10101010")

    payload = _single_claim_payload(article=article, snippet=snippet_one)
    payload["claims"] = []

    result = _persist_payloads(session, mesh_signature=mesh_signature, term=term, payloads=[payload])

    active_version = next(version for version in result.versions if version.status == "active")
    warnings = active_version.pipeline_metadata.get("schema_warnings", [])
    declared_claim_id = payload["drugs"][0]["claims"][0]

    assert any(
        warning["code"] == "orphaned-drug-claim-declaration"
        and warning["claim_id"] == declared_claim_id
        for warning in warnings
    )
    assert len(active_version.claims) == 0


def test_persist_processed_claims_warns_when_claim_references_undeclared_drug(session) -> None:
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

    result = _persist_payloads(session, mesh_signature=mesh_signature, term=term, payloads=[payload])

    active_version = next(version for version in result.versions if version.status == "active")
    warnings = active_version.pipeline_metadata.get("schema_warnings", [])
    claim_id = payload["claims"][0]["id"]

    assert any(
        warning["code"] == "claim-missing-drug-reference"
        and warning["claim_id"] == claim_id
        and warning["drug_id"] == "drug:sevoflurane"
        for warning in warnings
    )
    persisted_claim = active_version.claims[0]
    assert set(persisted_claim.drugs) == {"Succinylcholine", "Sevoflurane"}


def test_persist_processed_claims_warns_on_scalar_drug_catalog_entries(session) -> None:
    mesh_signature = "king-denborough|anesthesia"
    term, _ = _seed_search_term(session, mesh_signature)
    article, snippet_one, _ = _seed_article_with_snippets(session, term, pmid="20202021")

    payload = _single_claim_payload(article=article, snippet=snippet_one)
    payload["drugs"] = "not-a-list"

    claim_set = _persist_payloads(session, mesh_signature=mesh_signature, term=term, payloads=[payload])

    active_version = claim_set.get_active_version()
    assert active_version is not None
    warnings = active_version.pipeline_metadata.get("warnings", [])
    warning_entries = [entry for entry in warnings if entry.get("code") == "invalid-drug-catalog"]
    assert warning_entries
    assert any(entry.get("reason") == "non-array" for entry in warning_entries)


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


def test_persist_processed_claims_warns_when_top_level_drugs_missing(session) -> None:
    mesh_signature = "king-denborough|anesthesia"
    term, _ = _seed_search_term(session, mesh_signature)
    article, snippet_one, _ = _seed_article_with_snippets(session, term, pmid="21212121")

    payload = _single_claim_payload(article=article, snippet=snippet_one)
    payload.pop("drugs")

    claim_set = _persist_payloads(session, mesh_signature=mesh_signature, term=term, payloads=[payload])

    active_version = claim_set.get_active_version()
    assert active_version is not None
    codes = {entry.get("code") for entry in active_version.pipeline_metadata.get("warnings", [])}
    assert "missing-drug-catalog" in codes


def test_persist_processed_claims_warns_when_top_level_drugs_none(session) -> None:
    mesh_signature = "king-denborough|anesthesia"
    term, _ = _seed_search_term(session, mesh_signature)
    article, snippet_one, _ = _seed_article_with_snippets(session, term, pmid="21212122")

    payload = _single_claim_payload(article=article, snippet=snippet_one)
    payload["drugs"] = None

    claim_set = _persist_payloads(session, mesh_signature=mesh_signature, term=term, payloads=[payload])

    active_version = claim_set.get_active_version()
    assert active_version is not None
    codes = {entry.get("code") for entry in active_version.pipeline_metadata.get("warnings", [])}
    assert "missing-drug-catalog" in codes


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
        match=r"LLM payload parsed_json\(\) failed",
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
        match=r"LLM payload parsed_json\(\) must return a JSON object",
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


def test_persist_processed_claims_rejects_missing_reaction_payload(session) -> None:
    mesh_signature = "king-denborough|anesthesia"
    term, _ = _seed_search_term(session, mesh_signature)
    article, snippet_one, _ = _seed_article_with_snippets(session, term, pmid="32323233")

    payload = _single_claim_payload(article=article, snippet=snippet_one)
    payload["claims"][0]["idiosyncratic_reaction"] = None

    with pytest.raises(
        InvalidClaimPayload,
        match="LLM claim payload missing required field 'idiosyncratic_reaction'",
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