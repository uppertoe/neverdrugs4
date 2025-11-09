from __future__ import annotations

import pytest

from sqlalchemy.exc import IntegrityError

from app.models import (
    ProcessedClaim,
    ProcessedClaimFeedback,
    ProcessedClaimSet,
    ProcessedClaimSetVersion,
)


def test_claim_versions_relationships(session):
    claim_set = ProcessedClaimSet(mesh_signature="sig", condition_label="Cond", slug="cond-sig")
    session.add(claim_set)
    session.flush()

    version = ProcessedClaimSetVersion(
        claim_set=claim_set,
        version_number=1,
        status="draft",
    )
    session.add(version)

    claim = ProcessedClaim(
        claim_set=claim_set,
        claim_set_version=version,
        claim_id="risk:drug",
        classification="risk",
        summary="Summary",
        confidence="high",
        canonical_hash="hash-1",
        claim_group_id="group-1",
        drugs=["drug"],
        drug_classes=["class"],
        source_claim_ids=["risk:drug"],
    )
    session.add(claim)
    session.flush()

    assert version.claims == [claim]
    assert claim_set.claims == [claim]
    assert claim.claim_set_version is version
    assert claim.claim_set is claim_set
    assert claim.up_votes == 0
    assert claim.down_votes == 0

    feedback = ProcessedClaimFeedback(
        claim=claim,
        client_token="token-1",
        vote="up",
    )
    session.add(feedback)
    session.flush()

    assert feedback.created_at is not None
    assert claim.feedback == [feedback]


def test_claim_unique_within_version(session):
    claim_set = ProcessedClaimSet(mesh_signature="sig", condition_label="Cond", slug="cond-sig")
    session.add(claim_set)
    session.flush()

    version = ProcessedClaimSetVersion(
        claim_set=claim_set,
        version_number=1,
        status="draft",
    )
    session.add(version)
    session.flush()

    claim = ProcessedClaim(
        claim_set=claim_set,
        claim_set_version=version,
        claim_id="risk:drug",
        classification="risk",
        summary="Summary",
        confidence="high",
        canonical_hash="hash-1",
        claim_group_id="group-1",
        drugs=["drug"],
        drug_classes=["class"],
        source_claim_ids=["risk:drug"],
    )
    session.add(claim)
    session.flush()

    duplicate = ProcessedClaim(
        claim_set=claim_set,
        claim_set_version=version,
        claim_id="risk:drug",
        classification="risk",
        summary="Another",
        confidence="high",
        canonical_hash="hash-1",
        claim_group_id="group-1",
        drugs=["drug"],
        drug_classes=["class"],
        source_claim_ids=["risk:drug"],
    )
    session.add(duplicate)

    with pytest.raises(IntegrityError):
        session.flush()


def test_feedback_unique_per_claim_token(session):
    claim_set = ProcessedClaimSet(mesh_signature="sig", condition_label="Cond", slug="cond-sig")
    session.add(claim_set)
    session.flush()

    version = ProcessedClaimSetVersion(
        claim_set=claim_set,
        version_number=1,
        status="draft",
    )
    session.add(version)
    session.flush()

    claim = ProcessedClaim(
        claim_set=claim_set,
        claim_set_version=version,
        claim_id="risk:drug",
        classification="risk",
        summary="Summary",
        confidence="high",
        canonical_hash="hash-1",
        claim_group_id="group-1",
        drugs=["drug"],
        drug_classes=["class"],
        source_claim_ids=["risk:drug"],
    )
    session.add(claim)
    session.flush()

    feedback = ProcessedClaimFeedback(
        claim=claim,
        client_token="token-1",
        vote="up",
    )
    session.add(feedback)
    session.flush()

    duplicate = ProcessedClaimFeedback(
        claim=claim,
        client_token="token-1",
        vote="down",
    )
    session.add(duplicate)

    with pytest.raises(IntegrityError):
        session.flush()
