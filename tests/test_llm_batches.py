from __future__ import annotations

from datetime import datetime, timezone

from app.models import ArticleArtefact, ArticleSnippet, SearchTerm
from app.services.claims import ClaimEvidenceGroup
from app.services.llm_batches import (
    LLMRequestBatch,
    SnippetLLMEntry,
    _render_user_prompt,
    build_llm_batches,
)


def test_render_user_prompt_downranks_generic_groups() -> None:
    snippet = SnippetLLMEntry(
        pmid="pmid-1",
        snippet_id=1,
        drug="muscle relaxants",
        classification="risk",
        snippet_text="Muscle relaxants were mentioned in a case series of malignant hyperthermia.",
        snippet_score=2.5,
        cues=["risk"],
        article_rank=1,
        article_score=2.0,
        citation_url="url",
        article_title="title",
        content_source="pubmed",
        token_estimate=120,
    )

    group = ClaimEvidenceGroup[
        SnippetLLMEntry
    ](
        group_key="risk:neuromuscular-blockers",
        classification="risk",
        drug_label="neuromuscular blocking agents",
        drug_terms=("muscle relaxants",),
        drug_classes=("neuromuscular blocking agent", "generic-class"),
        snippets=[snippet],
        top_score=2.5,
    )

    prompt = _render_user_prompt(
        condition_label="King Denborough syndrome",
        mesh_terms=["King Denborough syndrome"],
        snippets=[snippet],
        claim_groups=[group],
    )

    assert "muscle relaxants" in prompt
    assert "overly broad" in prompt
    assert "uncertain:dantrolene" not in prompt
    assert "Checklist" not in prompt


def _make_search_term(session, canonical: str = "condition") -> SearchTerm:
    term = SearchTerm(canonical=canonical, created_at=datetime.now(timezone.utc))
    session.add(term)
    session.flush()
    return term


def _add_article(session, term: SearchTerm, *, pmid: str, rank: int, score: float, title: str) -> ArticleArtefact:
    artefact = ArticleArtefact(
        search_term_id=term.id,
        pmid=pmid,
        rank=rank,
        score=score,
        citation={
            "pmid": pmid,
            "preferred_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            "title": title,
        },
    )
    session.add(artefact)
    session.flush()
    return artefact


def _add_snippet(
    session,
    artefact: ArticleArtefact,
    *,
    drug: str,
    classification: str,
    snippet_text: str,
    snippet_score: float,
    cues: list[str],
    hash_suffix: str,
) -> ArticleSnippet:
    snippet = ArticleSnippet(
        article_artefact_id=artefact.id,
        snippet_hash=f"{artefact.pmid}-{drug}-{hash_suffix}",
        drug=drug,
        classification=classification,
        snippet_text=snippet_text,
        snippet_score=snippet_score,
        cues=cues,
    )
    session.add(snippet)
    session.flush()
    return snippet


def test_build_llm_batches_respects_token_budget_and_orders_snippets(session) -> None:
    term = _make_search_term(session)
    artefact_one = _add_article(
        session,
        term,
        pmid="11111111",
        rank=1,
        score=4.2,
        title="Risks associated with succinylcholine",
    )
    artefact_two = _add_article(
        session,
        term,
        pmid="22222222",
        rank=2,
        score=3.8,
        title="Propofol case series",
    )

    long_sentence = (
        "This detailed snippet underscores repeated clinical observations about "
        "perioperative outcomes in patients living with neuromuscular disorders."
    )
    _add_snippet(
        session,
        artefact_one,
        drug="succinylcholine",
        classification="risk",
        snippet_text=(long_sentence + " Avoid use due to malignant hyperthermia risk.") * 2,
        snippet_score=5.7,
        cues=["avoid", "malignant hyperthermia"],
        hash_suffix="s1",
    )
    _add_snippet(
        session,
        artefact_two,
        drug="propofol",
        classification="safety",
        snippet_text=(long_sentence + " Propofol infusion was well tolerated without complications.") * 2,
        snippet_score=4.3,
        cues=["well tolerated", "no complications"],
        hash_suffix="s2",
    )
    _add_snippet(
        session,
        artefact_two,
        drug="sevoflurane",
        classification="risk",
        snippet_text=(long_sentence + " Sevoflurane may potentiate malignant hyperthermia concerns.") * 2,
        snippet_score=3.1,
        cues=["malignant hyperthermia"],
        hash_suffix="s3",
    )

    batches = build_llm_batches(
        session,
        search_term_id=term.id,
        condition_label="Duchenne muscular dystrophy",
        mesh_terms=["Duchenne Muscular Dystrophy", "Neuromuscular Disorders"],
        max_prompt_tokens=420,
        max_snippets_per_batch=2,
    )

    assert 2 <= len(batches) <= 3
    assert all(isinstance(batch, LLMRequestBatch) for batch in batches)

    first_batch = batches[0]
    second_batch = batches[1]
    assert first_batch.snippets[0].drug == "succinylcholine"
    assert first_batch.snippets[0].snippet_score >= first_batch.snippets[-1].snippet_score
    assert second_batch.snippets[0].drug in {"propofol", "sevoflurane"}

    for batch in batches:
        if batch.token_estimate > 420:
            assert len(batch.snippets) == 1
        else:
            assert batch.token_estimate <= 420

    for batch in batches:
        assert batch.messages[0]["role"] == "system"
        assert batch.messages[1]["role"] == "user"
        user_content = batch.messages[1]["content"]
        assert "Return only JSON" in user_content
        assert '"claims"' in user_content
        assert '"supporting_evidence"' in user_content
        # Ensure snippet metadata is present
        for snippet in batch.snippets:
            assert snippet.citation_url.startswith("https://pubmed")
            assert snippet.token_estimate >= 1
            assert snippet.drug.lower() in user_content.lower()
        assert "muscular dystrophy patients" not in batch.messages[0]["content"].lower()
        # Ensure article titles are exposed to the LLM
        for snippet in batch.snippets:
            assert snippet.article_title is not None
            assert snippet.article_title in user_content
        # Verify claim grouping metadata attached to the batch
        assert batch.claim_groups, "Expected at least one claim group per batch"
        assert all(group.snippets for group in batch.claim_groups)
        assert any("claim_group_id" in line for line in user_content.splitlines())

    # ensure grouping merges volatile anesthetics label when available across batches
    assert any(
        group.drug_label == "volatile anesthetics"
        for batch in batches
        for group in batch.claim_groups
    )


def test_build_llm_batches_skips_generic_snippets_when_specific_available(session) -> None:
    term = _make_search_term(session)
    generic_article = _add_article(
        session,
        term,
        pmid="33333333",
        rank=5,
        score=1.0,
        title="Overview of congenital myopathies",
    )
    specific_article = _add_article(
        session,
        term,
        pmid="44444444",
        rank=1,
        score=4.5,
        title="Succinylcholine safety update",
    )

    _add_snippet(
        session,
        generic_article,
        drug="neuromuscular blocking agents",
        classification="risk",
        snippet_text="Broad statement noting neuromuscular blocking agents in congenital myopathies.",
        snippet_score=2.0,
        cues=["condition-match"],
        hash_suffix="generic",
    )
    _add_snippet(
        session,
        specific_article,
        drug="succinylcholine",
        classification="risk",
        snippet_text="Succinylcholine precipitated malignant hyperthermia in central core disease.",
        snippet_score=4.8,
        cues=["malignant hyperthermia"],
        hash_suffix="specific",
    )

    batches = build_llm_batches(
        session,
        search_term_id=term.id,
        condition_label="Central core disease",
        mesh_terms=["Central core disease"],
        max_prompt_tokens=600,
        max_snippets_per_batch=5,
    )

    assert len(batches) == 1
    batch = batches[0]
    assert all("neuromuscular blocking agents" not in s.drug.lower() for s in batch.snippets)
    user_prompt = batch.messages[1]["content"]
    assert "neuromuscular blocking agents" not in user_prompt
    assert "overly broad" not in user_prompt
    assert "succinylcholine" in user_prompt


def test_build_llm_batches_uses_single_batch_when_under_budget(session) -> None:
    term = _make_search_term(session)
    artefact = _add_article(
        session,
        term,
        pmid="55555555",
        rank=1,
        score=4.0,
        title="Therapy options",
    )

    _add_snippet(
        session,
        artefact,
        drug="dantrolene",
        classification="uncertain",
        snippet_text="Dantrolene may offer benefit but evidence remains limited.",
        snippet_score=2.2,
        cues=["therapy"],
        hash_suffix="therapy",
    )
    _add_snippet(
        session,
        artefact,
        drug="ketamine",
        classification="safety",
        snippet_text="Ketamine was used without complications in a small series.",
        snippet_score=1.9,
        cues=["well tolerated"],
        hash_suffix="safety",
    )

    batches = build_llm_batches(
        session,
        search_term_id=term.id,
        condition_label="Central core disease",
        mesh_terms=["Central core disease"],
        max_prompt_tokens=800,
        max_snippets_per_batch=1,
    )

    assert len(batches) == 1
    assert len(batches[0].snippets) == 2
    assert batches[0].token_estimate <= 800


def test_build_llm_batches_interleaves_safety_snippets(session) -> None:
    term = _make_search_term(session)
    artefact = _add_article(
        session,
        term,
        pmid="77777777",
        rank=1,
        score=5.0,
        title="Mixed outcomes",
    )

    risk_text = "Risk evidence snippet."
    safety_text = "Safety evidence snippet."

    _add_snippet(
        session,
        artefact,
        drug="succinylcholine",
        classification="risk",
        snippet_text=risk_text,
        snippet_score=5.5,
        cues=["risk"],
        hash_suffix="risk1",
    )
    _add_snippet(
        session,
        artefact,
        drug="propofol",
        classification="safety",
        snippet_text=safety_text,
        snippet_score=4.9,
        cues=["safe"],
        hash_suffix="safety1",
    )
    _add_snippet(
        session,
        artefact,
        drug="sevoflurane",
        classification="risk",
        snippet_text=risk_text,
        snippet_score=4.7,
        cues=["risk"],
        hash_suffix="risk2",
    )

    batches = build_llm_batches(
        session,
        search_term_id=term.id,
        condition_label="Central core disease",
        mesh_terms=["Central core disease"],
        max_prompt_tokens=12000,
        max_snippets_per_batch=10,
    )

    assert batches, "Expected at least one batch"
    first_batch = batches[0]
    classifications = [snippet.classification for snippet in first_batch.snippets[:3]]
    assert classifications[0] == "risk"
    assert "safety" in classifications[:2], "Safety snippets should appear alongside early risk snippets"

def test_build_llm_batches_returns_empty_when_no_snippets(session) -> None:
    term = _make_search_term(session)
    _add_article(
        session,
        term,
        pmid="33333333",
        rank=1,
        score=2.4,
        title="Empty article",
    )

    batches = build_llm_batches(
        session,
        search_term_id=term.id,
        condition_label="Sample",
        mesh_terms=[],
    )

    assert batches == []