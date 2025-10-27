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
    assert "broad drug category" in prompt


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

    assert len(batches) == 2
    assert all(isinstance(batch, LLMRequestBatch) for batch in batches)

    first_batch, second_batch = batches
    assert first_batch.snippets[0].drug == "succinylcholine"
    assert first_batch.snippets[0].snippet_score >= first_batch.snippets[-1].snippet_score
    assert first_batch.token_estimate <= 420
    assert second_batch.snippets[0].drug in {"propofol", "sevoflurane"}
    assert second_batch.token_estimate <= 420

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

    # ensure grouping merges volatile anesthetics label when available
    assert any(group.drug_label == "volatile anesthetics" for group in second_batch.claim_groups)

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