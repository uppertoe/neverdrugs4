from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from sqlalchemy.orm import Session, selectinload

from app.models import ArticleArtefact
from app.services.claims import ClaimEvidenceGroup, group_snippets_for_claims

_DEFAULT_SYSTEM_PROMPT = (
    "You are a clinical evidence synthesis assistant. "
    "Summarise whether article snippets describe safety or risk relationships between drugs and the referenced condition. "
    "Only infer what is explicitly supported by the snippets."
)
_PROMPT_OVERHEAD_TOKENS = 220
_SNIPPET_METADATA_TOKENS = 18
_SNIPPET_TOKEN_MULTIPLIER = 1.15
DEFAULT_MAX_PROMPT_TOKENS = 1800
DEFAULT_MAX_SNIPPETS_PER_BATCH = 8


@dataclass(slots=True)
class SnippetLLMEntry:
    pmid: str
    snippet_id: int
    drug: str
    classification: str
    snippet_text: str
    snippet_score: float
    cues: list[str]
    article_rank: int
    article_score: float
    citation_url: str
    article_title: str | None
    content_source: str | None
    token_estimate: int


@dataclass(slots=True)
class LLMRequestBatch:
    messages: list[dict[str, str]]
    snippets: list[SnippetLLMEntry]
    token_estimate: int
    claim_groups: list[ClaimEvidenceGroup[SnippetLLMEntry]] = field(default_factory=list)


def build_llm_batches(
    session: Session,
    *,
    search_term_id: int,
    condition_label: str,
    mesh_terms: Sequence[str] | None,
    max_prompt_tokens: int = DEFAULT_MAX_PROMPT_TOKENS,
    max_snippets_per_batch: int = DEFAULT_MAX_SNIPPETS_PER_BATCH,
    system_prompt: str | None = None,
) -> list[LLMRequestBatch]:
    """Collect snippets for a search term and prepare batched prompts for the LLM."""
    snippet_entries = _collect_snippet_entries(session, search_term_id=search_term_id)
    if not snippet_entries:
        return []

    prompt_tokens = max(1, max_prompt_tokens)
    snippets_per_batch = max(1, max_snippets_per_batch)
    system_content = system_prompt or _DEFAULT_SYSTEM_PROMPT

    batches: list[LLMRequestBatch] = []
    current_batch: list[SnippetLLMEntry] = []
    current_tokens = _PROMPT_OVERHEAD_TOKENS

    for entry in snippet_entries:
        entry_tokens = entry.token_estimate
        if current_batch and (
            len(current_batch) >= snippets_per_batch or current_tokens + entry_tokens > prompt_tokens
        ):
            batches.append(
                _build_batch(
                    system_content,
                    condition_label,
                    mesh_terms,
                    current_batch,
                    current_tokens,
                )
            )
            current_batch = []
            current_tokens = _PROMPT_OVERHEAD_TOKENS

        # If a single snippet is larger than the remaining budget, start a new batch with it.
        if not current_batch and entry_tokens + _PROMPT_OVERHEAD_TOKENS > prompt_tokens:
            batches.append(
                _build_batch(
                    system_content,
                    condition_label,
                    mesh_terms,
                    [entry],
                    _PROMPT_OVERHEAD_TOKENS + entry_tokens,
                )
            )
            continue

        current_batch.append(entry)
        current_tokens += entry_tokens

    if current_batch:
        batches.append(
            _build_batch(
                system_content,
                condition_label,
                mesh_terms,
                current_batch,
                current_tokens,
            )
        )

    return batches


def _collect_snippet_entries(session: Session, *, search_term_id: int) -> list[SnippetLLMEntry]:
    articles: Iterable[ArticleArtefact] = (
        session.query(ArticleArtefact)
        .options(selectinload(ArticleArtefact.snippets))
        .filter(ArticleArtefact.search_term_id == search_term_id)
        .order_by(ArticleArtefact.rank)
        .all()
    )

    entries: list[SnippetLLMEntry] = []
    for article in articles:
        citation = article.citation or {}
        citation_url = citation.get("preferred_url") or f"https://pubmed.ncbi.nlm.nih.gov/{article.pmid}/"
        article_title = citation.get("title")
        for snippet in sorted(article.snippets, key=lambda s: (-s.snippet_score, s.drug.lower())):
            token_estimate = _estimate_snippet_tokens(snippet.snippet_text)
            entries.append(
                SnippetLLMEntry(
                    pmid=article.pmid,
                    snippet_id=snippet.id,
                    drug=snippet.drug,
                    classification=snippet.classification,
                    snippet_text=snippet.snippet_text,
                    snippet_score=snippet.snippet_score,
                    cues=list(snippet.cues or []),
                    article_rank=article.rank,
                    article_score=article.score,
                    citation_url=citation_url,
                    article_title=article_title,
                    content_source=article.content_source,
                    token_estimate=token_estimate,
                )
            )

    entries.sort(key=lambda e: (-e.snippet_score, e.article_rank, e.drug.lower()))
    return entries


def _build_batch(
    system_prompt: str,
    condition_label: str,
    mesh_terms: Sequence[str] | None,
    snippets: list[SnippetLLMEntry],
    token_estimate: int,
) -> LLMRequestBatch:
    groups = group_snippets_for_claims(snippets)
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": _render_user_prompt(
                condition_label=condition_label,
                mesh_terms=mesh_terms,
                snippets=snippets,
                claim_groups=groups,
            ),
        },
    ]
    return LLMRequestBatch(
        messages=messages,
        snippets=list(snippets),
        token_estimate=token_estimate,
        claim_groups=groups,
    )


def _render_user_prompt(
    *,
    condition_label: str,
    mesh_terms: Sequence[str] | None,
    snippets: list[SnippetLLMEntry],
    claim_groups: Sequence[ClaimEvidenceGroup[SnippetLLMEntry]],
) -> str:
    mesh_line = ", ".join(mesh_terms) if mesh_terms else "None provided"
    schema_block = (
        "{\n"
        '  "condition": "string",\n'
        '  "claims": [\n'
        "    {\n"
        '      "claim_id": "string",\n'
        '      "classification": "risk | safety",\n'
        '      "drug_classes": ["string"],\n'
        '      "drugs": ["string"],\n'
        '      "summary": "string",\n'
        '      "confidence": "low | medium | high",\n'
        '      "supporting_evidence": [\n'
        "        {\n"
        '          "snippet_id": "string",\n'
        '          "pmid": "string",\n'
        '          "article_title": "string",\n'
        '          "key_points": ["string"],\n'
        '          "notes": "string"\n'
        "        }\n"
        "      ]\n"
        "    }\n"
        "  ]\n"
        "}\n"
    )

    snippet_identifiers: dict[int, str] = {}
    snippet_blocks: list[str] = []
    for idx, snippet in enumerate(snippets, start=1):
        cues_line = ", ".join(snippet.cues) if snippet.cues else "none"
        title = snippet.article_title or "(no title provided)"
        snippet_identifier = (
            str(snippet.snippet_id) if snippet.snippet_id is not None else f"{snippet.pmid}-s{idx}"
        )
        snippet_identifiers[id(snippet)] = snippet_identifier
        snippet_blocks.append(
            (
                f"{idx}. snippet_id: {snippet_identifier}\n"
                f"   article_pmid: {snippet.pmid}\n"
                f"   article_title: {title}\n"
                f"   drug: {snippet.drug}\n"
                f"   classification: {snippet.classification}\n"
                f"   score: {snippet.snippet_score:.2f}\n"
                f"   article_rank: {snippet.article_rank}\n"
                f"   source_url: {snippet.citation_url}\n"
                f"   cues: {cues_line}\n"
                f"   snippet: {snippet.snippet_text.replace('\n', ' ').strip()}"
            )
        )

    snippet_section = "\n".join(snippet_blocks)

    claim_blocks: list[str] = []
    for idx, group in enumerate(claim_groups, start=1):
        class_line = ", ".join(group.drug_classes) if group.drug_classes else "none"
        drugs_line = ", ".join(group.drug_terms)
        snippet_lines: list[str] = []
        for inner_idx, snippet in enumerate(group.snippets, start=1):
            snippet_identifier = snippet_identifiers.get(id(snippet))
            if snippet_identifier is None:
                snippet_identifier = (
                    str(snippet.snippet_id)
                    if snippet.snippet_id is not None
                    else f"{snippet.pmid}-g{idx}-s{inner_idx}"
                )
            cues_line = ", ".join(snippet.cues) if snippet.cues else "none"
            snippet_lines.append(
                (
                    f"      - snippet_id: {snippet_identifier}\n"
                    f"        pmid: {snippet.pmid}\n"
                    f"        article_title: {snippet.article_title or '(no title provided)'}\n"
                    f"        drug: {snippet.drug}\n"
                    f"        cues: {cues_line}\n"
                    f"        citation_url: {snippet.citation_url}\n"
                    f"        snippet_score: {snippet.snippet_score:.2f}\n"
                    f"        text: {snippet.snippet_text.replace('\n', ' ').strip()}"
                )
            )

        group_block = (
            f"{idx}. claim_group_id: {group.group_key}\n"
            f"   classification: {group.classification}\n"
            f"   drug_label: {group.drug_label}\n"
            f"   drug_classes: {class_line}\n"
            f"   drugs: {drugs_line}\n"
            f"   top_snippet_score: {group.top_score:.2f}\n"
            f"   supporting_snippets:\n"
            f"{'\n'.join(snippet_lines)}"
        )
        claim_blocks.append(group_block)

    claim_section = "\n".join(claim_blocks)

    return (
        f"Condition: {condition_label}\n"
        f"Related terms: {mesh_line}\n\n"
        "Respond with valid JSON following this schema:\n"
        f"{schema_block}\n"
        "Synthesise claims by weighing all snippets in each supporting group collectively.\n"
        "Each claim must reference the relevant claim_group_id, list the drug classes, and cite every supporting snippet id used.\n"
        "Confidence must be one of: low, medium, high.\n"
        "Return 'summary' as a short assertion about the condition and drugs.\n"
        "Use the provided snippet_id values exactly as written.\n"
        "Snippets (full listing):\n"
        f"{snippet_section}\n\n"
        "Claim groups (use these to organise the response):\n"
        f"{claim_section}\n\n"
        "Return only JSON."
    )


def _estimate_snippet_tokens(snippet_text: str) -> int:
    word_count = len(snippet_text.split())
    estimate = math.ceil(word_count * _SNIPPET_TOKEN_MULTIPLIER) + _SNIPPET_METADATA_TOKENS
    return max(estimate, 1)