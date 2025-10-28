from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from sqlalchemy.orm import Session, selectinload

try:
    import tiktoken
except ImportError:  # pragma: no cover - optional dependency for token counting
    tiktoken = None

from app.models import ArticleArtefact
from app.services.claims import ClaimEvidenceGroup, group_snippets_for_claims

_DEFAULT_SYSTEM_PROMPT = (
    "You are a clinical evidence synthesis assistant. Analyse the supplied snippets and claim groups before classifying each drug. "
    "When classifying a claim, begin with a concise checklist (3-7 bullet points) outlining the tasks you will perform. "
    "Focus on causality and therapeutic intent, cite only the provided snippet_ids, and keep reasoning grounded in the text. "
    "After each classification, briefly validate the decision in 1-2 lines against the cited snippets and proceed or self-correct if the validation fails, but do not include this checklist in the JSON schema output. "
    "Do not infer any new facts beyond those explicitly presented."
)
_SNIPPET_METADATA_TOKENS = 18
_SNIPPET_TOKEN_MULTIPLIER = 1.15
DEFAULT_MAX_PROMPT_TOKENS = 20_000
DEFAULT_MAX_SNIPPETS_PER_BATCH = 64
_DEFAULT_TOKENS_MODEL = "gpt-5-mini"

_SAMPLE_RESPONSE_JSON = (
    "{\n"
    "  \"condition\": \"King Denborough syndrome\",\n"
    "  \"claims\": [\n"
    "    {\n"
    "      \"claim_id\": \"risk:succinylcholine\",\n"
    "      \"classification\": \"risk\",\n"
    "      \"drug_classes\": [\"depolarising neuromuscular blocker\"],\n"
    "      \"drugs\": [\"succinylcholine\"],\n"
    "      \"summary\": \"Succinylcholine repeatedly precipitated malignant hyperthermia in the cited reports.\",\n"
    "      \"confidence\": \"high\",\n"
    "      \"supporting_evidence\": [\n"
    "        {\n"
    "          \"snippet_id\": \"4\",\n"
    "          \"pmid\": \"33190635\",\n"
    "          \"article_title\": \"Ryanodine receptor 1-related disorders: an historical perspective and proposal for a unified nomenclature.\",\n"
    "          \"key_points\": [\n"
    "            \"Review links volatile anesthetics and depolarising muscle relaxants (succinylcholine) with malignant hyperthermia episodes.\",\n"
    "            \"Describes malignant hyperthermia features that occurred following succinylcholine exposure.\"\n"
    "          ],\n"
    "          \"notes\": \"Classified as risk because succinylcholine is named as the precipitating agent.\"\n"
    "        }\n"
    "      ]\n"
    "    }\n"
    "  ]\n"
    "}"
)


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


@dataclass(slots=True)
class _PreparedBatch:
    snippets: list[SnippetLLMEntry]
    messages: list[dict[str, str]]
    token_count: int
    claim_groups: list[ClaimEvidenceGroup[SnippetLLMEntry]]


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

    snippet_entries = _prioritise_snippet_entries(snippet_entries)
    if not snippet_entries:
        return []

    snippet_entries = _interleave_snippet_classes(snippet_entries)

    prompt_tokens = max(1, max_prompt_tokens)
    snippets_per_batch = max(1, max_snippets_per_batch)
    system_content = system_prompt or _DEFAULT_SYSTEM_PROMPT

    prepared_all = _prepare_batch(
        system_content,
        condition_label,
        mesh_terms,
        snippet_entries,
    )
    if prepared_all is not None and prepared_all.token_count <= prompt_tokens:
        return [_finalise_batch(prepared_all)]

    batches: list[LLMRequestBatch] = []
    current_batch: list[SnippetLLMEntry] = []
    current_prepared: _PreparedBatch | None = None

    for entry in snippet_entries:
        candidate_batch = current_batch + [entry]
        if len(candidate_batch) > snippets_per_batch:
            if current_prepared is not None:
                batches.append(_finalise_batch(current_prepared))
            current_batch = [entry]
            current_prepared = _prepare_batch(
                system_content,
                condition_label,
                mesh_terms,
                current_batch,
            )
            if current_prepared is None:
                current_batch = []
            elif current_prepared.token_count > prompt_tokens:
                batches.append(_finalise_batch(current_prepared))
                current_batch = []
                current_prepared = None
            continue

        candidate_prepared = _prepare_batch(
            system_content,
            condition_label,
            mesh_terms,
            candidate_batch,
        )

        if candidate_prepared is None:
            continue

        if candidate_prepared.token_count <= prompt_tokens:
            current_batch = candidate_batch
            current_prepared = candidate_prepared
            continue

        if current_prepared is not None:
            batches.append(_finalise_batch(current_prepared))
            current_batch = [entry]
            current_prepared = _prepare_batch(
                system_content,
                condition_label,
                mesh_terms,
                current_batch,
            )
            if current_prepared is None:
                current_batch = []
                continue
            if current_prepared.token_count > prompt_tokens:
                batches.append(_finalise_batch(current_prepared))
                current_batch = []
                current_prepared = None
            continue

        # No existing batch and the single snippet still exceeds the limit; send it alone.
        batches.append(_finalise_batch(candidate_prepared))
        current_batch = []
        current_prepared = None

    if current_prepared is not None:
        batches.append(_finalise_batch(current_prepared))

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
        '      "classification": "risk | safety | uncertain",\n'
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
    generic_groups: list[ClaimEvidenceGroup[SnippetLLMEntry]] = []
    for idx, group in enumerate(claim_groups, start=1):
        class_line = ", ".join(group.drug_classes) if group.drug_classes else "none"
        drugs_line = ", ".join(group.drug_terms)
        if "generic-class" in group.drug_classes:
            generic_groups.append(group)

        snippet_ids = []
        for inner_idx, snippet in enumerate(group.snippets, start=1):
            identifier = snippet_identifiers.get(id(snippet))
            if identifier is None:
                identifier = (
                    str(snippet.snippet_id)
                    if snippet.snippet_id is not None
                    else f"{snippet.pmid}-g{idx}-s{inner_idx}"
                )
            snippet_ids.append(identifier)

        supporting_line = ", ".join(snippet_ids) if snippet_ids else "none"
        group_block = (
            f"{idx}. claim_group_id: {group.group_key}\n"
            f"   classification: {group.classification}\n"
            f"   drug_label: {group.drug_label}\n"
            f"   drug_classes: {class_line}\n"
            f"   drugs: {drugs_line}\n"
            f"   top_snippet_score: {group.top_score:.2f}\n"
            f"   supporting_snippet_ids: {supporting_line}"
        )
        claim_blocks.append(group_block)

    claim_section = "\n".join(claim_blocks)

    caution_section = ""
    if generic_groups:
        caution_labels: list[str] = []
        for group in generic_groups:
            if group.drug_terms:
                term_list = ", ".join(group.drug_terms)
                caution_labels.append(f"{group.drug_label} ({term_list})")
            else:
                caution_labels.append(group.drug_label)
        joined_labels = "; ".join(caution_labels)
        caution_section = (
            "CAUTION\n"
            f"The following claim groups are overly broad drug categories; prioritise specific named drugs when evidence supports them: {joined_labels}.\n\n"
        )

    instruction_block = (
        "INSTRUCTIONS\n"
        "1. Classify each drug (risk, safety, uncertain) using causality from the snippets only.\n"
        "2. Merge supporting snippets per claim and cite every snippet_id you rely on.\n"
        "3. If evidence conflicts, create separate claims and explain the disagreement in the summary or notes.\n"
        "4. Keep key_points concise (1–2 sentences) and grounded in the cited text; do not invent new facts.\n"
        "5. Output valid JSON conforming to the schema.\n\n"
    )

    context_block = (
        "CONTEXT\n"
        f"Condition: {condition_label}\n"
        f"Related terms: {mesh_line}\n"
        "Snippets are ranked by snippet_score (higher is stronger) and article_rank (lower is better).\n"
        "Each claim_group bundles snippets for the same drug or drug class.\n\n"
    )

    legend_block = (
        "REFERENCE\n"
        "- snippet_id: stable identifier for citation.\n"
        "- article_rank: lower values indicate higher priority articles.\n"
        "- cues: keywords that triggered initial classification; verify them before relying on them.\n\n"
    )

    return (
        context_block
        + legend_block
        + instruction_block
        + "SCHEMA\n"
        + schema_block
        + "SAMPLE RESPONSE\n"
        + _SAMPLE_RESPONSE_JSON
        + "\n\n"
        + caution_section
        + "Snippets (full listing):\n"
        + snippet_section
        + "\n\nClaim groups (use these to organise the response):\n"
        + claim_section
        + "\nReturn only JSON."
    )


def _estimate_snippet_tokens(snippet_text: str) -> int:
    word_count = len(snippet_text.split())
    estimate = math.ceil(word_count * _SNIPPET_TOKEN_MULTIPLIER) + _SNIPPET_METADATA_TOKENS
    return max(estimate, 1)


def _prioritise_snippet_entries(snippets: list[SnippetLLMEntry]) -> list[SnippetLLMEntry]:
    if not snippets:
        return []

    groups = group_snippets_for_claims(snippets)
    if not groups:
        return snippets

    specific_groups = [group for group in groups if "generic-class" not in group.drug_classes]
    if not specific_groups:
        return snippets

    allowed = {id(snippet) for group in specific_groups for snippet in group.snippets}
    filtered: list[SnippetLLMEntry] = []
    for snippet in snippets:
        classification = getattr(snippet, "classification", "").strip().lower()
        if id(snippet) in allowed or classification not in {"risk", "safety"}:
            filtered.append(snippet)
    return filtered


def _count_message_tokens(
    messages: Sequence[dict[str, str]],
    *,
    model: str = _DEFAULT_TOKENS_MODEL,
) -> int:
    if not messages:
        return 0

    if tiktoken is None:  # pragma: no cover - fallback when tiktoken missing
        total_words = 0
        for message in messages:
            content = message.get("content", "")
            total_words += len(str(content).split())
        # Over-estimate by assuming ~1.6 tokens per word plus per-message overhead.
        return int(total_words * 1.6) + 6 * len(messages) + 10

    try:
        encoding = tiktoken.encoding_for_model(model)
    except Exception:  # pragma: no cover - fallback to default encoding
        encoding = tiktoken.get_encoding("cl100k_base")

    total_tokens = 0
    for message in messages:
        content = message.get("content", "")
        text = content if isinstance(content, str) else str(content)
        total_tokens += len(encoding.encode(text)) + 4  # per-message overhead
    total_tokens += 2  # reply priming
    return total_tokens


def _prepare_batch(
    system_prompt: str,
    condition_label: str,
    mesh_terms: Sequence[str] | None,
    snippets: Sequence[SnippetLLMEntry],
) -> _PreparedBatch | None:
    snippet_list = list(snippets)
    if not snippet_list:
        return None

    groups = group_snippets_for_claims(snippet_list)
    user_prompt = _render_user_prompt(
        condition_label=condition_label,
        mesh_terms=mesh_terms,
        snippets=snippet_list,
        claim_groups=groups,
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    token_count = _count_message_tokens(messages)
    return _PreparedBatch(
        snippets=snippet_list,
        messages=messages,
        token_count=token_count,
        claim_groups=groups,
    )


def _finalise_batch(prepared: _PreparedBatch) -> LLMRequestBatch:
    return LLMRequestBatch(
        messages=prepared.messages,
        snippets=list(prepared.snippets),
        token_estimate=prepared.token_count,
        claim_groups=list(prepared.claim_groups),
    )


def _interleave_snippet_classes(snippets: list[SnippetLLMEntry]) -> list[SnippetLLMEntry]:
    if not snippets:
        return []

    risk: list[SnippetLLMEntry] = []
    safety: list[SnippetLLMEntry] = []
    remainder: list[SnippetLLMEntry] = []

    for snippet in snippets:
        classification = (snippet.classification or "").strip().lower()
        if classification == "risk":
            risk.append(snippet)
        elif classification == "safety":
            safety.append(snippet)
        else:
            remainder.append(snippet)

    if not risk or not safety:
        return snippets

    result: list[SnippetLLMEntry] = []
    risk_index = 0
    safety_index = 0

    next_risk_score = risk[0].snippet_score if risk else float("-inf")
    next_safety_score = safety[0].snippet_score if safety else float("-inf")
    turn = "risk" if next_risk_score >= next_safety_score else "safety"

    while risk_index < len(risk) or safety_index < len(safety):
        if turn == "risk":
            if risk_index < len(risk):
                result.append(risk[risk_index])
                risk_index += 1
            turn = "safety"
        else:
            if safety_index < len(safety):
                result.append(safety[safety_index])
                safety_index += 1
            turn = "risk"

        if risk_index >= len(risk):
            turn = "safety"
        if safety_index >= len(safety):
            turn = "risk"

    if risk_index < len(risk):
        result.extend(risk[risk_index:])
    if safety_index < len(safety):
        result.extend(safety[safety_index:])

    result.extend(remainder)
    return result