from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from sqlalchemy.orm import Session, selectinload

try:
    import tiktoken
except ImportError:  # pragma: no cover - optional dependency for token counting
    tiktoken = None

from app.models import ArticleArtefact
from app.services.claims import group_snippets_for_claims
from app.services.drug_classes import resolve_drug_group

_DEFAULT_SYSTEM_PROMPT = (
    "You are a clinical evidence synthesis assistant. Analyse the supplied snippets and claim groups before constructing cross-drug findings. "
    "When assessing each claim, draft a concise internal checklist (3-7 bullet points) covering the verification steps you will perform, then silently execute it. "
    "Focus on causality and therapeutic intent, cite only the provided article_ids, and keep reasoning grounded in the text. "
    "Reject any conclusions where the risk or advisory is inherent to the drug and unrelated to the target condition, and call out gaps when the evidence does not tie the effect to that condition. "
    "After each classification, briefly validate your decision against the cited evidence; confirm the condition link, self-correct if the validation fails, but do not include the checklist or validation in the JSON output. "
    "Identify idiosyncratic reactions explicitly when the evidence supports them, and do not infer facts beyond those presented."
)
_SNIPPET_METADATA_TOKENS = 18
_SNIPPET_TOKEN_MULTIPLIER = 1.15
DEFAULT_MAX_PROMPT_TOKENS = 20_000
DEFAULT_MAX_SNIPPETS_PER_BATCH = 64
_DEFAULT_TOKENS_MODEL = "gpt-5-mini"

_SAMPLE_RESPONSE_JSON = (
    "{\n"
    "  \"condition\": \"Central core disease\",\n"
    "  \"drugs\": [\n"
    "    {\"id\": \"drug:succinylcholine\", \"name\": \"Succinylcholine\", \"classifications\": [\"neuromuscular blocker\"], \"atc_codes\": [\"M03AB01\"], \"claims\": [\"claim:malignant-hyperthermia\"]},\n"
    "    {\"id\": \"drug:volatile-anaesthetics\", \"name\": \"Volatile anaesthetics\", \"classifications\": [\"inhalational anesthetic\"], \"atc_codes\": [\"N01AB\"], \"claims\": [\"claim:malignant-hyperthermia\"]}\n"
    "  ],\n"
    "  \"claims\": [{\"id\": \"claim:malignant-hyperthermia\", \"type\": \"risk\", \"summary\": \"Malignant hyperthermia recurred after depolarising neuromuscular blockers in susceptible patients.\", \"confidence\": \"high\", \"idiosyncratic_reaction\": {\"flag\": true, \"descriptors\": [\"malignant hyperthermia\"]}, \"articles\": [\"article:33190635\"], \"drugs\": [\"drug:succinylcholine\", \"drug:volatile-anaesthetics\"], \"supporting_evidence\": [{\"snippet_id\": \"snippet:33190635-1\", \"pmid\": \"33190635\", \"article_title\": \"Ryanodine receptor 1-related disorders: an historical perspective and proposal for a unified nomenclature.\", \"key_points\": [\"Volatile anesthetics and depolarising neuromuscular blockers triggered malignant hyperthermia events in susceptible patients.\"], \"notes\": \"\"}]}]\n"
    "}\n"
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
    severe_reaction_flag: bool
    severe_reaction_terms: list[str]


@dataclass(slots=True)
class LLMRequestBatch:
    messages: list[dict[str, str]]
    snippets: list[SnippetLLMEntry]
    token_estimate: int


@dataclass(slots=True)
class _PreparedBatch:
    snippets: list[SnippetLLMEntry]
    messages: list[dict[str, str]]
    token_count: int


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


def _extract_severe_reaction_from_tags(tags: Iterable[object]) -> tuple[bool, list[str]]:
    if not tags:
        return False, []

    terms: list[str] = []
    for tag in tags:
        if tag is None:
            continue
        if isinstance(tag, dict):
            if tag.get("kind") != "severe_reaction":
                continue
            label = str(tag.get("label") or "").strip()
        else:
            if getattr(tag, "kind", None) != "severe_reaction":
                continue
            label = str(getattr(tag, "label", "")).strip()
        if label:
            terms.append(label)

    if not terms:
        return False, []

    unique_terms = list(dict.fromkeys(terms))
    return True, unique_terms


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
            severe_flag, severe_terms = _extract_severe_reaction_from_tags(snippet.tags or [])
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
                    severe_reaction_flag=severe_flag,
                    severe_reaction_terms=severe_terms,
                )
            )

    entries.sort(key=lambda e: (-e.snippet_score, e.article_rank, e.drug.lower()))
    return entries

def _is_generic_term(term: str | None) -> bool:
    if not term:
        return True
    group = resolve_drug_group(term)
    if "generic-class" in getattr(group, "roles", ()):  # type: ignore[attr-defined]
        return True
    return False


def _normalise_specific_drug(term: str | None) -> str | None:
    if not term:
        return None
    cleaned = term.strip()
    if not cleaned:
        return None
    if _is_generic_term(cleaned):
        return None
    return cleaned


def _derive_prompt_context(
    snippets: Sequence[SnippetLLMEntry],
) -> tuple[dict[int, set[str]], list[str]]:
    related_drugs: dict[int, set[str]] = {}
    ordered_specific: list[str] = []
    seen_specific: set[str] = set()

    groups = group_snippets_for_claims(snippets)
    if not groups:
        return related_drugs, ordered_specific

    for group in groups:
        specific_terms: list[str] = []
        for term in group.drug_terms:
            specific = _normalise_specific_drug(term)
            if not specific:
                continue
            specific_terms.append(specific)
            lower = specific.lower()
            if lower not in seen_specific:
                ordered_specific.append(specific)
                seen_specific.add(lower)
        if not specific_terms:
            continue
        for snippet in group.snippets:
            related_drugs.setdefault(id(snippet), set()).update(specific_terms)

    return related_drugs, ordered_specific


def _render_user_prompt(
    *,
    condition_label: str,
    mesh_terms: Sequence[str] | None,
    snippets: list[SnippetLLMEntry],
    related_drug_map: Mapping[int, set[str]],
    drugs_in_scope: Sequence[str],
) -> str:
    mesh_line = ", ".join(mesh_terms) if mesh_terms else "None provided"
    schema_block = (
        '"condition": string\n'
        '"drugs": [{"id": "drug:identifier", "name": string, "classifications": [string], "atc_codes": [string], "claims": ["claim:identifier"]}]\n'
        '"claims": [{"id": "claim:identifier", "type": "risk | safety | uncertain | nuanced", "summary": string, "confidence": "low | medium | high", "idiosyncratic_reaction": {"flag": true | false, "descriptors": [string]}, "articles": ["article:identifier"], "drugs": ["drug:identifier"], "supporting_evidence": [{"snippet_id": string, "pmid": string, "article_title": string, "key_points": [string], "notes": string}]}]\n'
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
        article_id = f"article:{snippet.pmid}"
        related_terms = set(related_drug_map.get(id(snippet), set()))
        primary_term = _normalise_specific_drug(snippet.drug)
        if primary_term:
            related_terms.add(primary_term)
        related_line = ", ".join(sorted(t for t in related_terms if t)) or "none"
        snippet_blocks.append(
            (
                f"{idx}. snippet_id: {snippet_identifier}\n"
                f"   article_id: {article_id}\n"
                f"   article_title: {title}\n"
                f"   classification: {snippet.classification}\n"
                f"   cues: {cues_line}\n"
                f"   related_drugs: {related_line}\n"
                f"   snippet: {snippet.snippet_text.replace('\n', ' ').strip()}"
            )
        )

    snippet_section = "\n".join(snippet_blocks)

    instruction_block = (
        "INSTRUCTIONS\n"
        "1. Review DRUGS IN SCOPE and emit a drug entry for each listed drug; copy the name exactly, keep the provided class labels, include our ATC codes (or add only when the text clearly justifies it), and list the claim_ids you produce for that drug (leave the list empty when evidence is insufficient). Never reference a drug in any claim unless you have supplied a matching entry in the drugs array. When handling neuromuscular blockers, prefer precise families (depolarising, aminosteroid, benzylisoquinolinium) instead of the generic class unless the evidence is indistinct.\n"
        "2. Build shared findings (risk | safety | uncertain | nuanced) using only the supplied evidence, keep summaries factual, cite every supporting article_id, and attach all relevant drug_ids—even when a snippet mentions several drugs.\n"
        "3. Suppress any claim where the cited snippets only describe baseline properties of a drug (for example, malignant hyperthermia susceptibility) without linking the risk or benefit to the target condition; those belong in the out-of-scope set, not the output JSON.\n"
        "4. Do not fabricate claims for drugs with insufficient evidence; it is acceptable for a drug to have zero claims.\n"
        "5. Use the article_id identifiers we provide—cite an article once even if multiple snippets support it, never invent IDs, and flag unresolved citations in the claim summary if an article_id is missing.\n"
        "6. Detect idiosyncratic reactions yourself; when evidence shows unexpected patient-specific reactions (e.g., malignant hyperthermia), set idiosyncratic_reaction.flag true and copy the descriptors, otherwise leave the flag false with an empty list.\n"
        "7. Populate the supporting_evidence list for each claim using snippet_id values from the listing (include pmid, article_title, key_points, and optional notes; leave the array empty when evidence is too weak).\n"
        "8. Return JSON that matches the schema exactly; omit commentary.\n\n"
    )

    context_block = (
        "CONTEXT\n"
        f"Condition: {condition_label} | Related terms: {mesh_line}\n"
        "Snippets sorted by snippet_score (desc) then article_rank (asc); use the overlapping drug mentions to connect evidence across claims.\n\n"
    )

    legend_block = (
        "REFERENCE\n"
        "- snippet_id: prompt-only reference (omit from JSON).\n"
        "- article_id: prefix 'article:' + pmid; use for citations and flag if missing.\n"
        "- related_drugs: union of all drugs tied to the snippet; ensure each appears in the output (even if labelled insufficient evidence).\n"
        "- cues/heuristics: NLP hints; verify before relying on them.\n\n"
    )

    drugs_in_scope_block = "DRUGS IN SCOPE\n"
    if drugs_in_scope:
        for idx, drug in enumerate(drugs_in_scope, start=1):
            drugs_in_scope_block += f"{idx}. {drug}\n"
    else:
        drugs_in_scope_block += "None identified\n"
    drugs_in_scope_block += "\n"

    return (
        context_block
        + legend_block
        + drugs_in_scope_block
        + instruction_block
        + "SCHEMA\n"
        + schema_block
        + "SAMPLE RESPONSE\n"
        + _SAMPLE_RESPONSE_JSON
        + "\n\n"
        + "Snippets (full listing):\n"
        + snippet_section
        + "\n\nReturn only JSON."
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
        return _prioritise_unique_drug_coverage(snippets)

    specific_groups = [group for group in groups if "generic-class" not in group.drug_classes]
    if not specific_groups:
        return _prioritise_unique_drug_coverage(snippets)
    allowed = {id(snippet) for group in specific_groups for snippet in group.snippets}
    filtered: list[SnippetLLMEntry] = []
    for snippet in snippets:
        classification = (getattr(snippet, "classification", "") or "").strip().lower()
        if id(snippet) in allowed or classification not in {"risk", "safety"}:
            filtered.append(snippet)
    return _prioritise_unique_drug_coverage(filtered)


def _prioritise_unique_drug_coverage(snippets: list[SnippetLLMEntry]) -> list[SnippetLLMEntry]:
    if not snippets:
        return []

    seen: set[str] = set()
    unique_first: list[SnippetLLMEntry] = []
    remainder: list[SnippetLLMEntry] = []

    for snippet in snippets:
        key: str | None = None
        primary = _normalise_specific_drug(getattr(snippet, "drug", None))
        if primary:
            key = primary.lower()
        else:
            raw = (getattr(snippet, "drug", "") or "").strip().lower()
            key = raw or None

        if key is None:
            remainder.append(snippet)
            continue

        if key in seen:
            remainder.append(snippet)
            continue

        seen.add(key)
        unique_first.append(snippet)

    unique_first.extend(remainder)
    return unique_first


def _count_message_tokens(
    messages: Sequence[dict[str, str]],
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

    related_map, drugs_in_scope = _derive_prompt_context(snippet_list)
    user_prompt = _render_user_prompt(
        condition_label=condition_label,
        mesh_terms=mesh_terms,
        snippets=snippet_list,
        related_drug_map=related_map,
        drugs_in_scope=drugs_in_scope,
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
    )


def _finalise_batch(prepared: _PreparedBatch) -> LLMRequestBatch:
    return LLMRequestBatch(
        messages=prepared.messages,
        snippets=list(prepared.snippets),
        token_estimate=prepared.token_count,
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