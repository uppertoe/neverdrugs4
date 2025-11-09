# LLM Payload Refactor Plan

## Guiding Principles
- Overriding goal: surface, for any medical condition, a drug-first evidence bundle that tells the LLM which therapies matter, how confident we are, and what idiosyncratic risks exist—grounded solely in cited articles/snippets so downstream agents can make safe automation decisions without inventing facts.
- Work test-first for every behaviour change. Add failing unit/integration/contract tests before implementation, then make them pass, then enforce coverage.
- Treat breaking changes as expected. Remove legacy structures immediately; do not add fallbacks, feature flags, or backward-compatible adapters.
- Focus user value on surfacing anaesthetic drugs used for a condition, highlighting evidence-supported safety or harm. Disregard dosing changes or expected pharmacologic effects unless they directly map to harm.
- Explicitly capture idiosyncratic/severe reactions. Regression tests must ensure these flags remain accurate.

- Extend article ingestion to pull structured metadata (article type, citation count, MeSH descriptors, anaesthesia indicators) whenever abstracts/full texts are retrieved.
- Implement NLP-powered scoring heuristics (spaCy or equivalent) that prioritise:
  - Number of distinct drug or drug-class mentions.
  - Presence of risk/safety terminology.
  - Anaesthesia context (from metadata, title, abstract, or MeSH tags).
- Expose scoring details in logging/telemetry for audit. Fail hard if no articles pass minimum thresholds; downstream must handle empty sets explicitly.

- Build on robust NLP pipelines (dependency parsing, NER) to power drug/effect detection.
- For each selected article, extract snippets that:
  - Explicitly mention target drugs/drug classes alongside safety or harm statements.
  - Provide concise context (1-2 sentences) sufficient to substantiate the claim.
  - Highlight anaesthetic context when present.
- Ensure at least one snippet per unique drug-claim pair; allow multiple when distinct evidence exists.
- Build fixture-based tests covering positive, ambiguous, and rejected snippets (including anaesthesia emphasis and idiosyncratic reactions).

## Article Tagging
- Tag each article during retrieval with heuristics-driven labels such as:
  - Article type (guideline, RCT, case report, review, etc.).
  - Citation count bucket (e.g., high/medium/low) derived from retrieved metadata.
  - Anaesthesia-focus indicator and other contextual cues (e.g., perioperative setting).
- Tagging logic must be deterministic; cover with unit tests using cached metadata fixtures.
- Include tags in downstream payloads to guide the LLM interpretation.

- Adopt a simple canonical drug list for v0.1 (name + optional class tags) seeded from current ingestion results.
- Resolve and attach WHO ATC classification tiers for every drug encountered in snippets (at least the anatomical + therapeutic levels) so the LLM always sees consistent class information.
- Keep the data model forward-compatible (e.g., `atc_codes` arrays) to support deeper ATC expansion without structural rewrites.
- Document lookup responsibilities and add tests ensuring we can map drugs/classes deterministically, including fixtures for representative ATC codes.

## JSON Schema Redesign
- Replace the legacy claim-centric payload with a drug-first schema. Draft structure:

```json
{
  "condition": "string",
  "drugs": [
    {
      "id": "string",
      "name": "Morphine",
      "classifications": ["opioid"],
      "atc_codes": [],
      "claims": ["claim-id-1", "claim-id-2"]
    }
  ],
  "claims": [
    {
      "id": "claim-id-1",
      "type": "safety|risk",
      "confidence": "low|medium|high",
      "summary": "string",
      "severe_reaction": {
        "flag": true,
        "terms": ["laryngospasm"]
      },
      "snippets": ["snippet-id-7"],
      "articles": ["article-id-3"]
    }
  ]
}
```

- Use IDs to model many-to-many relationships between drugs, claims, snippets, and articles.
- Severe reaction flags live within each claim, ensuring the LLM highlights idiosyncratic harms when present.
- Provide JSON Schema definitions and contract tests to lock the structure before implementation. (We intentionally omit a top-level `articles` index in the LLM payload to keep token usage tight; downstream systems can recover the set of cited articles by aggregating `claim.articles`.)

## LLM Prompt Strategy
- Update prompt builder to iterate over every drug entry, summarising associated claims and citing evidence per snippet/article.
- Ensure prompt tests assert that all drug IDs are referenced and that severe_reaction flags result in explicit callouts.

## Implementation Roadmap
1. **Schema & Contracts**: Define JSON Schema files, write contract tests, update fixtures, and delete legacy claim payloads.
2. **Article Retrieval**: Implement metadata-enriched fetch and scoring heuristics with unit/integration tests.
3. **Snippet Pipeline**: Refine snippet extraction and validation logic; add regression fixtures emphasising anaesthesia contexts and severe reactions.
4. **Tagging Layer**: Build tagging heuristics with deterministic tests; integrate metadata sources.
5. **LLM Prompt Rewrite**: Consume the new schema, update snapshot tests to ensure complete drug coverage.
6. **API Layer**: Update `/api/claims/resolve` (and dependent endpoints) to emit the new payload. Remove old fields immediately after new tests pass.

## Outstanding Clarifications
- Confirm availability and quality of citation counts in existing article ingestion pipeline.
- Validate that metadata retrieval latency remains acceptable once citation counts and article types are fetched; add benchmarks if necessary.
- Identify any downstream consumers beyond the LLM that require updates (e.g., dashboards) and schedule their adoption accordingly.
