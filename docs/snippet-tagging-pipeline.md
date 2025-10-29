# Snippet Tagging Pipeline (Draft Plan)

## Objectives
- Identify high-quality article snippets that describe drug-condition interactions.
- Attach rich, reusable tags that help the LLM reason about severity, certainty, mechanism, and context.
- Keep the tagging mechanism pluggable so we can swap between deterministic rules, spaCy-based NLP, or trained models without rewriting downstream logic.
- Score and prune snippets dynamically based on corpus volume and tag confidence so the LLM receives the most informative evidence.
- Maintain a TDD workflow for every component to ensure regressions are caught quickly.

## High-Level Flow
1. **Candidate Extraction**
   - Scan article text for condition + drug co-occurrences.
   - Use a pluggable candidate finder (initially regex + window; future: spaCy sentence spans or dependency-based extraction).
2. **Tagging**
   - Run each candidate through the `SnippetTagger` interface.
   - Baseline implementation combines curated dictionaries and spaCy linguistic features (lemma matching, dependency roles, negation detection).
   - Future implementations could employ ML/LLM models that honour the same interface.
3. **Scoring**
   - Assign base scores using article metadata (rank, citation count) plus tag-derived weights.
   - Terms/phrases defined in a central configuration determine boost factors (e.g., severity, high-confidence diagnostics).
   - spaCy lemmatisation ensures variants map to the same canonical terms.
4. **Pruning**
   - Deduplicate overlapping windows using scores + tag priority (retain the most informative tag mix).
   - Apply article-level quotas through `apply_article_quotas`, which reuses `compute_quota` and centralises trimming rules.
   - Dynamically adjust keep thresholds based on the number of candidates and tag diversity (e.g., ensure at least one severe snippet survives if any exist).
5. **Batch Preparation**
   - Surface tag summaries alongside cues in the LLM prompt (`cues: risk, severe-arrhythmia`).
   - Provide rationale fields so the LLM understands why a snippet was flagged.
6. **LLM Decision**
   - The LLM remains the ultimate arbiter of claim classification and severity flags, using tags as guidance rather than hard rules.

## Module Structure
- `snippet_candidates.py`: candidate finder abstraction and default regex implementation.
- `snippet_tagger.py`: defines `SnippetTagger` protocol, tag schema (`Tag(type, label, confidence, source)`), and spaCy-backed implementation.
- `snippet_scoring.py`: scoring heuristics based on article metadata + tag weights.
- `snippet_pruning.py`: shared dedupe/prioritisation logic (currently powered by `prune_window_overlaps` + `apply_article_quotas`).
- `snippet_postprocessors.py`: reusable post-processing primitives (e.g., enforcing classification coverage, capping per-drug counts).
- `snippet_pipeline.py`: orchestrates extraction, post-processing hooks, and quota application via `SnippetExtractionPipeline` + tunable `SnippetPipelineConfig`.
- `snippet_tuning.py`: helper utilities for sweeping pipeline configs (`grid_search_pipeline_configs`) and building quota grids for experimentation.
- `snippets.py`: orchestrator composing the stages and persisting results.

Each module should have dedicated unit tests plus integration tests covering the full pipeline on representative fixtures.

## Tagging Details
- Tag types: `risk`, `safety`, `severity`, `mechanism`, `population`, `negation`, etc.
- spaCy usage:
  - Lemmas and noun chunks for variant matching.
  - Dependency parse to confirm subject/action relationships (e.g., “drug causes seizure” vs “drug prevents seizure”).
  - Named entity recognition to capture organ systems or diagnostic terms.
- Tag sources are recorded (`rules`, `spaCy`, `ml-model`, `llm`) for auditing.
- Confidence defaults to 1.0 for deterministic rules, configurable for probabilistic models.

## Scoring & Pruning Strategy
- Start with existing article rank + snippet score logic, add tag-driven modifiers (e.g., +1.0 for `severity:idiosyncratic`).
- Introduce coverage checks: ensure at least one snippet per tag category when available.
- Allow configuration of maximum snippets per article while adapting to total volume (larger corpora impose stricter thresholds).

## Test Strategy (TDD)
1. **Unit tests** for each stage (candidate extraction, tagging, scoring, pruning) using small fixtures.
2. **Integration tests** that process sample articles end-to-end and assert tag propagation into `SnippetCandidate` objects and persisted `ArticleSnippet`s.
3. **Contract tests** for the `SnippetTagger` interface so alternate implementations can be dropped in safely.
4. **LLM prompt tests** (snapshot or structured assertions) ensuring tag summaries appear as expected.
5. **Regression tests** around severity flag scenarios once the LLM schema is updated.

All new functionality should be developed via TDD, starting with failing tests that describe expected behaviour, then implementing the minimal code to satisfy them.

## Open Questions for Review
- Tag taxonomy breadth: do we need additional types (e.g., `dosage`, `procedure-stage`)?
- spaCy model choice and performance impact (small vs large pipelines, custom components).
- How to collect labelled data if/when we introduce ML-based tagging.
- Strategy for backfilling tags for already-stored snippets (migration vs on-demand reprocessing).

Feedback welcome—happy to revise the plan before we start wiring modules.

## Live Tuning Script
- Run `python scripts/tune_snippet_pipeline.py "<search term>"` to resolve the condition, fetch fresh PubMed content via the NIH pipeline (unless you `--skip-fetch`), and sweep quota configs against the stored full-text articles.
- Default grid spans `base_quota` 2-4 and `max_quota` 4-7; override with `--base-range`/`--max-range`.
- Scoring weights are tunable (`--score-weight`, `--coverage-weight`, etc.) so you can prioritise high scores, diversity, or classification coverage.
- Preview output prints the top configs and, for the best one, shows sample snippets per article; add `--output preview.json` to persist the results.
- Classification coverage ensures we keep at least one snippet for each required class (`risk`/`safety` by default); cap per-drug repeats with `--limit-per-drug`.
- Fetch controls: use `--force-refresh` to always refetch articles, `--pubmed-retmax`/`--full-text-*` to tweak retrieval volume, and `--fetch-*` quotas to adjust how many snippets are stored during ingestion.
- Prioritise harm/safety evidence with `--risk-weight`/`--safety-weight`, enforce explicit condition context via `--require-condition-match`, and trim low-signal snippets with `--min-cue-count` and `--cue-weight` so spurious protocol mentions fall away.
