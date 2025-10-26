# NIH Module Design Document

## 1. Problem Statement & Scope
- Build a Flask-based web application module, containerised with Docker and served via Gunicorn, that ingests a user-specified medical condition and surfaces anaesthesia-related drug safety evidence.
- Focus: end-to-end pipeline from user input through literature retrieval, NLP processing, LLM summarisation, and final structured outputs.
- Out of scope (for this iteration): user authentication, billing, multi-language support, heavy front-end styling.

## 2. Goals & Non-Goals
### Goals
- Provide Flask-importable endpoints/services that integrate with the host application's UI and database.
- Deliver a reproducible runtime using Docker Compose (web + worker + Postgres + Redis) with Gunicorn fronting the Flask app.
- Automate literature search across NIH E-utilities (ESearch, ESummary, EFetch) for condition + anaesthesia, optimising query construction while minimising total requests.
- Retrieve abstracts (and full texts where available) with authoritative URLs.
- Rank and slice articles for relevant anaesthesia drug mentions and safety/adverse indicators while constraining MeSH alignment to reduce off-target hits.
- Produce LLM-backed JSON analyses grouped by drug, including citations, within an OpenAI GPT-4o mini budget envelope.
- Persist reusable artefacts (queries, MeSH mappings, article texts, LLM summaries) with efficient lookup to support cache hits across sessions and users.
- Honor caller-defined TTL policies, returning stale artefacts immediately while scheduling background refresh (stale-while-revalidate).

### Non-Goals
- Manual curation of results.
- Long-term storage or analytics (beyond transient caching/logging for retries).
- Deep PDF parsing quality improvements (baseline only unless critical).

## 3. Key User Flows
1. Host Flask app collects a medical condition from the user and invokes the module.
2. Module executes the synchronous MVP pipeline (search → retrieval → analysis), emitting progress updates to the database/logs.
3. Host app queries status/results endpoints to render staged updates (counts, snippets, final drug summaries).
4. Host app stores and/or displays final JSON artefacts; future iteration may add export options.

## 4. High-Level Architecture
- **Integration Layer**: Blueprint factory or service objects that attach to the host Flask application and reuse its SQLAlchemy session.
- **Orchestration**: Core request handling remains synchronous while Celery/Redis is available for offloading long-running or retryable pipeline stages.
- **Execution Environment**: Docker Compose stack managing the Gunicorn-backed Flask web service, Celery worker, PostgreSQL database, and Redis broker for local and deployment parity.
- **Pipeline Services**:
  - Search client hitting NIH E-utilities/PubMed and other sources via REST.
  - Retrieval component for abstracts/full text (via EFetch, OA API, or fallback scraping services).
  - NLP processing pipeline (spaCy/NLTK/LangChain stack) handling ranking, windowing, and extraction.
  - LLM service wrapper for OpenAI API calls with retry/backoff and cost tracking.
- **Storage/State**: Persist artefacts in the host Flask database (PostgreSQL in production, SQLite during development); optional Redis cache added once async orchestration is enabled.
- **Observability**: Structured logging, metrics, and tracing hooks for pipeline stages.

## 5. Data Flow
1. **Input Validation**: Sanitise condition input; normalise via MeSH lookup where available.
2. **Query Generation**: Build search strings (condition + "anesthesia" + synonyms) with tuned operators to maximise precision.
3. **Cache Lookup**: Check database cache for existing artefacts keyed by canonicalised condition strings (case-insensitive, accent folded) and associated MeSH IDs; evaluate caller-specified TTL and default freshness rules.
4. **Search Execution**: On cache miss perform PubMed ESearch; on stale hit return cached payload while queuing a background refresh.
5. **Metadata Retrieval**: Fetch summaries via ESummary (title, journal, publication date, DOI) and cache metadata per article.
6. **Text Retrieval**: Pull abstracts (EFetch) and attempt full text via Open Access or document services; persist normalised text artefacts for reuse.
7. **URL Construction**: Build canonical URLs (e.g., `https://pubmed.ncbi.nlm.nih.gov/{pmid}`) and DOI links.
8. **MeSH Normalisation**: Map detected drugs/conditions to MeSH IDs and filter out off-target matches; cache mappings for future lookups.
9. **Relevance Ranking**: Score documents using keyword density, BERT embedding similarity, or BM25 variants.
10. **Windowed Extraction**: Slide configured windows to detect anaesthesia drug mentions and safety/adverse cues.
11. **Evidence Assembly**: Store matched snippets with article identifiers and metadata.
12. **LLM Summarisation**: Check cached summaries before invoking OpenAI; otherwise package context windows, send to the model, and persist the JSON response artefact with versioning and TTL metadata.
13. **Aggregation**: Group claims per drug, tag as safe/adverse, attach citations/URLs, and cache collated outputs.
14. **Output Delivery**: Persist results in the host database and expose via module interfaces.

## 6. Component Design
- **Flask Integration Layer**
  - Blueprint factory/service registering routes such as `/nih/search` (POST), `/nih/status/<request_id>` (GET), `/nih/results/<request_id>` (GET).
  - Optional WebSocket or Server-Sent Events for long-running statuses.
- **Task Orchestration**
  - MVP: synchronous execution with progress persisted to DB/logs while delegating long-running or refresh tasks to the Celery worker when needed.
  - Celery worker communicates via Redis broker (already provisioned in Docker Compose) enabling painless expansion to full async pipelines.
- **Cache Service**
  - Handles canonicalisation, TTL evaluation, stale-while-revalidate scheduling, and manual invalidation hooks.
  - Utilises Celery/Redis for scheduled refresh jobs; lightweight in-process fallbacks remain available for unit tests.
- **Search Client**
  - Wrap NIH Entrez utilities; manage rate limits via API key, caching, exponential backoff.
- **Retrieval Layer**
  - Normalise text encoding, store raw and cleaned versions, attach metadata.
- **NLP Pipeline**
  - Tools: spaCy for NER, SciSpacy for biomedical terms, custom regex for drug names, optionally metamap.
  - Window size configurable (e.g., 100-200 tokens) with overlap.
  - Sentiment/safety classification via rule-based lexicon + ML fallback.
- **LLM Integration**
  - Use OpenAI GPT-4o mini via official SDK with configurable per-request spend guards.
  - Prompt templates ensuring deterministic JSON schema, guard against token limits (chunking by drug/article).
- **Data Model**
  - Persist request metadata (`id`, `condition`, input variants, status, timestamps, errors, cost metrics`) via host SQLAlchemy session.
  - **SearchTerm**: stores canonical string (lowercase, accent folded), display variants, hashing for quick lookups.
  - **SearchArtefact**: joins search terms to MeSH IDs, retains constructed E-utilities query JSON, TTL policy, and last refreshed timestamps.
  - **ArticleArtefact**: caches article metadata, abstract/full-text payloads, canonical URLs, provenance info, TTL and stale flags.
  - **EvidenceSnippet**: article id, drug, sentiment (`safe`/`adverse`), snippet text, scoring metadata, window parameters.
  - **LLMSummary**: per drug aggregated claims stored as JSONB (PostgreSQL) or TEXT + JSON serialisation (SQLite dev) with model version, prompt hash, citation list, TTL metadata.
  - Maintain uniqueness constraints to ensure strings like `duchenne`, `Duchenne`, and `Duchenne muscular dystrophy` resolve to shared canonical entries.
- **Config & Secrets**
  - `.env` for API keys, queue settings; load via `python-dotenv` and propagate through Docker Compose service definitions.
- **Logging & Monitoring**
  - Use `structlog` or standard logging with JSON formatter.
  - Capture metrics (processed articles count, API latency) for debugging.

## 7. External Dependencies & Tooling
- Flask integration utilities, Celery 5.x with Redis broker, Requests/HTTPX, Biopython or `entrezpy` for NIH APIs, spaCy + SciSpacy, MeSH resources, OpenAI Python SDK targeting GPT-4o mini, optional Sentence Transformers (for ranking).
- Runtime platform: Gunicorn application server orchestrated via Docker Compose alongside PostgreSQL and Redis.
- Development tooling: pip-tools for dependency pinning, Black, Flake8, isort, and mypy for code quality, structlog for structured logging.
- PDF parsing: `pdfminer.six` or `pymupdf` when full text arrives as PDF.

## 8. Performance & Scaling Considerations
- Limit search results (e.g., top 200 PMIDs) with pagination.
- Parallelise retrieval and NLP across worker pool.
- Cache previously processed articles per condition to avoid redundant API calls.
- Introduce configurable timeouts and circuit breakers for external APIs.

## 9. Reliability & Error Handling
- Graceful degradation when full text unavailable (fallback to abstract only).
- Structured error propagation to UI with actionable messages.
- Retry policies tuned per external dependency (e.g., exponential backoff for PubMed).

## 10. Security & Compliance
- Store API keys securely (env vars, secrets manager).
- Respect NIH rate limits and terms of service.
- Ensure patient input is non-identifiable; avoid logging sensitive user data.
- Consider HIPAA/GDPR implications if storing results long term.

## 11. Testing Strategy
- Unit tests for query builder, parser, NLP windowing, LLM prompt formatting.
- Integration tests using recorded NIH API responses (VCR-like fixtures).
- Contract tests for LLM JSON schema validation.
- E2E test path via Flask test client and mocked background workers.

## 12. Deployment Plan
- Maintain Docker Compose stack with `web` (Gunicorn Flask app), `worker` (Celery), `postgres`, and `redis` services for local and hosted environments.
- Bake dependency pins via pip-tools during CI, rebuilding images after `pip-compile` updates to keep runtime deterministic.
- Staging environment hitting NIH sandbox or recorded responses.
- CI pipeline running lint/tests, gating deploys.

## 13. Risks & Mitigations
- **External API limits**: implement caching, throttle requests, support manual retry.
- **LLM hallucinations**: strict prompt instructions, post-processing validation, fallback of rule-based summaries.
- **Incorrect drug detection**: maintain whitelist/blacklist, incorporate curated dictionaries.
- **Latency**: asynchronous processing, streaming status updates, adjustable result limits.
- **Cache staleness**: enforce TTL policies, stale-while-revalidate refresh jobs, and manual invalidation tooling.

## 14. Open Questions
1. Do we need additional literature sources beyond PubMed/NIH for phase two?
2. What cache invalidation or freshness policies are acceptable for literature and LLM artefacts?
3. Are there guardrails for maximum runtime per request before async execution becomes mandatory?
4. How should off-target MeSH matches be curated (manual review vs. automated heuristics)?
5. What reporting/export formats (besides JSON) should we anticipate for future iterations?
