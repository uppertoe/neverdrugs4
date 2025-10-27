# Frontend–Backend Integration Plan

## Overview

We will expose the existing evidence pipeline through a lightweight Flask web application. Users submit clinical search terms, observe live pipeline progress (or cached results), and review the final LLM-derived claim summaries. The UI emphasises simplicity—Bootstrap for layout, HTMX for incremental updates—and may add WebSocket support later for richer real-time feedback.

## User Journey

1. **Search input** – User enters a condition or query term in the Flask UI.
2. **Immediate response**:
   - If a processed claim set already exists and is fresh, return it immediately.
   - Otherwise, show a status panel indicating that the pipeline is running.
3. **Pipeline execution** – Backend performs NIH search, article retrieval, snippet extraction, LLM batching, and post-hoc claim aggregation. Progress indicators are surfaced to the UI.
4. **Result presentation** – On completion, user sees an ordered list of claims (risk vs safety) alongside supporting evidence and optional per-drug filtering.
5. **Background refresh** – Cached results trigger a background refresh when data might be stale; users receive an asynchronous notification when updates are ready.

## Backend Responsibilities

### API Endpoints
- `POST /api/claims/resolve` – Accepts a search term, returns existing processed data or triggers a new pipeline run. Supports `force` flag.
- `GET /api/claims/<id>` – Retrieves a processed claim set, including evidence and drug mapping.
- `GET /api/claims/status/<task_id>` – Optional endpoint if processing runs asynchronously (Celery or background worker).
- `GET /api/articles/<search_term_id>` and `GET /api/snippets/<search_term_id>` – Optional drill-down endpoints if the UI requires raw artefacts.

### Persistence
- Store processed outputs (`ProcessedClaimSet`, `ProcessedClaim`, `ProcessedClaimEvidence`, `ProcessedClaimDrugLink`).
- Track job metadata (start time, completion time, errors) if async, so the frontend can poll status.
- Maintain article/snippet artefacts as today for auditability.

### Refresh Strategy
- On each `resolve` request:
  1. Look up the canonical mesh signature for the term.
  2. Return existing processed claims if within TTL and article list unchanged.
  3. Otherwise kick off pipeline run. The UI receives a job token or the old data plus a “refresh pending” flag.

## Frontend Responsibilities

### Stack
- Flask templates + Bootstrap for base layout.
- HTMX for small, server-rendered fragments (e.g., updating progress sections, refreshing claim list) without full page reloads.
- Optional WebSocket (Flask-SocketIO or similar) for real-time status updates; can fall back to HTMX polling if needed.

### Components
1. **Search Form** – Single input with submit button, posts to `resolve` endpoint.
2. **Status Panel** – Displays the current stage (NIH search, article fetch, snippet extraction, LLM processing, claim aggregation). Updates via HTMX partials or WebSockets.
3. **Results View** – Shows processed claims grouped by classification (“Risk” vs “Safety”), sorted by confidence. Each claim displays summary, confidence, supporting snippets with citation URLs.
4. **Drug Filter Sidebar** – Lists drugs/classes from the processed claim set; selecting a drug filters the claims in-place (HTMX request that re-renders the claim list with applied filter).
5. **Notifications** – When a background refresh completes, push a toast/alert (via WebSocket or HTMX poll that detects completion) prompting the user to reload results.

## Data Contracts
- Responses echoed to the frontend should align with the structure now emitted under `processed_claims` in `e2e_capture.json`:
  ```json
  {
    "id": 12,
    "mesh_signature": "king denborough|anesthesia",
    "condition_label": "King Denborough syndrome",
    "claims": [
      {
        "claim_id": "risk:succinylcholine",
        "classification": "risk",
        "confidence": "high",
        "summary": "…",
        "drugs": ["succinylcholine"],
        "drug_classes": ["depolarising neuromuscular blocker"],
        "supporting_evidence": [
          {
            "snippet_id": "42",
            "pmid": "11111111",
            "article_title": "Safety considerations…",
            "citation_url": "https://pubmed…",
            "key_points": ["…"],
            "notes": "…"
          }
        ],
        "drug_links": [
          {"term": "succinylcholine", "term_kind": "drug"},
          {"term": "depolarising neuromuscular blocker", "term_kind": "drug_class"}
        ]
      }
    ]
  }
  ```
- Include `updated_at`, `created_at`, and job status metadata in responses to support UI messaging.

## Real-Time Status Options
- **HTMX polling**: status panel issues periodic `GET /api/claims/status/<task_id>` requests; when status is `completed`, replace the panel with final results.
- **WebSockets**: when job starts, backend emits stage updates; client subscribes to `ws://…/status/<task_id>` channel and updates UI as events arrive.
- Initial implementation can use HTMX polling (simpler) with eventual upgrade to WebSockets if latency or UX demands it.

## Outstanding Questions
- TTL policy: what constitutes “stale” (time-based vs article list changes)?
- Background worker choice: Celery vs threading vs async task queue.
- Authentication: is the interface internal-only, or do we need session-based auth?
- Notification UX: toast vs banner vs unobtrusive indicator.
- Schema migrations: plan for generating Alembic migrations to deploy new tables.

## Next Steps
1. Finalise TTL and refresh heuristics.
2. Implement API blueprint with read + resolve endpoints and JSON schemas.
3. Add session management helpers (scoped session per request).
4. Build HTML templates with Bootstrap + HTMX partials.
5. Implement background refresh (initially synchronous, upgrade to async).
6. Write integration tests covering API endpoints and HTMX fragments.
7. Extend ops tooling (Alembic migrations, deployment notes).
