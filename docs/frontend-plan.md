# Pipeline Frontend Plan

## Goals
- Provide a template-driven UI (Bootstrap + HTMX) so non-technical users can drive the claim pipeline from the browser.
- Keep interactions snappy: every request returns immediately with either cached results or a progress panel; background work updates via polling HTMX endpoints that return partial templates.
- Allow users to search for a condition, reuse cached resolution when available, or walk through MeSH term selection when we cannot auto-resolve.
- Surface live status indicators that mirror the backend pipeline milestones (NIH search, MeSH resolution, full-text retrieval, LLM batches).

## High-Level Flow
1. **Landing/Search Page**
   - Simple form (condition input + optional advanced options toggle).
   - On submit: `hx-post` to `/ui/search`, targeting a results container.
2. **Cache Hit Path**
   - Server detects existing `SearchTerm`/claim set and immediately renders the cached claims table.
   - Provide a "Refresh" button to force a new pipeline run if desired.
3. **Cache Miss Path**
   - Resolve MeSH terms via NIH dispatcher (background job).
   - Instantly render a progress panel with placeholders for each pipeline stage and an HTMX polling component pointed at `/ui/status/<job_id>`.
4. **Manual MeSH Selection**
   - If the dispatcher cannot pick a confident primary term, render a partial containing a ranked list (radio buttons) of candidate MeSH descriptors plus "Run"/"Cancel" actions.
   - Submission triggers a new `/ui/mesh-select` endpoint to start the pipeline with the selected terms.
5. **Progress Updates**
   - Poll `/ui/status/<job_id>`; response returns a fragment updating:
     - Current stage & timestamp (e.g. "Search sent to NIH" ✔️, "Mesh terms resolved" loading spinner).
     - Optional debug info (resolved terms, snippet counts) behind a collapsible panel.
      - When the job finishes, the status endpoint swaps in the claims table and emits a toast confirming completion.
      - Empty or failed runs surface a warning/error banner with retry guidance.

6. **Completion**
   - Final status endpoint returns the full claims table partial.
   - Polling stops via `hx-trigger="load"` or server-sent instruction.

## Endpoint & Template Plan

| Endpoint | Method | Purpose | Returns |
| --- | --- | --- | --- |
| `/ui` | GET | Landing page | Full page template (`layout.html`) with search form + empty results panel |
| `/ui/search` | POST (HX) | Handle condition submissions | Partial: either cached results (`_claims_table.html`) or progress panel (`_progress_panel.html`) |
| `/ui/mesh-options/<job_id>` | GET | Serve the MeSH selection UI when needed | Partial `_mesh_selector.html` |
| `/ui/mesh-select/<job_id>` | POST (HX) | Accept user-selected mesh terms | Partial progress panel (restarted) |
| `/ui/status/<job_id>` | GET | Return current pipeline status | Partial `_progress_panel.html` or `_claims_table.html` on completion |
| `/ui/claims/<claim_set_id>` | GET | Render claims view directly (non-HX fallback) | Full page |

### Templates (Jinja2/Flask)
- `layout.html`: global Bootstrap shell, includes HTMX & partial containers.
- `_search_form.html`: reused on landing and maybe sidebar.
- `_progress_panel.html`: status timeline list with stage badges (`queued`, `running`, `completed`, `failed`).
- `_mesh_selector.html`: instructions + candidate mesh list, uses `hx-post` to submit selection.
- `_claims_table.html`: summary header (condition, mesh signature, refresh controls) + table of claims grouped by classification; evidence collapsible per claim.
- `_status_toast.html`: optional toast component for transient notifications (errors, refresh started).

## Status Tracking Contract
Stages to visualise (matching backend `progress_state`):
1. **Search sent to NIH** – we have queued NIH requests (dispatcher).  
2. **Mesh terms resolved** – either cached terms reused or new ones chosen.  
3. **Full text search began** – PubMed/full-text pipeline fetching articles/snippets.  
4. **LLM request sent** – compilation of batches and dispatch to OpenAI.  
5. **Claims assembled** – final data persisted.

Each status update should include timestamps and optional detail (e.g. resolved terms, snippet counts). We can map backend progress payload to a simple structure consumed by the template.

## HTMX Interactions
- Use `hx-target` to swap specific sections. Example:
  ```html
  <form hx-post="/ui/search" hx-target="#results" hx-indicator="#search-spinner">
  ```
- Polling example:
  ```html
  <div hx-get="/ui/status/{{ job_id }}" hx-trigger="load delay:1s" hx-target="#status" hx-swap="outerHTML" hx-poll="2s"></div>
  ```
  We can disable polling when the response includes `hx-trigger="stop"` or a hidden flag.

## Bootstrap Layout Hints
- Use `container` + `row` for main layout.
- Summaries in cards; progress via `list-group` with icons/spinners.
- Claim evidence collapsible accordions to keep page readable.

## Future Extensions
- Auth wrapper if we need to restrict usage.
- WebSocket fallback for real-time updates without polling.
- Download/export options (CSV, JSON) for claim sets.
- Admin panel to inspect queued jobs, errors.
