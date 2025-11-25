# neverdrugs4

Clinical evidence aggregation service that normalises NIH search terms, collects PubMed artefacts, and distils them into structured claims with LLM assistance. The Flask API surfaces cached claim sets, exposes search diagnostics, and queues Celery jobs when background refreshes are required.

## System Architecture
- **API**: Flask app in `app/__init__.py` with routes under `app/api/routes.py`.
- **Database**: SQLAlchemy models backed by Postgres (default) or any `DATABASE_URL`; migrations managed by Alembic in `migrations/`.
- **Workers**: Celery (`app/celery_app.py`) with Redis broker/result backend executing `refresh_claims_for_condition`.
- **NIH Pipeline**: Services in `app/services/` orchestrate PubMed search, article retrieval, LLM batching, and processed-claim persistence.
- **LLM Integration**: `OpenAIChatClient` streams batches to OpenAI Responses API (defaults to `gpt-5-mini`).

## Quick Start (Docker)
1. Copy the example environment file and fill in the namespaced secrets (values prefixed with `NEVERDRUGS4_`):
   ```sh
   cp .env.example .env
   # edit .env to add real keys (NEVERDRUGS4_SECRET_KEY must be a long random string in production)
   ```
2. Build and launch the stack:
   ```sh
   docker compose up --build
   ```
3. The API listens on `http://localhost:8000`. The entrypoint runs `alembic upgrade head` before starting.
4. Stop the stack with `docker compose down`. Add `--volumes` to wipe Postgres data.

## Local Development Without Docker
1. Create and activate a virtual environment for Python 3.11+.
   ```sh
   python -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements-dev.txt
   ```
3. Export environment variables (see below) and start Postgres/Redis locally.
4. Run migrations:
   ```sh
   alembic upgrade head
   ```
5. Start the API:
   ```sh
   flask --app app.wsgi:app run --debug
   ```
6. Start the Celery worker in a separate shell:
   ```sh
   celery -A app.celery_app.celery worker --loglevel=info
   ```

## Configuration
| Variable | Purpose | Default |
| --- | --- | --- |
| `DATABASE_URL` | SQLAlchemy database URL | `sqlite+pysqlite:///:memory:` (overridden in Docker) |
| `REDIS_URL` | Redis connection for Celery shortcuts | `redis://redis:6379/0` |
| `CELERY_BROKER_URL` | Celery broker URL | `REDIS_URL` |
| `CELERY_RESULT_BACKEND` | Celery result backend | `CELERY_BROKER_URL` |
| `SECRET_KEY` | Flask session & CSRF signing key | _required in production_ |
| `CELERY_RESULT_TTL` | Celery result expiry (seconds) | `3600` |
| `NIH_CONTACT_EMAIL` | Email registered with the NIH API | `DEFAULT_NIH_CONTACT_EMAIL` in code |
| `NIH_API_KEY` / `NCBI_API_KEY` | API key for NIH/NCBI requests | _recommended_ |
| `OPENAI_API_KEY` | API key for OpenAI Responses API | _required_ |
| `FLASK_ENV` / `FLASK_DEBUG` | Runtime mode and debug toggles | `production` / `0` |
| `SEARCH_REFRESH_TTL_SECONDS` | Cache TTL before forcing NIH refresh | `604800` (7 days) |
| `REFRESH_JOB_STALE_SECONDS` | Override for detecting stalled running jobs | `300` |
| `REFRESH_JOB_STALE_QUEUE_SECONDS` | Override for detecting stalled queued jobs | `60` |
| `RUN_DB_MIGRATIONS` | Toggle Alembic upgrade in entrypoint | `1` |

Additional Flask configuration keys propagate via `app.config`.

## API
Base path: `http://localhost:8000/api`

### GET /health
Readiness probe. Returns `{ "status": "ok" }`.

### POST /claims/resolve
Resolve a condition to MeSH terms and optionally kick off a refresh.

**Request**
```json
{
  "condition": "Duchenne muscular dystrophy"
}
```

**Responses**
- `200 OK` – includes `resolution`, optional cached `claim_set` (with both `id` and `slug`), optional `job`, and `refresh_url` when work continues asynchronously. Clients should poll `refresh_url` until `status` becomes `completed`.
- `400 Bad Request` – missing `condition`.
- `422 Unprocessable Entity` – condition could not be matched; payload includes `suggested_mesh_terms`.
- `503 Service Unavailable` – background queueing failed.

### GET /claims/refresh/<mesh_signature>
Return the status of a claim refresh job. Accepts a mesh signature, processed claim set ID, or claim set slug. Payload fields:
- `status`: `queued`, `running`, `completed`, `failed`, `no-batches`, `no-responses`, or `skipped`.
- `progress.stage` and `progress.details` reflecting pipeline milestones.
- `resolution` snapshot containing normalized condition and mesh terms.
- `can_retry`: `true` when a client may re-trigger refresh.
- `claim_set_id` / `claim_set_slug`: populated when a processed claim set already exists for the job.

### GET /claims/<claim_set_ref>
Fetch the persisted processed-claim set (by numeric ID or human-readable slug), including claims, evidence, and drug links.

### GET /search/<search_term_ref>/query
Expose NIH search metadata: canonical condition, mesh signature, query payload, and last refresh timestamp. Payload includes `search_term_id` and `search_term_slug` so callers can persist either form of the identifier.

### GET /search/<search_term_ref>/articles
Return ranked article artefacts (PMID, score, citation, preferred URL, retrieved timestamp). Accepts either ID or slug and echoes both in the response.

### GET /search/<search_term_ref>/snippets
Return snippet-level evidence with cue metadata tied to the corresponding article ranking. Accepts either ID or slug and echoes both in the response.

## Background Jobs & Polling
- `POST /claims/resolve` enqueues `refresh_claims_for_condition` when cache is missing, stale, or recoverable.
- The Celery task progresses through: resolving updated MeSH terms, collecting PubMed artefacts, building LLM batches, invoking OpenAI, and persisting processed claims.
- Job state is tracked in `claim_set_refreshes` with `refresh_url` pointing to `/api/claims/refresh/<signature>`.

## Database & Migrations
- Models live in `app/models.py` and inherit from `app.database.Base`.
- Use Alembic for schema changes:
  ```sh
  alembic revision --autogenerate -m "describe change"
  alembic upgrade head
  ```
- Docker entrypoint applies the latest migration on startup when `RUN_DB_MIGRATIONS=1`.

## Testing
- Run the full suite:
  ```sh
  pytest
  ```
- Targeted runs are useful for contract checks, e.g.:
  ```sh
  pytest tests/test_api_scaffolding.py
  pytest tests/test_http_integration.py
  ```
- HTTP integration tests replay fixture data from `tests/fixtures/`; they require no external network calls.
- Ensure Celery is not running concurrently when invoking local tests to avoid unexpected background writes.

## Project Structure
- `app/api/` – Flask blueprints.
- `app/services/` – NIH search, LLM batching, persistence helpers.
- `app/tasks.py` – Celery task entrypoints.
- `migrations/` – Alembic environment and revisions.
- `deployment/` – Docker/Compose definitions, entrypoints, and deployment docs.
- `scripts/` – Developer utilities.
- `tests/` – Pytest suites and recorded fixtures.
- `docs/` – Supplemental design notes.

## Troubleshooting
- If jobs appear stuck, check `/api/claims/refresh/<signature>` for `can_retry=true` and re-trigger resolution.
- Reset the local database with `docker compose down --volumes` when schema or fixture drift causes failures.
- Ensure `OPENAI_API_KEY` is present before starting the worker; the LLM client fails fast otherwise.
