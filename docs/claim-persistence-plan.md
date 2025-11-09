# Claim Persistence & Feedback Plan

## Goals
- Persist post-LLM claims in the database so the UI can surface and annotate them (votes, notes, moderation).
- Protect existing data from corruption by staging, validation, and versioning.
- Preserve the ability to roll back to a known-good snapshot.

## Proposed Data Model Changes
- **ProcessedClaimSetVersion**
  - Links to `ProcessedClaimSet` (one-to-many).
  - Tracks `version_number`, `created_at`, `status` (`draft`, `active`, `superseded`, `failed`).
  - Holds metadata about the run (mesh signature, pipeline config hash, LLM model, job id).
- **ProcessedClaim**
  - Associate each claim with a `ProcessedClaimSetVersion` (instead of directly with the set).
  - Add immutable `canonical_hash` for idempotent upserts.
  - Introduce a cross-version `claim_group_id` (derived from the canonical hash) so feedback can aggregate across versions.
  - Keep existing fields (classification, summary, drugs, evidence) unchanged.
- **ProcessedClaimFeedback**
  - Stores individual user votes/flags (`user_id`, `vote` enum, optional `comment`).
  - Foreign key to `ProcessedClaim` with timestamps.
  - Aggregated columns (e.g. `up_votes`, `down_votes`) denormalised on `ProcessedClaim` for fast reads.
- Consider a lightweight **ProcessedClaimAudit** table (JSON diff snapshot) if we need full audit trails.

## Pipeline Workflow Updates
1. **Staging Phase**
   - Collect LLM responses into an in-memory structure.
   - Deduplicate/snippet aggregation as we do today.
   - Run validation checks (non-empty drugs, valid classifications, scores within range, evidence count).
   - Record validation stats for logging/alerting.
2. **Persistence Phase**
   - Begin a transaction.
   - Create a new `ProcessedClaimSetVersion` with status `draft`.
  - Upsert claims: reuse existing `canonical_hash`/`claim_group_id` to avoid duplicates; preserve feedback counters when reshaping existing claims.
   - Insert/update evidence and snippet links tied to claim id.
  - Mark old active version as `superseded`, but keep rows (soft-delete only after retention window expires).
   - Set new version status to `active` once all checks passed.
   - Commit.
3. **Rollback Strategy**
   - If validation fails, abort before creating the version.
   - If persistence fails mid-flight, roll back transaction to leave previous version untouched.
   - A CLI/maintenance script can revert the active flag to a previous version.

## Validation & Sanity Checks
- **Structural**: ensure every claim has classification, summary, at least one drug, and evidence snippet.
- **Statistical**: compare counts per classification/drug to previous version within tolerance; alert on large deviations.
- **Consistency**: verify canonical hashes unique per version; ensure no duplicate evidence ids.
- **Integrity**: check foreign key relationships, alignment with mesh signature used for the run.

## Migration Plan
1. Create Alembic migration to add new tables/columns with defaults.
2. Backfill existing `ProcessedClaim` rows into a `ProcessedClaimSetVersion` (version 1) to preserve history.
3. Update ORM models/relationships; adjust serializers.
4. Deploy with feature flag that keeps pipeline in read-only mode until validated.
5. Re-run pipeline on a sample condition, validate metrics, then enable globally.

## API & UI Considerations
- Extend `/api/claims/{id}` to return active version metadata plus vote aggregates.
- Add endpoints for submitting votes/feedback (with authentication/limiting).
  - For anonymous users, issue signed client tokens and apply per-token/IP rate limiting before accepting votes.
- Provide a `/api/claims/resolve` response that includes version id so the UI can refresh on change.
- Potentially expose audit history for transparency (version list endpoint).

## Observability
- Emit logging/metrics per run: claim counts, validation failures, dedupe ratios, voting totals.
- Alert on failed validations or rollback events.
- Store run metadata in `ProcessedClaimSetVersion` for later inspection.
- Optionally prune superseded versions beyond a defined rolling retention window once archived.

## Open Questions
- How long do we retain superseded versions? Unlimited vs. rolling window (default: rolling window with manual override).
- Do we need soft-delete semantics for claims that vanish between runs? (Likely soft delete with retention window.)
- Where does user identity for feedback come from (existing auth vs. future system)?
- Feedback should span all versions of the same canonical claim while preserving per-version history for auditing.

> Next step: review and refine this plan, confirm schema details, then start with the migration + staging pipeline changes.
