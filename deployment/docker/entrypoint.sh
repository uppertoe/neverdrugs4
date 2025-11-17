#!/bin/sh
set -euo pipefail

if [ -z "${DATABASE_URL:-}" ]; then
  export DATABASE_URL="postgresql+psycopg2://postgres:postgres@postgres:5432/neverdrugs4"
fi

RUN_DB_MIGRATIONS="${RUN_DB_MIGRATIONS:-1}"

if [ "$RUN_DB_MIGRATIONS" = "1" ] || [ "$RUN_DB_MIGRATIONS" = "true" ] || [ "$RUN_DB_MIGRATIONS" = "True" ]; then
  echo "[*] Running alembic upgrade head"
  alembic upgrade head
else
  echo "[*] Skipping alembic upgrade head"
fi

echo "[*] Starting command: $*"
exec "$@"
