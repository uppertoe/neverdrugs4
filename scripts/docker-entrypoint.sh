#!/bin/sh
# Legacy shim: delegate to the deployment entrypoint so existing docs still work.
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname "$0")" && pwd)"
exec "$SCRIPT_DIR/../deployment/docker/entrypoint.sh" "$@"
