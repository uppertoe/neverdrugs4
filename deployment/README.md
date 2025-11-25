# Deployment

This directory contains the assets required to build and run the neverdrugs4 stack in Docker.

## Images and Publishing

1. Build the production image (tag locally and with the Docker Hub repo name):
   ```sh
   docker build -f deployment/docker/Dockerfile -t neverdrugs4:latest .
   docker tag neverdrugs4:latest docker.io/<your-org>/neverdrugs4:latest
   ```
2. Push to Docker Hub:
   ```sh
   docker push docker.io/<your-org>/neverdrugs4:latest
   ```
3. Optionally push a versioned tag:
   ```sh
   export VERSION="v$(date +%Y.%m.%d)"
   docker tag neverdrugs4:latest docker.io/<your-org>/neverdrugs4:${VERSION}
   docker push docker.io/<your-org>/neverdrugs4:${VERSION}
   ```

## Running with Docker Compose

The primary compose entrypoint lives at the repository root (`docker-compose.yml`). It runs in production mode by default (`FLASK_ENV=production`, `FLASK_DEBUG=0`), so be sure to set a strong `SECRET_KEY` in `.env` before deploying.

```sh
# Start (foreground)
COMPOSE_PROJECT_NAME=neverdrugs4 docker compose up --build

# Start (detached)
COMPOSE_PROJECT_NAME=neverdrugs4 docker compose up --build -d

# Stop
COMPOSE_PROJECT_NAME=neverdrugs4 docker compose down

# Wipe state as well
COMPOSE_PROJECT_NAME=neverdrugs4 docker compose down --volumes
```

Setting `COMPOSE_PROJECT_NAME` (or adding `name: neverdrugs4` inside `compose.yaml`) keeps service, volume, and network names unique when this stack shares a host with other Compose apps like Planka or Keycloak. The sample production file `docker-compose-example.yml` also hardcodes namespaced service keys, container names, network, and volume identifiers so collisions are less likely even without the project name override. The `.env` file stores only sensitive values, each prefixed with `NEVERDRUGS4_` so they don’t collide with other stacks on the same host; compose maps them back to the canonical variable names inside the containers.

If you prefer to keep the deploy command scoped to this directory, continue to pass `-f deployment/docker-compose.yml`; both files define the same services.

The compose configuration mounts the repository into `/usr/src/app`, so edits on the host refresh inside the containers while developing.

## Environment Variables

Set the following before running in production:

- `OPENAI_API_KEY` – required for the LLM pipeline.
- `DATABASE_URL` – override the default Postgres credentials when not using the bundled database.
- `RUN_DB_MIGRATIONS` – set to `0` if you manage schema upgrades separately.

## Entrypoint

`deployment/docker/entrypoint.sh` runs Alembic migrations (when enabled) and then forwards to the command supplied by compose.
