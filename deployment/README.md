# Deployment

This directory contains the assets required to build and run the neverdrugs4 stack in Docker.

## Images and Publishing

1. Build the production image (tag locally and with the Docker Hub repo name):
   ```sh
   docker build -f Dockerfile -t neverdrugs4:latest .
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

The compose file is now scoped to this directory to keep operational assets in one place. It runs in production mode by default (`FLASK_ENV=production`, `FLASK_DEBUG=0`), so be sure to set a strong `SECRET_KEY` in `.env` before deploying.

```sh
# Start (foreground)
docker compose -f deployment/docker-compose.yml up --build

# Start (detached)
docker compose -f deployment/docker-compose.yml up --build -d

# Stop
docker compose -f deployment/docker-compose.yml down

# Wipe state as well
docker compose -f deployment/docker-compose.yml down --volumes
```

The compose file mounts the repository into `/usr/src/app`, so edits on the host refresh inside the containers while developing.

## Environment Variables

Set the following before running in production:

- `OPENAI_API_KEY` – required for the LLM pipeline.
- `DATABASE_URL` – override the default Postgres credentials when not using the bundled database.
- `RUN_DB_MIGRATIONS` – set to `0` if you manage schema upgrades separately.

## Entrypoint

`deployment/docker/entrypoint.sh` runs Alembic migrations (when enabled) and then forwards to the command supplied by compose.
