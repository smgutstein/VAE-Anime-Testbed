#!/usr/bin/env bash
set -euo pipefail

# Launch an interactive development shell inside the VAE Docker container.
# Commands are run from inside the opened shell.

# Directory containing this script: project-root/Dockerfiles
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.yml"
SERVICE_NAME="vae-anime"

# Pass the host user's identity into docker compose.
# docker-entrypoint.sh uses these to create a matching user inside the container.
export HOST_UID="$(id -u)"
export HOST_GID="$(id -g)"
export USER_NAME="${USER:-vaeuser}"
export GROUP_NAME="$(id -gn 2>/dev/null || echo vaeuser)"

# Build the image if needed and start the dev container if needed.
docker compose -f "${COMPOSE_FILE}" up -d --build "${SERVICE_NAME}" >/dev/null

# Open a bash shell inside the running container as the host-matching user.
# docker compose exec does not rerun ENTRYPOINT, so --user is needed here.
# --noprofile --norc avoids base-image shell startup scripts that may try
# to run root-only setup such as ldconfig.
docker compose -f "${COMPOSE_FILE}" exec \
  --user "${HOST_UID}:${HOST_GID}" \
  -e PS1="(vae-docker) \u@vae:\w\$ " \
  "${SERVICE_NAME}" \
  bash --noprofile --norc -i