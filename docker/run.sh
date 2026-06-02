#!/usr/bin/env bash
# Convenience wrapper around `docker compose run`.
#
# Usage:
#   ./run.sh                # interactive shell (default)
#   ./run.sh asv run --quick
#   ./run.sh python -c 'import jax; print(jax.devices())'
#   ./run.sh nsys profile -o /work/myrun python script.py
#
# Required: the parent dir of this script must contain a `.env` file (see
# `../.env.example`). Path variables (DATA_DIR, OPERA_CSLC_DIR) and ASF
# credentials are all loaded from there.
#
# Override paths ad-hoc with shell exports if needed, e.g.
#   DOLPHIN_SRC=/other/clone ./run.sh

set -euo pipefail

cd "$(dirname "$0")"

if [[ ! -f ../.env ]]; then
    echo "error: ../.env not found. Copy ../.env.example to ../.env and fill it in." >&2
    exit 1
fi

# Auto-export host UID/GID so the image rebuilds with the right ownership
# the first time. After that, docker layer cache short-circuits the build.
export USER_ID="${USER_ID:-$(id -u)}"
export GROUP_ID="${GROUP_ID:-$(id -g)}"

# --env-file makes the same .env feed BOTH compose-time path substitution
# AND the container's runtime environment.
COMPOSE=(docker compose --env-file ../.env)

if [[ $# -eq 0 ]]; then
    exec "${COMPOSE[@]}" run --rm dev
else
    exec "${COMPOSE[@]}" run --rm dev bash -lc "$*"
fi
