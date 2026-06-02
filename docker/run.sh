#!/usr/bin/env bash
# Convenience wrapper around `docker compose run`.
#
# Usage:
#   ./run.sh                # interactive shell (default)
#   ./run.sh asv run --quick
#   ./run.sh python -c 'import jax; print(jax.devices())'
#   ./run.sh nsys profile -o /work/myrun python script.py
#
# Override paths via environment:
#   DOLPHIN_SRC=/other/clone ./run.sh
#   DATA_DIR=/other/safe ./run.sh

set -euo pipefail

cd "$(dirname "$0")"

# Auto-export host UID/GID so the image rebuilds with the right ownership
# the first time. After that, docker layer cache short-circuits the build.
export USER_ID="${USER_ID:-$(id -u)}"
export GROUP_ID="${GROUP_ID:-$(id -g)}"

if [[ $# -eq 0 ]]; then
    exec docker compose run --rm dev
else
    exec docker compose run --rm dev bash -lc "$*"
fi
