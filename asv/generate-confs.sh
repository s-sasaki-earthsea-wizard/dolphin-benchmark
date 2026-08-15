#!/usr/bin/env bash
# Generate asv config variants from dolphin's upstream asv.conf.json.
#
# Upstream asv.conf.json is the single source of truth and is never edited.
# These configs are derived mechanically so upstream field changes propagate
# on regeneration instead of drifting out of sync (asv's --config is a full
# replacement, not an overlay — a hand-written partial config silently drops
# upstream fields like build_command or conda_channels).
#
#   asv-existing-gpu.conf.json   R1/R2: container's existing dolphin-env, GPU
#   asv-existing-cpu.conf.json   R3:    same env, run with JAX_PLATFORMS=cpu
#   asv-ci.conf.json             R4/R5: upstream-equivalent mamba env; only
#                                paths are overridden
#
# GPU and CPU runs get physically separate results_dir trees. asv's
# `existing` environment type ignores the requirement matrix entirely
# (get_environments() in asv/environment.py: "# Ignore requirement matrix"),
# so matrix env_nobuild cannot split env names, and both backends would
# otherwise write the identically-named result file (machine + commit +
# env name) — the second run silently overwriting the first.
#
# All paths inside the configs are container paths (docker-compose mounts):
#   /dolphin  = dolphin clone (bind mount)
#   /work     = dolphin-benchmark/results (bind mount; only JSONs land on NAS)
#   /asv-env  = named volume on container-local disk (env + build writes)
#
# Note: the comment stripper removes whole-line // comments only — the form
# upstream uses. An inline // after a value would survive and break jq.

set -euo pipefail

cd "$(dirname "$0")"

UPSTREAM_CONF="${UPSTREAM_CONF:-../../asv.conf.json}"
# Branch recorded for existing-env runs; must match the checkout at /dolphin.
ASV_BRANCH="${ASV_BRANCH:-feature/545-jax-similarity}"

strip_comments() { grep -v -E '^[[:space:]]*//' "$UPSTREAM_CONF"; }

common='.repo = "/dolphin" | .benchmark_dir = "/dolphin/benchmarks"'
existing='.environment_type = "existing"
  | .pythons = ["same"]
  | .branches = [$branch]
  | .env_dir = "/asv-env/existing"'

for backend in gpu cpu; do
    strip_comments | jq --arg branch "$ASV_BRANCH" --arg be "$backend" "
        $common | $existing
        | .results_dir = (\"/work/asv-baseline/\" + \$be)
        | .html_dir = (\"/asv-env/html/existing-\" + \$be)
    " > "asv-existing-${backend}.conf.json"
done

# CI-reproduction config: keep every upstream field (mamba, pythons,
# conda_channels, build_command); override paths only. conda_environment_file
# must be absolute — asv resolves it against the cwd, not the config location.
strip_comments | jq "
    $common
    | .conda_environment_file = \"/dolphin/conda-env.yml\"
    | .env_dir = \"/asv-env/ci\"
    | .results_dir = \"/work/asv-ci\"
    | .html_dir = \"/asv-env/html/ci\"
" > asv-ci.conf.json

echo "Generated from ${UPSTREAM_CONF} (branch label: ${ASV_BRANCH}):"
ls -l asv-existing-gpu.conf.json asv-existing-cpu.conf.json asv-ci.conf.json
