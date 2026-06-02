SHELL := /bin/bash
.DEFAULT_GOAL := help

# All actual work runs inside the dev container via the wrapper, which
# auto-loads ../.env and exports host UID/GID. The wrapper is the single
# source of truth for compose flags.
RUN := ./docker/run.sh

.PHONY: help
help:  ## Show this help.
	@awk 'BEGIN {FS = ":.*?## "} \
		/^[a-zA-Z][a-zA-Z0-9_-]*:.*?## / {printf "  \033[36m%-30s\033[0m %s\n", $$1, $$2}' \
		$(MAKEFILE_LIST)

# ---------------------------------------------------------------------------
# Container lifecycle
# ---------------------------------------------------------------------------

.PHONY: build
build:  ## Build the GPU dev image.
	cd docker && export USER_ID=$$(id -u) GROUP_ID=$$(id -g) && \
		docker compose --env-file ../.env build

.PHONY: rebuild
rebuild:  ## Rebuild the image from scratch (no cache).
	cd docker && export USER_ID=$$(id -u) GROUP_ID=$$(id -g) && \
		docker compose --env-file ../.env build --no-cache

.PHONY: shell
shell:  ## Interactive shell inside the dev container.
	$(RUN)

# ---------------------------------------------------------------------------
# CSLC tutorial data
# ---------------------------------------------------------------------------
#
# Defaults match dolphin's basic walkthrough notebook: West Texas, track 78,
# burst T078-165573-IW2, S1B. The 12-month range yields ~30-40 CSLCs, ~10 GB.
# The downloader is idempotent — re-running skips files already on disk.

.PHONY: download-cslc-tutorial
download-cslc-tutorial:  ## Download the full 12-month tutorial CSLC stack (~10 GB).
	$(RUN) python /dolphin-benchmark/scripts/download_cslc.py

.PHONY: download-cslc-small
download-cslc-small:  ## Download a 1-month subset for quick testing (~1 GB).
	$(RUN) python /dolphin-benchmark/scripts/download_cslc.py \
		--start 2021-06-01 --end 2021-07-01

.PHONY: download-cslc-dry
download-cslc-dry:  ## Show what would be downloaded for the tutorial set, without fetching.
	$(RUN) python /dolphin-benchmark/scripts/download_cslc.py --dry-run

.PHONY: list-cslc
list-cslc:  ## List CSLC files currently on disk.
	$(RUN) 'ls -lh /cslc/*.h5 2>/dev/null || echo "(no CSLCs yet — run make download-cslc-tutorial)"'

# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

.PHONY: check-gpu
check-gpu:  ## Verify GPU passthrough and JAX backend.
	$(RUN) 'nvidia-smi --query-gpu=name,memory.total --format=csv,noheader && python -c "import jax; print(\"JAX backend:\", jax.default_backend(), jax.devices())"'

.PHONY: check-dolphin
check-dolphin:  ## Verify dolphin is importable from the bind-mounted source.
	$(RUN) 'pip install --no-deps --quiet -e /dolphin && python -c "import dolphin; print(dolphin.__version__)"'
