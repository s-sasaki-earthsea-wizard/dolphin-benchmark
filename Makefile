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

# `dev` is named explicitly: the unwrap service (issue #13) builds FROM the
# dev image, and compose has no notion of that ordering.
.PHONY: build
build:  ## Build the GPU dev image.
	cd docker && export USER_ID=$$(id -u) GROUP_ID=$$(id -g) && \
		docker compose --env-file ../.env build dev

.PHONY: rebuild
rebuild:  ## Rebuild the image from scratch (no cache).
	cd docker && export USER_ID=$$(id -u) GROUP_ID=$$(id -g) && \
		docker compose --env-file ../.env build --no-cache dev

.PHONY: build-unwrap
build-unwrap: build  ## Build the unwrapper-comparison image (isce3/tophu + whirlwind).
	cd docker && export USER_ID=$$(id -u) GROUP_ID=$$(id -g) && \
		docker compose --env-file ../.env build unwrap

.PHONY: shell
shell:  ## Interactive shell inside the dev container.
	$(RUN)

.PHONY: shell-unwrap
shell-unwrap:  ## Interactive shell inside the unwrapper-comparison container.
	SERVICE=unwrap $(RUN)

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
# ASV baseline (issue #2)
# ---------------------------------------------------------------------------
#
# Configs are generated from dolphin's asv.conf.json — never edited by hand.
# GPU and CPU use physically separate results trees (results/asv-baseline/gpu
# and /cpu) because asv's `existing` env type ignores the requirement matrix,
# so env names cannot distinguish the backends. See asv/generate-confs.sh.

ASV_GPU_CONF := /dolphin-benchmark/asv/asv-existing-gpu.conf.json
ASV_CPU_CONF := /dolphin-benchmark/asv/asv-existing-cpu.conf.json
# Upstream benchmark.yml sets only these three; NUMBA_NUM_THREADS stays unset
# for CI-equivalent runs (R4/R5) and is a separate decision for R2/R3.
ASV_THREADS := OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1

.PHONY: asv-confs
asv-confs:  ## Regenerate asv configs from dolphin's asv.conf.json.
	./asv/generate-confs.sh

# --set-commit-hash: without it, asv silently skips saving result JSONs for
# `existing` environments (it cannot infer which commit the env represents).
.PHONY: asv-smoke
asv-smoke: asv-confs  ## R1 smoke: asv run --quick, GPU, existing env.
	$(RUN) 'pip install --no-deps --quiet -e /dolphin && \
		export $(ASV_THREADS) JAX_PLATFORMS=cuda && \
		asv machine --yes --config $(ASV_GPU_CONF) && \
		asv run --quick --show-stderr --python=same \
			--set-commit-hash $$(git -C /dolphin rev-parse HEAD) \
			--config $(ASV_GPU_CONF)'

# Baseline runs restrict to the benchmarks that work. Both exclusions are
# upstream dolphin bugs of the same kind — the suite calls an API that changed
# underneath it: ShpBenchmark on HALF_WINDOW["y"] (#203) and
# SingleMinistackBenchmark on run_wrapped_phase_sequential's signature (#334).
# Both recorded as `failed` by the R1 smoke, no point re-measuring here; both
# are fixed on the fork's fix/asv-benchmark-suite branch.
#
# Until then this is a plain override, e.g. to measure one of them:
#   make asv-baseline-gpu ASV_WORKING_BENCHES=SingleMinistackBenchmark
#
# NUMBA_NUM_THREADS stays unset: none of the working benchmarks touch numba,
# and leaving it unset matches upstream CI.
ASV_WORKING_BENCHES := "CovarianceBenchmark|PhaseLinkingBenchmark"

.PHONY: asv-baseline-gpu
asv-baseline-gpu: asv-confs  ## R2: full baseline run on GPU (existing env).
	$(RUN) 'pip install --no-deps --quiet -e /dolphin && \
		export $(ASV_THREADS) JAX_PLATFORMS=cuda && \
		asv machine --yes --config $(ASV_GPU_CONF) && \
		asv run --show-stderr --python=same \
			--set-commit-hash $$(git -C /dolphin rev-parse HEAD) \
			-b $(ASV_WORKING_BENCHES) \
			--config $(ASV_GPU_CONF)'

.PHONY: asv-baseline-cpu
asv-baseline-cpu: asv-confs  ## R3: full baseline run on CPU (JAX_PLATFORMS=cpu).
	$(RUN) 'pip install --no-deps --quiet -e /dolphin && \
		export $(ASV_THREADS) JAX_PLATFORMS=cpu && \
		asv machine --yes --config $(ASV_CPU_CONF) && \
		asv run --show-stderr --python=same \
			--set-commit-hash $$(git -C /dolphin rev-parse HEAD) \
			-b $(ASV_WORKING_BENCHES) \
			--config $(ASV_CPU_CONF)'

# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

.PHONY: check-gpu
check-gpu:  ## Verify GPU passthrough and JAX backend.
	$(RUN) 'nvidia-smi --query-gpu=name,memory.total --format=csv,noheader && python -c "import jax; print(\"JAX backend:\", jax.default_backend(), jax.devices())"'

# `from dolphin import io` below is deliberate: the bare top-level import is
# lazy and once passed while dolphin.io was broken by a stale opera-utils.
.PHONY: check-dolphin
check-dolphin:  ## Verify dolphin is importable from the bind-mounted source.
	$(RUN) 'pip install --no-deps --quiet -e /dolphin && python -c "import dolphin; from dolphin import io, shp; print(dolphin.__version__)"'
