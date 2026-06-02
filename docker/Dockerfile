# GPU-enabled benchmark image for dolphin.
#
# Differs from upstream dolphin/docker/Dockerfile in three ways:
#   1. CUDA base image (upstream is CPU-only ubuntu:22.04)
#   2. JAX installed with CUDA 12 wheels via pip
#   3. Profiling and benchmark tooling: asv, py-spy, pynvml, Nsight Systems (from CUDA devel image)
#
# dolphin itself is *not* installed at build time. Instead the host's dolphin
# clone is bind-mounted at /dolphin and `pip install -e /dolphin` is performed
# at container start (or manually), so source edits take effect without rebuild.

ARG CUDA_VERSION=12.6.3
ARG UBUNTU_VERSION=22.04
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}

LABEL org.opencontainers.image.description="GPU benchmarking environment for isce-framework/dolphin"
LABEL org.opencontainers.image.source="https://github.com/s-sasaki-earthsea-wizard/dolphin-benchmark"

ARG DEBIAN_FRONTEND=noninteractive

# System packages: build tools (some conda packages compile), git, curl for micromamba,
# plus a few QoL utilities for interactive use inside the container.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        wget \
        git \
        bzip2 \
        build-essential \
        less \
        vim \
        sudo \
    && rm -rf /var/lib/apt/lists/*

# Install micromamba (same approach as upstream Dockerfile, but unpinned to keep updating easy).
ENV MAMBA_ROOT_PREFIX=/opt/conda
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
        | tar -xj -C /usr/local/bin --strip-components=1 bin/micromamba

# User account matching host UID/GID so bind-mounted files stay owner-correct.
ARG USER_ID=1000
ARG GROUP_ID=1000
ARG USERNAME=dolphin
RUN groupadd -g ${GROUP_ID} ${USERNAME} \
    && useradd -m -u ${USER_ID} -g ${GROUP_ID} -s /bin/bash ${USERNAME} \
    && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers \
    && mkdir -p /opt/conda \
    && chown -R ${USERNAME}:${USERNAME} /opt/conda

USER ${USERNAME}
ENV PATH=/opt/conda/envs/dolphin-env/bin:/opt/conda/bin:/usr/local/cuda/bin:$PATH

# Build the conda env from a snapshot of dolphin's conda-env.yml.
# Snapshot is kept here intentionally so the build context stays tiny — see
# README in this directory for the sync procedure.
COPY --chown=${USERNAME}:${USERNAME} conda-env.yml /tmp/conda-env.yml
RUN micromamba create -n dolphin-env -y -f /tmp/conda-env.yml \
    && micromamba clean --all --yes

# Make the conda env the default shell environment.
SHELL ["/bin/bash", "-lc"]
ENV CONDA_DEFAULT_ENV=dolphin-env

# JAX with CUDA 12 support (pip — conda-forge jax has no CUDA wheels).
# `--upgrade` overrides the CPU-only jax pulled in by conda-forge with the
# matching jaxlib+CUDA wheel pair from PyPI. No version pin: let pip resolve
# a self-consistent (jax, jaxlib) pair. Pin later in CI once we lock numbers.
RUN micromamba run -n dolphin-env pip install --no-cache-dir --upgrade \
        "jax[cuda12]" \
        snaphu

# Benchmarking + profiling tools.
RUN micromamba run -n dolphin-env pip install --no-cache-dir \
        asv \
        py-spy \
        pynvml \
        ipython

# Note: nsys (Nsight Systems CLI) ships with the CUDA devel image at
# /usr/local/cuda/bin/nsys. No extra install needed.

# Set up bashrc so the dolphin-env is implicitly active.
# PATH already prepends the env's bin (see ENV above), so we don't need
# `micromamba activate` here — its CLI also changed between 1.x and 2.x, and
# activation only adds prompt cosmetics over what PATH already provides.
RUN echo 'eval "$(micromamba shell hook --shell bash)" 2>/dev/null || true' >> /home/${USERNAME}/.bashrc \
    && echo 'micromamba activate dolphin-env 2>/dev/null || true' >> /home/${USERNAME}/.bashrc

# Quiet JAX's default pre-allocation behavior so multiple processes can share the GPU.
# Users can override at runtime if they want the default fast-path.
ENV XLA_PYTHON_CLIENT_PREALLOCATE=false
ENV XLA_PYTHON_CLIENT_MEM_FRACTION=0.75

# Bind-mount targets (/dolphin, /dolphin-benchmark, /data) are created
# automatically by Docker at mount time. /work is created by WORKDIR below.
WORKDIR /work

# Default entrypoint: interactive login bash, which activates the env via .bashrc.
# Override with `docker run ... <cmd>` for one-shot invocations.
CMD ["bash", "-l"]
