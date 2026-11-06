ARG UBUNTU_IMAGE=ubuntu:24.04@sha256:fe81560990fae86dc4743202b2cb4d3df98d0bcf949173cecef15ce1127ef852
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.11.29@sha256:eb2843a1e56fd9e30c7276ce1a52cba86e64c7b385f5e3279a0e08e02dd058fc
FROM ${UV_IMAGE} AS uv
FROM ${UBUNTU_IMAGE}
ARG RUN_UID=1000
ARG RUN_GID=1000
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash ca-certificates curl git git-lfs python3 build-essential pkg-config \
    ffmpeg unzip rsync libgl1 libegl1 libglib2.0-0 libglx0 libglvnd0 \
    libvulkan1 vulkan-tools libx11-6 libxi6 libxrandr2 libxinerama1 \
    libxcursor1 libxkbcommon0 libxrender1 libxt6 libsm6 libice6 \
    libasound2t64 libgomp1 libnuma1 libnss3 libxdamage1 libxcomposite1 \
    libgbm1 libgtk-3-0 imagemagick libmagickwand-dev && \
    rm -rf /var/lib/apt/lists/*
COPY --from=uv /uv /uvx /usr/local/bin/
# Ubuntu's default uid 1000 account is replaced with a host-matching user.
RUN if id ubuntu >/dev/null 2>&1; then userdel -r ubuntu; fi; \
    (getent group ${RUN_GID} >/dev/null || groupadd --gid ${RUN_GID} runner) && \
    useradd --non-unique --create-home --uid ${RUN_UID} --gid ${RUN_GID} --shell /bin/bash runner
USER runner
WORKDIR /workspace
ENTRYPOINT ["/workspace/bench"]
CMD ["doctor"]
