#!/usr/bin/env bash
set -euo pipefail
kit_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$kit_root"
mkdir -p runtime/container-home
export RUN_UID="$(id -u)"
export RUN_GID="$(id -g)"
if [[ "${1:-}" == build ]]; then
    shift
    exec docker compose build "$@" runner
fi
exec docker compose run --rm runner "$@"
