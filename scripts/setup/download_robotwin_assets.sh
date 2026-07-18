#!/usr/bin/env bash
set -euo pipefail

COSMOS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$COSMOS_ROOT/scripts/_common.sh"
cosmos_load_local_env
ROBOTWIN_DIR="${ROBOTWIN_DIR:-$COSMOS_ROOT/external/RoboTwin}"

if [[ ! -f "$ROBOTWIN_DIR/script/_download_assets.sh" ]]; then
  echo "ERROR: RoboTwin source is not prepared at $ROBOTWIN_DIR" >&2
  exit 1
fi

cd "$ROBOTWIN_DIR"
bash script/_download_assets.sh
