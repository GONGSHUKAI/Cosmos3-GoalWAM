#!/usr/bin/env bash
set -euo pipefail

COSMOS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$COSMOS_ROOT/scripts/_common.sh"
cosmos_load_local_env

INTEGRATION_ROOT="$COSMOS_ROOT/integrations/robotwin"
UPSTREAM_URL="${ROBOTWIN_UPSTREAM_URL:-https://github.com/RoboTwin-Platform/RoboTwin.git}"
UPSTREAM_COMMIT="$(tr -d '[:space:]' < "$INTEGRATION_ROOT/UPSTREAM_COMMIT")"
ROBOTWIN_DIR="${ROBOTWIN_DIR:-$COSMOS_ROOT/external/RoboTwin}"

if [[ ! -d "$ROBOTWIN_DIR/.git" ]]; then
  mkdir -p "$(dirname "$ROBOTWIN_DIR")"
  git clone "$UPSTREAM_URL" "$ROBOTWIN_DIR"
fi

unexpected_tracked="$(git -C "$ROBOTWIN_DIR" diff --name-only | grep -v '^script/eval_policy.py$' || true)"
if [[ -n "$unexpected_tracked" ]]; then
  echo "ERROR: refusing to overwrite tracked RoboTwin changes:" >&2
  echo "$unexpected_tracked" >&2
  exit 1
fi

if ! git -C "$ROBOTWIN_DIR" cat-file -e "$UPSTREAM_COMMIT^{commit}" 2>/dev/null; then
  git -C "$ROBOTWIN_DIR" fetch origin "$UPSTREAM_COMMIT"
fi
if [[ "$(git -C "$ROBOTWIN_DIR" rev-parse HEAD)" != "$UPSTREAM_COMMIT" ]]; then
  if [[ -n "$(git -C "$ROBOTWIN_DIR" diff --name-only)" ]]; then
    echo "ERROR: cannot change RoboTwin revision while tracked changes are present" >&2
    exit 1
  fi
  git -C "$ROBOTWIN_DIR" checkout --detach "$UPSTREAM_COMMIT"
fi

PATCH="$INTEGRATION_ROOT/patches/0001-cosmos-goal-evaluation.patch"
if git -C "$ROBOTWIN_DIR" apply --unidiff-zero --reverse --check "$PATCH" 2>/dev/null; then
  echo "[prepare_robotwin] patch already applied"
else
  git -C "$ROBOTWIN_DIR" apply --unidiff-zero --check "$PATCH"
  git -C "$ROBOTWIN_DIR" apply --unidiff-zero "$PATCH"
fi

cp -a "$INTEGRATION_ROOT/overlay/." "$ROBOTWIN_DIR/"
echo "[prepare_robotwin] ready: $ROBOTWIN_DIR @ $UPSTREAM_COMMIT"
