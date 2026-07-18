#!/usr/bin/env bash
# Download the RoboTwin LeRobot v3.0 dataset, auto-retrying on interruption
# until the command exits successfully (huggingface-cli resumes partial files).
#
# Usage:
#   bash scripts/download/download_robotwin_lerobot_v3.0.sh
#
# Stop it any time with Ctrl-C; re-run to resume where it left off.

set -u

# --- Environment -------------------------------------------------------------
# Activate the project venv so the right Hugging Face CLI is on PATH.
COSMOS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$COSMOS_ROOT"
source "$COSMOS_ROOT/scripts/_common.sh"
cosmos_load_local_env
cosmos_activate_venv
# Set HF_TOKEN here (or `export` it in your shell) if the dataset is gated.
# export HF_TOKEN=hf_xxx

# More tolerant network timeouts for large, long-running pulls.
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-60}"

# --- Config ------------------------------------------------------------------
REPO="hxma/RoboTwin-LeRobot-v3.0"
: "${ROBOTWIN_DATA_ROOT:?Set ROBOTWIN_DATA_ROOT in config/local.env or the shell environment}"
LOCAL_DIR="${LOCAL_DIR:-$ROBOTWIN_DATA_ROOT/robotwin2.0_lerobot_v3.0}"
RETRY_SLEEP="${RETRY_SLEEP:-5}"   # seconds to wait between attempts

mkdir -p "${LOCAL_DIR}"

# --- Retry loop --------------------------------------------------------------
attempt=0
while true; do
    attempt=$((attempt + 1))
    echo "=============================================================="
    echo "[$(date '+%F %T')] Download attempt #${attempt} for ${REPO}"
    echo "=============================================================="

    huggingface-cli download \
        --repo-type dataset \
        "${REPO}" \
        --local-dir "${LOCAL_DIR}" \
        --local-dir-use-symlinks False
    status=$?

    if [ "${status}" -eq 0 ]; then
        echo "[$(date '+%F %T')] Download completed successfully after ${attempt} attempt(s)."
        break
    fi

    echo "[$(date '+%F %T')] Attempt #${attempt} failed (exit ${status}). Retrying in ${RETRY_SLEEP}s..."
    sleep "${RETRY_SLEEP}"
done
