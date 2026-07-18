#!/usr/bin/env bash
# Benchmark Cosmos3-Nano-Policy-RoboTwin action inference latency for the
# RoboTwin ablation checkpoints. This launches the policy server, sends
# synthetic observations through the same TCP protocol used by RoboTwin eval,
# prints latency stats, then stops the server before the next checkpoint.

# Usage:
# RUN_SET=all SERVE_GPU=0 bash scripts/eval/benchmark_robotwin_downsample_latency.sh
# RUN_SET=baseline SERVE_GPU=0 bash scripts/eval/benchmark_robotwin_downsample_latency.sh
# RUN_SET=action_norm SERVE_GPU=0 bash scripts/eval/benchmark_robotwin_downsample_latency.sh
# RUN_SET=downsample SERVE_GPU=0 bash scripts/eval/benchmark_robotwin_downsample_latency.sh
# RUN_SET=downsample_action_norm SERVE_GPU=0 bash scripts/eval/benchmark_robotwin_downsample_latency.sh

set -euo pipefail

COSMOS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$COSMOS_ROOT/scripts/_common.sh"
cosmos_load_local_env
SERVE_SCRIPT="$COSMOS_ROOT/scripts/eval/serve_robotwin_policy.sh"
CLIENT_SCRIPT="$COSMOS_ROOT/scripts/eval/benchmark_robotwin_policy_latency.py"

: "${SERVE_GPU:=0}"
: "${SERVER_HOST:=127.0.0.1}"
: "${SERVER_READY_TIMEOUT:=1200}"
: "${WARMUP:=3}"
: "${ITERS:=20}"
: "${HEIGHT:=240}"
: "${WIDTH:=320}"
: "${LOG_ROOT:=$COSMOS_ROOT/evaluate_results/robotwin_latency}"
: "${RUN_SET:=all}"  # all | baseline | action_norm | downsample | downsample_action_norm

RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="$LOG_ROOT/$RUN_TS"
mkdir -p "$LOG_DIR"

RUN_LABELS=()
case "$RUN_SET" in
  all)
    RUN_LABELS=(baseline action_norm downsample downsample_action_norm)
    ;;
  baseline|action_norm|downsample|downsample_action_norm)
    RUN_LABELS=("$RUN_SET")
    ;;
  *)
    echo "ERROR: RUN_SET must be one of: all, baseline, action_norm, downsample, downsample_action_norm" >&2
    exit 1
    ;;
esac

server_pid=""
cleanup_server() {
  if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
    echo "[benchmark] stopping server pgid=$server_pid"
    kill -TERM -"$server_pid" 2>/dev/null || kill -TERM "$server_pid" 2>/dev/null || true
    for _ in $(seq 1 20); do
      kill -0 "$server_pid" 2>/dev/null || break
      sleep 1
    done
    kill -KILL -"$server_pid" 2>/dev/null || true
  fi
  server_pid=""
}

cleanup_all() {
  cleanup_server
}
trap cleanup_all EXIT INT TERM

port_is_busy() {
  local host="$1"
  local port="$2"
  python - "$host" "$port" <<'PY'
import socket
import sys

host, port = sys.argv[1], int(sys.argv[2])
s = socket.socket()
s.settimeout(0.5)
try:
    s.connect((host, port))
except OSError:
    raise SystemExit(1)
else:
    raise SystemExit(0)
finally:
    s.close()
PY
}

wait_for_server() {
  local host="$1"
  local port="$2"
  local log_file="$3"
  echo "[benchmark] waiting for server on $host:$port timeout=${SERVER_READY_TIMEOUT}s"
  SECONDS=0
  until port_is_busy "$host" "$port"; do
    if ! kill -0 "$server_pid" 2>/dev/null; then
      echo "[benchmark] server exited during startup. Last log lines:" >&2
      tail -n 80 "$log_file" >&2 || true
      return 1
    fi
    if (( SECONDS >= SERVER_READY_TIMEOUT )); then
      echo "[benchmark] timed out waiting for server. Last log lines:" >&2
      tail -n 80 "$log_file" >&2 || true
      return 1
    fi
    sleep 3
  done
  sleep 1
  if ! kill -0 "$server_pid" 2>/dev/null; then
    echo "[benchmark] server exited after opening port. Last log lines:" >&2
    tail -n 80 "$log_file" >&2 || true
    return 1
  fi
  echo "[benchmark] server ready after ${SECONDS}s"
}

run_one() {
  local label="$1"
  local ckpt=""
  local port=""
  local action_norm=""
  local downsample_video=""

  case "$label" in
    baseline)
      ckpt="${BASELINE_CKPT:-$COSMOS_ROOT/outputs/train/cosmos3_action/action_sft/action_policy_robotwin/checkpoints/iter_000004000}"
      port="${BASELINE_PORT:-9876}"
      action_norm="none"
      downsample_video="false"
      ;;
    action_norm)
      ckpt="${ACTION_NORM_CKPT:-$COSMOS_ROOT/outputs/train/cosmos3_action/action_sft/action_policy_robotwin_action_norm/checkpoints/iter_000004000}"
      port="${ACTION_NORM_PORT:-9877}"
      action_norm="meanstd"
      downsample_video="false"
      ;;
    downsample)
      ckpt="${DOWNSAMPLE_CKPT:-$COSMOS_ROOT/outputs/train/cosmos3_action/action_sft/action_policy_robotwin_downsample_video/checkpoints/iter_000004000}"
      port="${DOWNSAMPLE_PORT:-9878}"
      action_norm="none"
      downsample_video="true"
      ;;
    downsample_action_norm)
      ckpt="${DOWNSAMPLE_ACTION_NORM_CKPT:-$COSMOS_ROOT/outputs/train/cosmos3_action/action_sft/action_policy_robotwin_downsample_video_action_norm/checkpoints/iter_000004000}"
      port="${DOWNSAMPLE_ACTION_NORM_PORT:-9879}"
      action_norm="meanstd"
      downsample_video="true"
      ;;
    *)
      echo "ERROR: unknown benchmark label: $label" >&2
      return 1
      ;;
  esac

  local server_log="$LOG_DIR/server_${label}.log"
  local client_log="$LOG_DIR/client_${label}.log"

  echo "[benchmark] ------------------------------------------------------------"
  echo "[benchmark] label=$label"
  echo "[benchmark] checkpoint=$ckpt"
  echo "[benchmark] host=$SERVER_HOST port=$port serve_gpu=$SERVE_GPU"
  echo "[benchmark] action_norm=$action_norm downsample_video=$downsample_video factor=4"
  echo "[benchmark] warmup=$WARMUP iters=$ITERS image=${HEIGHT}x${WIDTH}"
  echo "[benchmark] server_log=$server_log"
  echo "[benchmark] client_log=$client_log"

  [[ -d "$ckpt" ]] || {
    echo "ERROR: checkpoint dir not found: $ckpt" >&2
    return 1
  }
  [[ -f "$SERVE_SCRIPT" ]] || {
    echo "ERROR: server launcher missing: $SERVE_SCRIPT" >&2
    return 1
  }
  [[ -f "$CLIENT_SCRIPT" ]] || {
    echo "ERROR: timing client missing: $CLIENT_SCRIPT" >&2
    return 1
  }
  if port_is_busy "$SERVER_HOST" "$port"; then
    echo "ERROR: $SERVER_HOST:$port is already in use" >&2
    return 1
  fi

  server_env=(
    "CKPT=$ckpt"
    "PORT=$port"
    "SERVE_GPU=$SERVE_GPU"
    "ROBOTWIN_ACTION_NORMALIZATION=$action_norm"
    "ROBOTWIN_DOWNSAMPLE_VIDEO_FRAMES=$downsample_video"
    "ROBOTWIN_VIDEO_DOWNSAMPLE_FACTOR=4"
  )
  if [[ -n "${ROBOTWIN_ACTION_STATS_PATH:-}" ]]; then
    server_env+=("ROBOTWIN_ACTION_STATS_PATH=$ROBOTWIN_ACTION_STATS_PATH")
  fi

  setsid env "${server_env[@]}" bash -c 'cd "$1" && exec bash scripts/eval/serve_robotwin_policy.sh' _ "$COSMOS_ROOT" \
    >"$server_log" 2>&1 &
  server_pid=$!

  wait_for_server "$SERVER_HOST" "$port" "$server_log"

  (
    cd "$COSMOS_ROOT"
    export LD_LIBRARY_PATH=
    "$COSMOS_ROOT/.venv/bin/python" "$CLIENT_SCRIPT" \
      --host "$SERVER_HOST" \
      --port "$port" \
      --height "$HEIGHT" \
      --width "$WIDTH" \
      --warmup "$WARMUP" \
      --iters "$ITERS" \
      --label "$label"
  ) | tee "$client_log"

  cleanup_server
}

for label in "${RUN_LABELS[@]}"; do
  run_one "$label"
done

echo "[benchmark] all requested latency runs finished. Logs: $LOG_DIR"
