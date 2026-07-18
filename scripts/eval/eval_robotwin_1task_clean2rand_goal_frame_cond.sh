#!/usr/bin/env bash
# One-command RoboTwin closed-loop eval for Cosmos3-Nano-Policy-RoboTwin with
# GOAL-IMAGE-ONLY conditioning (cond=goal_frame_cond; the instruction is ignored by the model).
#
# The oracle goal image is the terminal observation of the rule-based expert
# rollout that the eval loop already runs per seed (expert_check): the client
# captures it via set_episode_ref_obs and forwards goal_head/left/right to the
# server, which lays it out (concat/cam_high) exactly as at training time.
#
# The model and the simulator live in TWO different Python environments that
# cannot share a process:
#   - Cosmos inference server -> cosmos .venv  (py3.13 / torch 2.10 / cu130)
#   - RoboTwin eval client     -> conda `robotwin` env (py3.10 / sapien)
# This script bridges them: it starts the server in the BACKGROUND (in the cosmos
# .venv, via scripts/eval/serve_robotwin_policy.sh), waits for its TCP port to come up,
# then runs the RoboTwin eval in the conda env (via the policy's own eval.sh),
# and ALWAYS tears the server down on exit (success, failure, or Ctrl-C).
#
# Usage (from anywhere) — checkpoint dir is the first argument (a default is used
# when omitted, so a bare `bash scripts/eval/eval_robotwin.sh` just works):
#   bash scripts/eval/eval_robotwin.sh [ckpt_dir] [task_name] [task_config] [seed] [eval_gpu]
# Example:
#   bash scripts/eval/eval_robotwin.sh \
#     $COSMOS_ROOT/outputs/train/cosmos3_action/action_sft/action_policy_robotwin/checkpoints/iter_000000100 \
#     place_a2b_left demo_clean 0 0
#
# Env knobs (all optional):
#   PORT                 server TCP port (default 9876); server + client are pinned to it
#   SERVE_GPU            GPU for the server (default 0)
#   ACTION_NORMALIZATION auto/none/meanstd (default auto; must match checkpoint training)
#   DOWNSAMPLE_VIDEO_FRAMES auto/true/false (default auto; must match checkpoint training)
#   VIDEO_DOWNSAMPLE_FACTOR factor used when downsample is true (default 4)
#   ROBOTWIN_NUM_STEPS  diffusion sampling steps used by the server (default 5 for this eval)
#   REPLAN_STEPS        number of actions to execute from each 32-step chunk (default 32)
#   GRIPPER_HYSTERESIS  diagnostic gripper override: after a gripper is observed
#                       closed, force later high gripper commands to stay high
#                       to test whether gripper actions were learned (default 0)
#   LABEL                eval_result/ folder name (default: derived from ckpt, e.g. iter100)
#   ROBOTWIN_DIR         RoboTwin code clone that holds policy/cosmos_policy (default below)
#   SERVER_READY_TIMEOUT seconds to wait for the model to load + bind (default 1200)
#   RUN_SMOKE=1          send one synthetic obs through the server before the real eval
#   SAVE_LOG=0           disable automatic terminal-output logging
#   LOG_ROOT             root for console/server logs (default: <cosmos>/evaluate_results/robotwin)
set -euo pipefail

COSMOS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$COSMOS_ROOT/scripts/_common.sh"
cosmos_load_local_env

# ---------------------------------------------------------------------------- #
# args: checkpoint is positional #1 (defaults to the iter_100 run below); the
# rest mirror RoboTwin eval.sh
# ---------------------------------------------------------------------------- #
DEFAULT_CKPT="$COSMOS_ROOT/outputs/train/cosmos3_action/action_sft/action_policy_robotwin_50tasks_clean2rand_goal_frame_cond/checkpoints/iter_000004000"
CKPT="${1:-$DEFAULT_CKPT}"
TASK_NAME="${2:-place_a2b_left}"
TASK_CONFIG="${3:-demo_clean}"
EVAL_SEED="${4:-${EVAL_SEED:-0}}"
EVAL_GPU="${5:-${EVAL_GPU:-0}}"

# Action de-normalization — MUST match how this checkpoint was TRAINED.
# auto reads the training config; set ACTION_NORMALIZATION=none or meanstd to force it.
ACTION_NORMALIZATION="${ACTION_NORMALIZATION:-meanstd}"
# Video temporal sampling — MUST match how this checkpoint was TRAINED.
# auto reads the training config; set DOWNSAMPLE_VIDEO_FRAMES=false/true to force it.
DOWNSAMPLE_VIDEO_FRAMES="${DOWNSAMPLE_VIDEO_FRAMES:-true}"
VIDEO_DOWNSAMPLE_FACTOR="${VIDEO_DOWNSAMPLE_FACTOR:-4}"
ROBOTWIN_POLICY_NAME="${ROBOTWIN_POLICY_NAME:-cosmos_policy_50tasks_clean2rand_goal_frame_cond}"
# Goal-image conditioning — MUST match how this checkpoint was TRAINED.
# auto reads cond/goal_layout from the training config; set COND/GOAL_LAYOUT to force.
COND="${COND:-goal_frame_cond}"
GOAL_LAYOUT="${GOAL_LAYOUT:-auto}"
GOAL_SOURCE="${GOAL_SOURCE:-oracle}"
# Action-denorm stats — MUST match training. clean2rand checkpoints were trained
# with the dataset-bundled stats; the server would otherwise fall back to the
# repo-bundled robotwin_lerobot_stats.json (WRONG for clean2rand).
ROBOTWIN_ACTION_STATS_PATH="${ROBOTWIN_ACTION_STATS_PATH:-$COSMOS_ROOT/cosmos_framework/data/vfm/action/datasets/stats/robotwin_clean_lerobot_stats.json}"
# Raw-DCP fallback experiment (goal checkpoints were trained with the goal SKU).
ROBOTWIN_EXPERIMENT="${ROBOTWIN_EXPERIMENT:-action_policy_robotwin_nano_goal}"
ROBOTWIN_NUM_STEPS="${ROBOTWIN_NUM_STEPS:-5}"
REPLAN_STEPS="${REPLAN_STEPS:-32}"
GRIPPER_HYSTERESIS="${GRIPPER_HYSTERESIS:-0}"

PORT="${PORT:-9883}"
SERVE_GPU="${SERVE_GPU:-0}"
SERVER_HOST="127.0.0.1"            # server binds 0.0.0.0; client connects locally
SERVER_READY_TIMEOUT="${SERVER_READY_TIMEOUT:-1200}"

# ---------------------------------------------------------------------------- #
# paths (COSMOS_ROOT is derived from this script's location, so cwd-independent)
# ---------------------------------------------------------------------------- #
ROBOTWIN_DIR="${ROBOTWIN_DIR:-$COSMOS_ROOT/external/RoboTwin}"
SERVE_SCRIPT="$COSMOS_ROOT/scripts/eval/serve_robotwin_policy.sh"
EVAL_SCRIPT="$ROBOTWIN_DIR/policy/cosmos_policy/eval.sh"

# RoboTwin writes results under eval_result/<task>/cosmos_policy/<config>/<LABEL>/.
# LABEL is ONLY a folder name here (the server already holds the real checkpoint;
# unlike RoboTwin's built-in policies, eval.sh does not use it to find a ckpt).
LABEL="${LABEL:-}"
if [[ -z "$LABEL" ]]; then
  base="$(basename "$CKPT")"; n="${base#iter_}"
  if [[ "$n" =~ ^[0-9]+$ ]]; then LABEL="iter$((10#$n))"; else LABEL="$base"; fi
fi

# ---------------------------------------------------------------------------- #
# logging + preflight
# ---------------------------------------------------------------------------- #
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-$COSMOS_ROOT/evaluate_results/robotwin}"
LOG_DIR="$LOG_ROOT/$LABEL/$RUN_TS"
CONSOLE_LOG="$LOG_DIR/eval_${TASK_NAME}_${RUN_TS}.log"
SERVER_LOG="$LOG_DIR/server_${TASK_NAME}_${RUN_TS}.log"

if [[ "${SAVE_LOG:-1}" == "1" ]]; then
  mkdir -p "$LOG_DIR"
  # Save the whole terminal stream, FastWAM-style, while still printing it live.
  exec > >(tee -a "$CONSOLE_LOG") 2>&1
  CONSOLE_LOG_DISPLAY="$CONSOLE_LOG"
else
  SERVER_LOG="$(mktemp -t robotwin_server_XXXXXX.log)"
  CONSOLE_LOG_DISPLAY="disabled (SAVE_LOG=0)"
fi

die() { echo "[eval_robotwin] ERROR: $*" >&2; exit 1; }
[[ -d "$CKPT" ]]        || die "checkpoint dir not found: $CKPT"
[[ -f "$SERVE_SCRIPT" ]] || die "server launcher missing: $SERVE_SCRIPT"
[[ -f "$EVAL_SCRIPT"  ]] || die "RoboTwin eval.sh missing: $EVAL_SCRIPT (is ROBOTWIN_DIR correct?)"

if (exec 3<>"/dev/tcp/$SERVER_HOST/$PORT") 2>/dev/null; then
  die "port $SERVER_HOST:$PORT is already in use before starting this run; choose another port, e.g. PORT=9877 bash scripts/eval/eval_robotwin.sh, or stop the old server first"
fi

cat <<EOF
[eval_robotwin] ----------------------------------------------------------------
  cosmos repo : $COSMOS_ROOT
  robotwin dir: $ROBOTWIN_DIR
  checkpoint  : $CKPT
  server      : host=$SERVER_HOST port=$PORT gpu=$SERVE_GPU  (log: $SERVER_LOG)
  eval        : task=$TASK_NAME config=$TASK_CONFIG label=$LABEL seed=$EVAL_SEED gpu=$EVAL_GPU
  policy args : name=$ROBOTWIN_POLICY_NAME action_norm=$ACTION_NORMALIZATION downsample_video=$DOWNSAMPLE_VIDEO_FRAMES factor=$VIDEO_DOWNSAMPLE_FACTOR num_steps=$ROBOTWIN_NUM_STEPS replan_steps=$REPLAN_STEPS gripper_hysteresis=$GRIPPER_HYSTERESIS
  goal cond   : cond=$COND goal_layout=$GOAL_LAYOUT goal_source=$GOAL_SOURCE experiment=$ROBOTWIN_EXPERIMENT
  console log : $CONSOLE_LOG_DISPLAY
[eval_robotwin] ----------------------------------------------------------------
EOF

# ---------------------------------------------------------------------------- #
# start the server in its own session/process-group so we can kill the whole tree
# (bash launcher -> python). serve_robotwin_policy.sh sources the cosmos .venv and,
# for a raw DCP, adds --experiment + ROBOTWIN_ROOT + HF_HUB_OFFLINE itself.
# ---------------------------------------------------------------------------- #
SERVER_PID=""
cleanup() {
  local code=$?
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[eval_robotwin] stopping server (pgid $SERVER_PID) …"
    kill -TERM -"$SERVER_PID" 2>/dev/null || kill -TERM "$SERVER_PID" 2>/dev/null || true
    for _ in $(seq 1 10); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 1; done
    kill -KILL -"$SERVER_PID" 2>/dev/null || true
  fi
  exit "$code"
}
trap cleanup EXIT INT TERM

setsid env CKPT="$CKPT" PORT="$PORT" SERVE_GPU="$SERVE_GPU" \
  ROBOTWIN_ACTION_NORMALIZATION="$ACTION_NORMALIZATION" \
  ROBOTWIN_DOWNSAMPLE_VIDEO_FRAMES="$DOWNSAMPLE_VIDEO_FRAMES" \
  ROBOTWIN_VIDEO_DOWNSAMPLE_FACTOR="$VIDEO_DOWNSAMPLE_FACTOR" \
  ROBOTWIN_NUM_STEPS="$ROBOTWIN_NUM_STEPS" \
  ROBOTWIN_COND="$COND" \
  ROBOTWIN_GOAL_LAYOUT="$GOAL_LAYOUT" \
  ROBOTWIN_GOAL_SOURCE="$GOAL_SOURCE" \
  ROBOTWIN_EXPERIMENT="$ROBOTWIN_EXPERIMENT" \
  ${ROBOTWIN_ACTION_STATS_PATH:+ROBOTWIN_ACTION_STATS_PATH="$ROBOTWIN_ACTION_STATS_PATH"} \
  bash -c 'cd "$1" && exec bash scripts/eval/serve_robotwin_policy.sh' _ "$COSMOS_ROOT" \
  >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

# ---------------------------------------------------------------------------- #
# wait until the port is accepting connections. the server binds ONLY after the
# model finishes loading, so an open port == ready. abort early if it dies.
# ---------------------------------------------------------------------------- #
echo "[eval_robotwin] waiting for server on $SERVER_HOST:$PORT (timeout ${SERVER_READY_TIMEOUT}s) …"
SECONDS=0
until (exec 3<>"/dev/tcp/$SERVER_HOST/$PORT") 2>/dev/null; do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[eval_robotwin] server exited during startup. Last log lines:" >&2
    tail -n 60 "$SERVER_LOG" >&2 || true
    exit 1
  fi
  if (( SECONDS >= SERVER_READY_TIMEOUT )); then
    echo "[eval_robotwin] timed out after ${SERVER_READY_TIMEOUT}s. Last log lines:" >&2
    tail -n 60 "$SERVER_LOG" >&2 || true
    exit 1
  fi
  sleep 3
done
# If a racing process grabbed the port, our server can fail right after model load.
# Give it a moment to surface bind errors before trusting the open port.
sleep 1
if ! kill -0 "$SERVER_PID" 2>/dev/null; then
  echo "[eval_robotwin] server exited when binding the port. Last log lines:" >&2
  tail -n 60 "$SERVER_LOG" >&2 || true
  exit 1
fi
echo "[eval_robotwin] server is up (after ${SECONDS}s)."

# ---------------------------------------------------------------------------- #
# optional: prove the model path end-to-end with one synthetic obs (no sim).
# ---------------------------------------------------------------------------- #
if [[ "${RUN_SMOKE:-0}" == "1" ]]; then
  echo "[eval_robotwin] running smoke client …"
  ( cd "$COSMOS_ROOT" && export LD_LIBRARY_PATH= \
      && "$COSMOS_ROOT/.venv/bin/python" scripts/eval/smoke_robotwin_client.py --host "$SERVER_HOST" --port "$PORT" --cond "$COND" )
fi

# ---------------------------------------------------------------------------- #
# run the real eval in the conda `robotwin` env. eval.sh activates the env, sets
# the Blackwell Vulkan/OIDN vars, cd's to RoboTwin root, and drives the sim.
# COSMOS_SERVER_HOST/PORT pin the client to THIS server (overrides deploy_policy.yml).
# ---------------------------------------------------------------------------- #
echo "[eval_robotwin] launching RoboTwin eval …"
set +e
COSMOS_SERVER_HOST="$SERVER_HOST" COSMOS_SERVER_PORT="$PORT" COSMOS_POLICY_NAME="$ROBOTWIN_POLICY_NAME" \
COSMOS_POLICY_REPLAN_STEPS="$REPLAN_STEPS" COSMOS_GRIPPER_HYSTERESIS="$GRIPPER_HYSTERESIS" \
COSMOS_GOAL_COND=1 COSMOS_GOAL_SOURCE="$GOAL_SOURCE" \
  bash "$EVAL_SCRIPT" "$TASK_NAME" "$TASK_CONFIG" "$LABEL" "$EVAL_SEED" "$EVAL_GPU"
EVAL_RC=$?
set -e

if [[ "$EVAL_RC" -eq 0 ]]; then
  echo "[eval_robotwin] eval finished OK. Results under $ROBOTWIN_DIR/eval_result/$TASK_NAME/$ROBOTWIN_POLICY_NAME/$TASK_CONFIG/$LABEL/"
  echo "[eval_robotwin] console log saved to: $CONSOLE_LOG_DISPLAY"
  echo "[eval_robotwin] server log saved to: $SERVER_LOG"
else
  echo "[eval_robotwin] eval exited with code $EVAL_RC" >&2
  echo "[eval_robotwin] console log: $CONSOLE_LOG_DISPLAY" >&2
  echo "[eval_robotwin] server log: $SERVER_LOG" >&2
fi
exit "$EVAL_RC"
