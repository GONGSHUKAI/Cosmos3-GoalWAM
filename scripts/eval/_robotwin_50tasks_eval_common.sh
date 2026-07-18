#!/usr/bin/env bash
# Shared 50-task RoboTwin eval orchestrator (sourced by eval_robotwin_50tasks_*.sh).
#
# Evaluates ONE checkpoint on ALL 50 RoboTwin tasks (or a subset) on the clean or
# randomized task config, parallelized across GPUs: each GPU gets ONE persistent
# policy server (model loaded once) plus a sequential stream of sim evals on the
# same GPU; tasks are pulled from a shared flock-guarded queue so fast GPUs pick
# up more tasks (FastWAM run_robotwin_manager.py pattern, adapted to the
# server/client split).
#
# The sourcing variant script must define:
#   VARIANT                    run-id, e.g. 50tasks_clean2rand_text_goal_frame_cond
#   DEFAULT_CKPT               checkpoint dir used when arg 1 is omitted
#   ROBOTWIN_POLICY_NAME       RoboTwin policy dir (client side)
#   ROBOTWIN_EXPERIMENT        serve fallback experiment for raw-DCP checkpoints
#   ROBOTWIN_ACTION_STATS_PATH stats matching how the checkpoint was TRAINED
#   BASE_PORT                  first server port (workers use BASE_PORT+i)
#   COSMOS_GOAL_COND           1 for goal-conditioned variants, else 0
#   COND / GOAL_LAYOUT / GOAL_SOURCE   (goal variants; defaults auto/auto/oracle)
#
# Usage (of the sourcing script):
#   bash eval_robotwin_50tasks_<variant>.sh [ckpt_dir] [clean|random]
# Env knobs:
#   GPUS        comma-separated GPU ids, one worker per GPU (default 0,1,...,7)
#   TEST_NUM    episodes per task (default 100; use e.g. 20 for quick sweeps)
#   TASKS       space-separated task subset (default: all 50 from _eval_step_limit.yml)
#   LABEL       eval_result/ folder label (default derived from ckpt, e.g. iter4000)
#   EVAL_SEED   RoboTwin seed (default 0)
#   REPLAN_STEPS / ROBOTWIN_NUM_STEPS / SERVER_READY_TIMEOUT / ROBOTWIN_DIR
#   DRY_RUN=1   print the plan (tasks, workers, commands) without launching
set -uo pipefail   # NO -e: a failed task is recorded and the run continues

COSMOS_ROOT="${COSMOS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[1]}")/../.." && pwd)}"
source "$COSMOS_ROOT/scripts/_common.sh"
cosmos_load_local_env

# ---------------------------------------------------------------------------- #
# args + variant contract
# ---------------------------------------------------------------------------- #
for var in VARIANT DEFAULT_CKPT ROBOTWIN_POLICY_NAME ROBOTWIN_EXPERIMENT \
           ROBOTWIN_ACTION_STATS_PATH BASE_PORT COSMOS_GOAL_COND; do
  [[ -n "${!var:-}" ]] || { echo "ERROR: variant script must set $var" >&2; exit 1; }
done
COND="${COND:-auto}"
GOAL_LAYOUT="${GOAL_LAYOUT:-auto}"
GOAL_SOURCE="${GOAL_SOURCE:-oracle}"

CKPT="${1:-$DEFAULT_CKPT}"
PHASE="${2:-clean}"
case "$PHASE" in
  clean)              TASK_CONFIG="demo_clean" ;;
  random|randomized)  TASK_CONFIG="demo_randomized" ;;
  demo_*)             TASK_CONFIG="$PHASE" ;;
  *) echo "ERROR: phase must be clean|random (or a literal demo_* config), got '$PHASE'" >&2; exit 1 ;;
esac

GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
IFS=',' read -ra GPU_ARR <<< "$GPUS"
NUM_WORKERS="${#GPU_ARR[@]}"
TEST_NUM="${TEST_NUM:-100}"
EVAL_SEED="${EVAL_SEED:-0}"
REPLAN_STEPS="${REPLAN_STEPS:-32}"
ROBOTWIN_NUM_STEPS="${ROBOTWIN_NUM_STEPS:-5}"
SERVER_READY_TIMEOUT="${SERVER_READY_TIMEOUT:-1800}"
ACTION_NORMALIZATION="${ACTION_NORMALIZATION:-meanstd}"
DOWNSAMPLE_VIDEO_FRAMES="${DOWNSAMPLE_VIDEO_FRAMES:-true}"
VIDEO_DOWNSAMPLE_FACTOR="${VIDEO_DOWNSAMPLE_FACTOR:-4}"
SERVER_HOST="127.0.0.1"

ROBOTWIN_DIR="${ROBOTWIN_DIR:-$COSMOS_ROOT/external/RoboTwin}"
SERVE_SCRIPT="$COSMOS_ROOT/scripts/eval/serve_robotwin_policy.sh"
EVAL_SCRIPT="$ROBOTWIN_DIR/policy/cosmos_policy/eval.sh"
STEP_LIMIT_FILE="$ROBOTWIN_DIR/task_config/_eval_step_limit.yml"

die() { echo "[eval_50tasks] ERROR: $*" >&2; exit 1; }
[[ -d "$CKPT" ]]            || die "checkpoint dir not found: $CKPT"
[[ -f "$SERVE_SCRIPT" ]]    || die "server launcher missing: $SERVE_SCRIPT"
[[ -f "$EVAL_SCRIPT" ]]     || die "RoboTwin eval.sh missing: $EVAL_SCRIPT"
[[ -f "$STEP_LIMIT_FILE" ]] || die "task list missing: $STEP_LIMIT_FILE"

# LABEL defaults to iter<N> from the checkpoint dir name
LABEL="${LABEL:-}"
if [[ -z "$LABEL" ]]; then
  base="$(basename "$CKPT")"; n="${base#iter_}"
  if [[ "$n" =~ ^[0-9]+$ ]]; then LABEL="iter$((10#$n))"; else LABEL="$base"; fi
fi

# Task list: TASKS env override, else all keys of _eval_step_limit.yml (in order)
if [[ -n "${TASKS:-}" ]]; then
  read -ra TASK_ARR <<< "$TASKS"
else
  mapfile -t TASK_ARR < <("$COSMOS_ROOT/.venv/bin/python" - "$STEP_LIMIT_FILE" <<'PYEOF'
import sys, yaml
seen = set()
for k in yaml.safe_load(open(sys.argv[1])):
    if k not in seen:
        seen.add(k); print(k)
PYEOF
)
fi
[[ "${#TASK_ARR[@]}" -gt 0 ]] || die "empty task list"

# ---------------------------------------------------------------------------- #
# run dir, queue, logging
# ---------------------------------------------------------------------------- #
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="$COSMOS_ROOT/evaluate_results/robotwin_50tasks/${VARIANT}/${LABEL}_${TASK_CONFIG}_${RUN_TS}"
mkdir -p "$RUN_DIR/logs"
QUEUE_FILE="$RUN_DIR/.task_queue"
QUEUE_LOCK="$RUN_DIR/.task_queue.lock"
RESULTS_CSV="$RUN_DIR/results.csv"
FAILED_TXT="$RUN_DIR/failed_tasks.txt"
printf '%s\n' "${TASK_ARR[@]}" > "$QUEUE_FILE"
: > "$QUEUE_LOCK"
echo "task,task_config,success_rate,episodes,gpu,eval_rc" > "$RESULTS_CSV"
: > "$FAILED_TXT"

exec > >(tee -a "$RUN_DIR/manager.log") 2>&1

cat <<EOF
[eval_50tasks] ----------------------------------------------------------------
  variant     : $VARIANT   (cond=$COND goal_layout=$GOAL_LAYOUT goal_cond=$COSMOS_GOAL_COND)
  checkpoint  : $CKPT   (label=$LABEL)
  task config : $TASK_CONFIG   episodes/task=$TEST_NUM   seed=$EVAL_SEED
  tasks       : ${#TASK_ARR[@]}
  workers     : $NUM_WORKERS on GPUs [$GPUS], server ports $BASE_PORT..$((BASE_PORT + NUM_WORKERS - 1))
  policy      : $ROBOTWIN_POLICY_NAME   stats=$(basename "$ROBOTWIN_ACTION_STATS_PATH")
  run dir     : $RUN_DIR
[eval_50tasks] ----------------------------------------------------------------
EOF

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[eval_50tasks] DRY_RUN=1 — plan only, nothing launched. Task list:"
  printf '  %s\n' "${TASK_ARR[@]}"
  for i in "${!GPU_ARR[@]}"; do
    echo "  worker $i: GPU ${GPU_ARR[$i]}, server port $((BASE_PORT + i)), pulls tasks from the shared queue"
  done
  exit 0
fi

# ---------------------------------------------------------------------------- #
# queue + result helpers (flock-guarded, shared across workers)
# ---------------------------------------------------------------------------- #
pop_task() {  # -> task name on stdout, empty when queue drained
  local task
  {
    flock 9
    task="$(head -n 1 "$QUEUE_FILE" 2>/dev/null || true)"
    [[ -n "$task" ]] && sed -i '1d' "$QUEUE_FILE"
  } 9<>"$QUEUE_LOCK"
  echo "$task"
}

append_locked() {  # $1=file, $2=line
  {
    flock 9
    echo "$2" >> "$1"
  } 9<>"$QUEUE_LOCK"
}

parse_result() {  # $1=task -> newest _result.txt success rate (last numeric line)
  local dir="$ROBOTWIN_DIR/eval_result/$1/$ROBOTWIN_POLICY_NAME/$TASK_CONFIG/$LABEL"
  local newest
  newest="$(ls -td "$dir"/*/ 2>/dev/null | head -1)"
  [[ -n "$newest" && -f "$newest/_result.txt" ]] || return 1
  awk '/^[0-9.]+$/{v=$0} END{if (v=="") exit 1; print v}' "$newest/_result.txt"
}

# ---------------------------------------------------------------------------- #
# per-GPU worker: one persistent server + sequential task evals on the same GPU
# ---------------------------------------------------------------------------- #
run_worker() {
  local wid="$1" gpu="$2" port="$3"
  local tag="[worker$wid gpu$gpu]"
  local server_log="$RUN_DIR/logs/server_gpu${gpu}.log"

  setsid env CKPT="$CKPT" PORT="$port" SERVE_GPU="$gpu" \
    ROBOTWIN_ACTION_NORMALIZATION="$ACTION_NORMALIZATION" \
    ROBOTWIN_DOWNSAMPLE_VIDEO_FRAMES="$DOWNSAMPLE_VIDEO_FRAMES" \
    ROBOTWIN_VIDEO_DOWNSAMPLE_FACTOR="$VIDEO_DOWNSAMPLE_FACTOR" \
    ROBOTWIN_NUM_STEPS="$ROBOTWIN_NUM_STEPS" \
    ROBOTWIN_COND="$COND" \
    ROBOTWIN_GOAL_LAYOUT="$GOAL_LAYOUT" \
    ROBOTWIN_GOAL_SOURCE="$GOAL_SOURCE" \
    ROBOTWIN_EXPERIMENT="$ROBOTWIN_EXPERIMENT" \
    ROBOTWIN_ACTION_STATS_PATH="$ROBOTWIN_ACTION_STATS_PATH" \
    bash -c 'cd "$1" && exec bash scripts/eval/serve_robotwin_policy.sh' _ "$COSMOS_ROOT" \
    >"$server_log" 2>&1 &
  local server_pid=$!
  echo "$server_pid" >> "$RUN_DIR/.server_pids"

  echo "$tag waiting for server on port $port (timeout ${SERVER_READY_TIMEOUT}s) ..."
  local waited=0
  until (exec 3<>"/dev/tcp/$SERVER_HOST/$port") 2>/dev/null; do
    if ! kill -0 "$server_pid" 2>/dev/null; then
      echo "$tag server DIED during startup; draining nothing. Last log lines:" >&2
      tail -n 20 "$server_log" >&2 || true
      append_locked "$FAILED_TXT" "__server_gpu${gpu}__,startup_failed"
      return 1
    fi
    (( waited >= SERVER_READY_TIMEOUT )) && { echo "$tag server startup TIMED OUT" >&2; kill -TERM -"$server_pid" 2>/dev/null; return 1; }
    sleep 5; waited=$((waited + 5))
  done
  echo "$tag server ready after ${waited}s"

  local task
  while task="$(pop_task)"; [[ -n "$task" ]]; do
    local t0=$SECONDS
    echo "$tag START $task ($TASK_CONFIG, $TEST_NUM eps)"
    COSMOS_SERVER_HOST="$SERVER_HOST" COSMOS_SERVER_PORT="$port" \
    COSMOS_POLICY_NAME="$ROBOTWIN_POLICY_NAME" \
    COSMOS_POLICY_REPLAN_STEPS="$REPLAN_STEPS" \
    COSMOS_GOAL_COND="$COSMOS_GOAL_COND" COSMOS_GOAL_SOURCE="$GOAL_SOURCE" \
    COSMOS_TEST_NUM="$TEST_NUM" \
    WARP_CACHE_PATH="$RUN_DIR/.warp_cache/gpu${gpu}" \
      bash "$EVAL_SCRIPT" "$task" "$TASK_CONFIG" "$LABEL" "$EVAL_SEED" "$gpu" \
      > "$RUN_DIR/logs/${task}_${TASK_CONFIG}.log" 2>&1
    local rc=$?
    local sr=""
    if [[ $rc -eq 0 ]] && sr="$(parse_result "$task")"; then
      append_locked "$RESULTS_CSV" "$task,$TASK_CONFIG,$sr,$TEST_NUM,$gpu,$rc"
      echo "$tag DONE  $task success_rate=$sr ($(( (SECONDS - t0) / 60 )) min)"
    else
      append_locked "$RESULTS_CSV" "$task,$TASK_CONFIG,,${TEST_NUM},$gpu,$rc"
      append_locked "$FAILED_TXT" "$task,rc=$rc,log=$RUN_DIR/logs/${task}_${TASK_CONFIG}.log"
      echo "$tag FAIL  $task rc=$rc (see logs/${task}_${TASK_CONFIG}.log) — continuing" >&2
    fi
  done

  echo "$tag queue drained; stopping server"
  kill -TERM -"$server_pid" 2>/dev/null || kill -TERM "$server_pid" 2>/dev/null || true
}

# ---------------------------------------------------------------------------- #
# launch workers, wait, aggregate
# ---------------------------------------------------------------------------- #
: > "$RUN_DIR/.server_pids"
WORKER_PIDS=()
cleanup() {
  local code=$?
  trap - EXIT INT TERM
  for pid in "${WORKER_PIDS[@]:-}"; do kill "$pid" 2>/dev/null || true; done
  if [[ -f "$RUN_DIR/.server_pids" ]]; then
    while read -r spid; do
      [[ -n "$spid" ]] && { kill -TERM -"$spid" 2>/dev/null || kill -TERM "$spid" 2>/dev/null || true; }
    done < "$RUN_DIR/.server_pids"
  fi
  exit "$code"
}
trap cleanup EXIT INT TERM

for i in "${!GPU_ARR[@]}"; do
  run_worker "$i" "${GPU_ARR[$i]}" "$((BASE_PORT + i))" &
  WORKER_PIDS+=($!)
  sleep 2   # stagger server launches (checkpoint read contention)
done
wait "${WORKER_PIDS[@]}"
WORKER_PIDS=()

# summary: per-task table + overall mean (over tasks WITH results)
"$COSMOS_ROOT/.venv/bin/python" - "$RESULTS_CSV" "$RUN_DIR/summary.csv" <<'PYEOF'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
rows.sort(key=lambda r: r["task"])
rates = [float(r["success_rate"]) for r in rows if r["success_rate"] != ""]
n_fail = sum(1 for r in rows if r["success_rate"] == "")
with open(sys.argv[2], "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["task", "task_config", "success_rate"])
    for r in rows:
        w.writerow([r["task"], r["task_config"], r["success_rate"]])
    if rates:
        w.writerow(["__overall__", rows[0]["task_config"] if rows else "", f"{sum(rates)/len(rates):.4f}"])
print()
print(f"{'task':<40s} {'success_rate':>12s}")
for r in rows:
    print(f"{r['task']:<40s} {r['success_rate'] or 'FAILED':>12s}")
print("-" * 53)
if rates:
    print(f"{'OVERALL MEAN (' + str(len(rates)) + ' tasks)':<40s} {sum(rates)/len(rates):>12.4f}")
if n_fail:
    print(f"FAILED tasks: {n_fail} (see failed_tasks.txt)")
PYEOF

echo "[eval_50tasks] summary: $RUN_DIR/summary.csv"
echo "[eval_50tasks] per-task logs: $RUN_DIR/logs/"
[[ -s "$FAILED_TXT" ]] && { echo "[eval_50tasks] some tasks FAILED: $FAILED_TXT" >&2; exit 2; }
exit 0
