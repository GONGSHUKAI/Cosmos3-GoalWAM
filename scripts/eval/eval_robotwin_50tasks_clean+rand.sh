#!/usr/bin/env bash
# ALL-50-TASK RoboTwin eval — text-cond baseline trained on clean+rand (unified 27.5k eps)
#
# One persistent policy server per GPU (model loaded once), tasks pulled from a
# shared queue, results aggregated to summary.csv. See
# _robotwin_50tasks_eval_common.sh for the full contract.
#
# Usage:
#   bash scripts/eval/eval_robotwin_50tasks_clean+rand.sh [ckpt_dir] [clean|random]
# Env: GPUS=0,1,...,7  TEST_NUM=100  TASKS="taskA taskB"  LABEL  DRY_RUN=1  ...
#
# Single-task debugging lives in eval_robotwin_1task_clean+rand.sh.

COSMOS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$COSMOS_ROOT/scripts/_common.sh"
cosmos_load_local_env

VARIANT="50tasks_clean+rand"
DEFAULT_CKPT="${DEFAULT_CKPT:-$COSMOS_ROOT/outputs/train/cosmos3_action/action_sft/action_policy_robotwin_50tasks_clean+rand/checkpoints/iter_000032000}"
ROBOTWIN_POLICY_NAME="${ROBOTWIN_POLICY_NAME:-cosmos_policy_50tasks_clean+rand}"
ROBOTWIN_EXPERIMENT="${ROBOTWIN_EXPERIMENT:-action_policy_robotwin_nano}"
ROBOTWIN_ACTION_STATS_PATH="${ROBOTWIN_ACTION_STATS_PATH:-$COSMOS_ROOT/cosmos_framework/data/vfm/action/datasets/stats/robotwin_lerobot_stats.json}"
COSMOS_GOAL_COND=0
COND="${COND:-auto}"
BASE_PORT="${BASE_PORT:-9700}"

source "$(dirname "${BASH_SOURCE[0]}")/_robotwin_50tasks_eval_common.sh"
