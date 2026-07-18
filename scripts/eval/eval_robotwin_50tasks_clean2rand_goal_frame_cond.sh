#!/usr/bin/env bash
# ALL-50-TASK RoboTwin eval — oracle-goal-image-only conditioning (instruction ignored), trained on clean only
#
# One persistent policy server per GPU (model loaded once), tasks pulled from a
# shared queue, results aggregated to summary.csv. See
# _robotwin_50tasks_eval_common.sh for the full contract.
#
# Usage:
#   bash scripts/eval/eval_robotwin_50tasks_clean2rand_goal_frame_cond.sh [ckpt_dir] [clean|random]
# Env: GPUS=0,1,...,7  TEST_NUM=100  TASKS="taskA taskB"  LABEL  DRY_RUN=1  ...
#
# Single-task debugging lives in eval_robotwin_1task_clean2rand_goal_frame_cond.sh.

COSMOS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$COSMOS_ROOT/scripts/_common.sh"
cosmos_load_local_env

VARIANT="50tasks_clean2rand_goal_frame_cond"
DEFAULT_CKPT="${DEFAULT_CKPT:-$COSMOS_ROOT/outputs/train/cosmos3_action/action_sft/action_policy_robotwin_50tasks_clean2rand_goal_frame_cond/checkpoints/iter_000006000}"
ROBOTWIN_POLICY_NAME="${ROBOTWIN_POLICY_NAME:-cosmos_policy_50tasks_clean2rand_goal_frame_cond}"
ROBOTWIN_EXPERIMENT="${ROBOTWIN_EXPERIMENT:-action_policy_robotwin_nano_goal}"
ROBOTWIN_ACTION_STATS_PATH="${ROBOTWIN_ACTION_STATS_PATH:-$COSMOS_ROOT/cosmos_framework/data/vfm/action/datasets/stats/robotwin_clean_lerobot_stats.json}"
COSMOS_GOAL_COND=1
COND="${COND:-goal_frame_cond}"
BASE_PORT="${BASE_PORT:-9730}"

source "$(dirname "${BASH_SOURCE[0]}")/_robotwin_50tasks_eval_common.sh"
