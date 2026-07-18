#!/usr/bin/env bash
COSMOS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$COSMOS_ROOT"
source "$COSMOS_ROOT/scripts/_common.sh"
cosmos_load_local_env
cosmos_activate_venv
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"

# RoboTwin action-policy SFT with GOAL-IMAGE-ONLY conditioning
# (cond=goal_frame_cond): the instruction text is BLANKED (metadata sentences kept) and the episode-final frame conditions the policy
# through the frozen Qwen3-VL vision tower on the reasoner branch, instead of
# the instruction. Always meanstd action normalization + 4x video
# downsampling (frames 0,4,...,32; actions stay 33 rows).
#
#   GOAL_LAYOUT=concat   (default) 3-camera inverted-T composite goal frame
#   GOAL_LAYOUT=cam_high head-camera-only goal frame
#
# Eval side: scripts/eval/eval_robotwin_50tasks_clean2rand_goal_frame_cond.sh (the server
# auto-reads cond/goal_layout from this run's saved config).

# clean2rand scenario: 50 tasks x 50 CLEAN episodes (2500 episodes), evaluated on
# randomized scenes. Uses the repo-bundled robotwin_clean_lerobot_stats.json
# (identical to the dataset's action_stats.json); the eval scripts default to
# the same file, so train/eval denormalization match automatically.
: "${ROBOTWIN_DATA_ROOT:?Set ROBOTWIN_DATA_ROOT in config/local.env or the shell environment}"
export DATASET_PATH="${DATASET_PATH:-$ROBOTWIN_DATA_ROOT/robotwin2.0_clean2rand_lerobot_v3.0}"
export ROBOTWIN_ROOT="$DATASET_PATH"
cosmos_require_dir DATASET_PATH "$DATASET_PATH"
export ROBOTWIN_ACTION_STATS_PATH="${ROBOTWIN_ACTION_STATS_PATH:-$COSMOS_ROOT/cosmos_framework/data/vfm/action/datasets/stats/robotwin_clean_lerobot_stats.json}"
cosmos_prepare_training_env
# WANDB_API_KEY is loaded from config/local.env, ~/.netrc, or the shell environment.
export WANDB_INIT_TIMEOUT=300
export WANDB__SERVICE_WAIT=300

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
IFS=',' read -ra _GPUS <<< "$CUDA_VISIBLE_DEVICES"
export NPROC_PER_NODE="${#_GPUS[@]}"
export MASTER_PORT="${MASTER_PORT:-50021}"
export LOG_FILENAME="${LOG_FILENAME:-action_policy_robotwin_50tasks_clean2rand_goal_frame_cond_sft.log}"

export GOAL_LAYOUT="${GOAL_LAYOUT:-concat}"
export TOML_FILE=examples/toml/sft_config/action_policy_robotwin_goal.toml

export EXTRA_TAIL_OVERRIDES=" \
    job.name=action_policy_robotwin_50tasks_clean2rand_goal_frame_cond \
    dataloader_train.dataloader.datasets.robotwin.dataset.cond=goal_frame_cond \
    dataloader_train.dataloader.datasets.robotwin.dataset.goal_layout=${GOAL_LAYOUT} \
    dataloader_train.dataloader.datasets.robotwin.dataset.action_normalization=meanstd \
    dataloader_train.dataloader.datasets.robotwin.dataset.use_image_augmentation=false \
    dataloader_train.dataloader.datasets.robotwin.dataset.use_offline_concat=true \
    dataloader_train.dataloader.datasets.robotwin.dataset.downsample_video_frames=true \
    model.config.tokenizer.encode_exact_durations=[9] \
    model.config.parallelism.data_parallel_shard_degree=${NPROC_PER_NODE} \
"

bash examples/launch_sft_action_policy_robotwin.sh
