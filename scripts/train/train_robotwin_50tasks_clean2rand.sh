#!/usr/bin/env bash
COSMOS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$COSMOS_ROOT"
source "$COSMOS_ROOT/scripts/_common.sh"
cosmos_load_local_env
cosmos_activate_venv
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"

# RoboTwin 50-task clean-only action-policy SFT for clean-to-random evaluation.
# Video uses frames 0,4,8,...,32 (9 frames); actions stay 33 rows
# (initial qpos + 32 action steps).

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
export MASTER_PORT="${MASTER_PORT:-50018}"
export LOG_FILENAME="${LOG_FILENAME:-action_policy_robotwin_50tasks_clean2rand_sft.log}"

export EXTRA_TAIL_OVERRIDES=" \
    job.name=action_policy_robotwin_50tasks_clean2rand \
    dataloader_train.dataloader.datasets.robotwin.dataset.action_normalization=meanstd \
    dataloader_train.dataloader.datasets.robotwin.dataset.use_image_augmentation=false \
    dataloader_train.dataloader.datasets.robotwin.dataset.use_offline_concat=true \
    dataloader_train.dataloader.datasets.robotwin.dataset.downsample_video_frames=true \
    model.config.tokenizer.encode_exact_durations=[9] \
    model.config.parallelism.data_parallel_shard_degree=${NPROC_PER_NODE} \
"

bash examples/launch_sft_action_policy_robotwin.sh
