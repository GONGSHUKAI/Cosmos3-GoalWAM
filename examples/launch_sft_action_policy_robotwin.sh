#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# ============================================================================
# Structured-TOML launch for RoboTwin action-policy SFT on Cosmos3-Nano (MoT).
# Drives cosmos_framework.scripts.train against
# examples/toml/sft_config/action_policy_robotwin.toml (selects the registered
# `action_policy_robotwin_nano` experiment; res384x320, 14D dual-arm joint +
# use_state, 3-camera concat_view, trains the generation + action heads). See
# action_policy_droid_posttrain.md for the DROID analog.
#
# Env vars (override for your filesystem):
#   DATASET_PATH          RoboTwin aloha-agilex lerobot-v3.0 dir (has meta/info.json)
#   BASE_CHECKPOINT_PATH  DCP of nvidia/Cosmos3-Nano (convert_model_to_dcp; see docs)
#   WAN_VAE_PATH          Wan2.2 VAE .pth (Wan-AI/Wan2.2-TI2V-5B)
#   WANDB_API_KEY         for online logging (TOML wandb_mode="online")
#   NPROC_PER_NODE        torchrun --nproc_per_node (default 8)
#   EXTRA_TAIL_OVERRIDES  space-separated Hydra overrides
#
# Single-node smoke (config/data sanity, a few iters):
#   export EXTRA_TAIL_OVERRIDES="trainer.max_iter=10 checkpoint.save_iter=10 \
#                                dataloader_train.max_samples_per_batch=32"
#   bash examples/launch_sft_action_policy_robotwin.sh
# ============================================================================

TOML_FILE="examples/toml/sft_config/action_policy_robotwin.toml"
: "${DATASET_PATH:=examples/data/lerobot_v30/robotwin_lerobot/success}"
: "${BASE_CHECKPOINT_PATH:=examples/checkpoints/Cosmos3-Nano}"

# The experiment reads ${oc.env:ROBOTWIN_ROOT}; bridge the launcher's DATASET_PATH to it.
export ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-$DATASET_PATH}"

EXTRA_DATASET_CHECK='[[ -f "$ROBOTWIN_ROOT/meta/info.json" ]] || { echo "ERROR: missing $ROBOTWIN_ROOT/meta/info.json (prepare a RoboTwin lerobot-v3.0 dataset — see ROBOT_NOTE.md)" >&2; exit 1; }'

# Extra Hydra overrides from the environment: a space-separated string word-split into
# the TAIL_OVERRIDES array. An exported string survives `bash <wrapper>` (a child
# process), unlike a TAIL_OVERRIDES array set in your shell.
TAIL_OVERRIDES=(
    ${EXTRA_TAIL_OVERRIDES:-}
)

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
