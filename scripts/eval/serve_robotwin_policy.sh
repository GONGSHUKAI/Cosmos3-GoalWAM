#!/usr/bin/env bash
COSMOS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$COSMOS_ROOT"
source "$COSMOS_ROOT/scripts/_common.sh"
cosmos_load_local_env
cosmos_activate_venv

# Cosmos3-Nano-Policy-RoboTwin inference server (cosmos .venv, py3.13).
# Pairs with RoboTwin/policy/cosmos_policy (run in the robotwin conda env).
# Run from anywhere:
#   bash /path/to/cosmos-framework/scripts/eval/serve_robotwin_policy.sh
#
# ── STEP 1 (once per checkpoint): export the trained DCP -> self-contained HF safetensors ──
#   The robust/documented serving artifact is the EXPORTED dir (carries config.json +
#   checkpoint.json so the server recovers the tokenizer from metadata, and EMA weights
#   are applied). Export it yourself (heavy, ~minutes, needs network for the ViT):
#
#     ITER=/path/to/action_policy_robotwin/checkpoints/iter_000000100
#     PYTHONPATH=. python -m cosmos_framework.scripts.export_model \
#       --checkpoint-path "$ITER" --experiment action_policy_robotwin_nano \
#       -o /path/to/weights/Cosmos3-Nano-Policy-RoboTwin-iter100
#
# ── STEP 2: serve the exported dir (this script) ──

: "${COSMOS_WEIGHTS_ROOT:=$(dirname "${BASE_CHECKPOINT_PATH:-$COSMOS_ROOT/weights/Cosmos3-Nano-dcp}")}"
: "${CKPT:=$COSMOS_WEIGHTS_ROOT/Cosmos3-Nano-Policy-RoboTwin-iter100}"
: "${PORT:=9876}"
: "${ROBOTWIN_RESOLUTION:=384x320}"   # set 736x640 for RoboTwin data rendered from 640x480 cameras
: "${ROBOTWIN_DOWNSAMPLE_VIDEO_FRAMES:=auto}"  # auto/true/false; true -> video frames 0,4,8,...,32
: "${ROBOTWIN_VIDEO_DOWNSAMPLE_FACTOR:=4}"
: "${ROBOTWIN_NUM_STEPS:=4}"
# Action-normalization ablation: must match how the checkpoint was TRAINED.
#   auto    -> read from the training config (none for raw qpos, meanstd if normalized)
#   none    -> return raw model output (raw-qpos checkpoints)
#   meanstd -> denormalize with per-joint z-score stats (normalized checkpoints)
: "${ROBOTWIN_ACTION_NORMALIZATION:=auto}"
: "${ROBOTWIN_ACTION_STATS_PATH:=}"   # blank -> dataset's bundled robotwin_lerobot_stats.json
# Goal-image conditioning: must match how the checkpoint was TRAINED.
#   auto -> read cond/goal_layout from the training config (goal-trained runs
#           record them; older text-only checkpoints resolve to text_cond).
: "${ROBOTWIN_COND:=auto}"            # auto/text_cond/goal_frame_cond/text_goal_frame_cond
: "${ROBOTWIN_GOAL_LAYOUT:=auto}"     # auto/concat/cam_high
: "${ROBOTWIN_GOAL_SOURCE:=oracle}"   # oracle (expert terminal obs) | generated (NotImplemented seam)
# R/B hotfix (default true): the lerobot TRAINING videos are R/B-swapped
# (cv2.imencode + PIL decode mismatch); swapping incoming obs + goal frames
# feeds checkpoints the color world they were trained in. Set
# ROBOTWIN_SWAP_RB=false only for checkpoints trained on regenerated data.
: "${ROBOTWIN_SWAP_RB:=true}"
: "${ROBOTWIN_EXPERIMENT:=action_policy_robotwin_nano}"  # raw-DCP fallback experiment (goal ckpts: action_policy_robotwin_nano_goal)
export CUDA_VISIBLE_DEVICES="${SERVE_GPU:-0}"   # pick a GPU free of the trainer

if [ ! -d "$CKPT" ]; then
  echo "ERROR: checkpoint dir not found: $CKPT"
  echo "Export a trained DCP first (see STEP 1 in this script header), or set CKPT=<exported dir>."
  exit 1
fi

# Direct-DCP fallback (skip export): set CKPT to an iter_<N> DCP dir; you then ALSO need
# --experiment + ROBOTWIN_ROOT. Export-first (above) is recommended.
EXTRA=()
if compgen -G "$CKPT/*.distcp" > /dev/null || compgen -G "$CKPT/model/*.distcp" > /dev/null; then
  echo "[serve] CKPT looks like a raw DCP; loading directly with --experiment (export-first is recommended)."
  EXTRA+=(--experiment "$ROBOTWIN_EXPERIMENT")
  : "${ROBOTWIN_DATA_ROOT:?Set ROBOTWIN_DATA_ROOT in config/local.env or the shell environment}"
  export ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-$ROBOTWIN_DATA_ROOT/robotwin2.0/place_a2b_left/aloha-agilex_combined_550_lerobot_v3.0}"
  export HF_HUB_OFFLINE=1
fi

if [ -n "$ROBOTWIN_ACTION_STATS_PATH" ]; then
  EXTRA+=(--action-stats-path "$ROBOTWIN_ACTION_STATS_PATH")
fi

PYTHONPATH=. python -m cosmos_framework.scripts.action_policy_server_robotwin \
    --checkpoint-path "$CKPT" \
    --port "$PORT" \
    --resolution "$ROBOTWIN_RESOLUTION" \
    --downsample-video-frames "$ROBOTWIN_DOWNSAMPLE_VIDEO_FRAMES" \
    --video-downsample-factor "$ROBOTWIN_VIDEO_DOWNSAMPLE_FACTOR" \
    --action-normalization "$ROBOTWIN_ACTION_NORMALIZATION" \
    --cond "$ROBOTWIN_COND" \
    --goal-layout "$ROBOTWIN_GOAL_LAYOUT" \
    --goal-source "$ROBOTWIN_GOAL_SOURCE" \
    --swap-rb "$ROBOTWIN_SWAP_RB" \
    --conditioning-fps 30 \
    --action-chunk-size 32 \
    --guidance 3.0 \
    --num-steps "$ROBOTWIN_NUM_STEPS" \
    --shift 5.0 \
    "${EXTRA[@]}"
