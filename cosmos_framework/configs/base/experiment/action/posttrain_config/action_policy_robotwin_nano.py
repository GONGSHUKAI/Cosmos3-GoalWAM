# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""``action_policy_robotwin_nano`` — Cosmos3-Nano RoboTwin action policy SFT recipe.

Mirrors ``action_policy_droid_nano`` (joint video + action via rectified flow) but
feeds the RoboTwin aloha-agilex action dataset (14D absolute joint + ``use_state``,
raw/un-normalized, 3-camera concat_view) through ``ActionTransformPipeline``, and
trains the generation + action heads from the public ``nvidia/Cosmos3-Nano`` base.

Generic ``robotwin`` naming: starts on one task (``place_a2b_left``) but the same
recipe serves more RoboTwin tasks (point ``ROBOTWIN_ROOT`` at any aloha-agilex
lerobot-v3.0 dataset).

Usage (1 node, 8 GPU)::

    ROBOTWIN_ROOT=/path/to/<task>_lerobot_v3.0 \\
    BASE_CHECKPOINT_PATH=<Cosmos3-Nano DCP dir> \\
    WAN_VAE_PATH=<Wan2.2_VAE.pth> \\
    torchrun --nproc_per_node=8 -m cosmos_framework.scripts.train \\
        --sft-toml examples/toml/sft_config/action_policy_robotwin.toml
"""

import copy

from hydra.core.config_store import ConfigStore

from cosmos_framework.configs.base.experiment.sft.models.nano_model_config import NANO_MODEL_CONFIG
from cosmos_framework.data.vfm.action.datasets.action_sft_dataset import get_action_robotwin_sft_dataset
from cosmos_framework.data.vfm.joint_dataloader import (
    PackingDataLoader,
    RankPartitionedDataLoader,
)
from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.utils.lazy_config import LazyDict

cs = ConfigStore.instance()


action_policy_robotwin_nano = LazyDict(
    dict(
        defaults=[
            {"override /model": "mot_fsdp"},
            {"override /data_train": None},
            {"override /data_val": None},
            # FusedAdam with fp32 master_weights + eps 1e-8 (bf16 params + eps 1e-6
            # diverged on the action loss).
            {"override /optimizer": "fusedadamw"},
            {"override /scheduler": "lambdalinear"},  # linear LR decay
            {"override /checkpoint": "s3"},
            {
                "override /callbacks": [
                    "basic",
                    "optimization",
                    "job_monitor",
                ]
            },
            {"override /ema": "power"},
            {"override /tokenizer": "wan2pt2_tokenizer"},
            {"override /sound_tokenizer": None},
            {"override /vlm_config": None},
            {"override /ckpt_type": "dcp"},
            "_self_",
        ],
        job=dict(
            project="cosmos3",
            group="action_sft",
            name="action_policy_robotwin_nano",
            wandb_mode="disabled",
        ),
        model=dict(
            config=copy.deepcopy(NANO_MODEL_CONFIG),  # action_gen=True, max_action_dim=64
        ),
        optimizer=dict(
            betas=[0.9, 0.99],
            eps=1.0e-08,
            fused=True,  # popped by build_optimizer for FusedAdam (fused by construction)
            # Train the generation + action heads.
            keys_to_select=[
                "moe_gen",
                "time_embedder",
                "vae2llm",
                "llm2vae",
                "action2llm",
                "llm2action",
                "action_modality_embed",
            ],
            lr=2.0e-04,  # for the 8192 global batch (scale down via TOML/CLI for smaller batches)
            lr_multipliers={
                "action2llm": 5.0,
                "llm2action": 5.0,
                "action_modality_embed": 5.0,
            },
            optimizer_type="FusedAdam",
            weight_decay=0.05,
        ),
        scheduler=dict(
            lr_scheduler_type="LambdaLinear",
            cycle_lengths=[100],  # smoke: 100 iters (real run sets via TOML)
            f_max=[0.4],
            f_min=[0.0],
            f_start=[0.0],
            verbosity_interval=0,
            warm_up_steps=[0],
        ),
        trainer=dict(
            distributed_parallelism="fsdp",
            grad_accum_iter=1,
            logging_iter=1,
            max_iter=100,  # smoke
            max_val_iter=None,
            run_validation=False,
            run_validation_on_start=False,
            save_zero_checkpoint=False,
            seed=42,
            timeout_period=999999999,
            validation_iter=100,
            compile_config=dict(recompile_limit=8, use_duck_shape=False),
            cudnn=dict(benchmark=True, deterministic=False),
            ddp=dict(broadcast_buffers=True, find_unused_parameters=False, static_graph=True),
            grad_scaler_args=dict(enabled=False),
            callbacks=dict(
                device_monitor=dict(
                    every_n=200, log_memory_detail=True, save_s3=False, step_size=1, upload_every_n_mul=5
                ),
                grad_clip=dict(clip_norm=1.0, force_finite=True),
                heart_beat=dict(every_n=200, save_s3=False, step_size=1, update_interval_in_minute=20),
                iter_speed=dict(every_n=1, hit_thres=50, save_s3=False, save_s3_every_log_n=500),
                low_precision=dict(update_iter=1),
                manual_gc=dict(every_n=5, gc_level=1, warm_up=1),
                param_count=dict(save_s3=False),
                skip_nan_step=dict(max_consecutive_nan=100),
                training_stats=dict(log_freq=100),
            ),
        ),
        checkpoint=dict(
            broadcast_via_filesystem=False,
            dcp_async_mode_enabled=False,
            enable_gcs_patch_in_boto3=True,
            keys_not_to_resume=[],
            # Skip net_ema. (EMA warm-starts from net, see dcp.py) and the action
            # heads, so they init fresh from the base (the base has no action heads).
            keys_to_skip_loading=[
                "net_ema.",
                "action2llm",
                "llm2action",
                "action_modality_embed",
                "action_pos_embed",
            ],
            load_ema_to_reg=False,
            load_path="???",  # Cosmos3-Nano DCP dir; supply via TOML/env
            load_training_state=False,
            only_load_scheduler_state=False,
            save_iter=100,
            strict_resume=False,  # base init: tolerate key set differences
            verbose=True,
            hf_export=dict(
                enabled=False,
                export_every_n=1,
                hf_repo_id=None,
                upload_to_object_store=dict(bucket="", credentials="", enabled=False),
            ),
            jit=dict(device="cuda", dtype="bfloat16", enabled=False, input_shape=None, strict=True),
            load_from_object_store=dict(bucket="", credentials="", enabled=False),
            save_to_object_store=dict(bucket="", credentials="", enabled=False),
        ),
        dataloader_train=L(PackingDataLoader)(
            audio_sample_rate=48000,
            dataset_name="action_robotwin",
            max_samples_per_batch=128,  # per rank -> 8192 global batch at 64 ranks
            max_sequence_length=None,  # None disables token packing (TOML can't express null)
            patch_spatial=2,
            sound_latent_fps=0,
            tokenizer_spatial_compression_factor=16,
            tokenizer_temporal_compression_factor=4,
            dataloader=L(RankPartitionedDataLoader)(
                batch_size=1,
                in_order=False,
                num_workers=4,
                persistent_workers=True,
                pin_memory=True,
                prefetch_factor=4,
                sampler=None,
                # Shuffling is handled by the dataset (iterable_shuffle=True below):
                # ActionIterableShuffleDataset streams rank x worker-sharded, episode-order-
                # shuffled, sequential-within-episode.
                datasets=dict(
                    robotwin=dict(
                        ratio=1,
                        dataset=L(get_action_robotwin_sft_dataset)(
                            root="${oc.env:ROBOTWIN_ROOT}",
                            fps=30.0,
                            chunk_length=32,
                            mode="policy",
                            use_state=True,
                            iterable_shuffle=True,  # rank x worker episode-shuffle stream
                            episode_shuffle_seed=42,
                            use_image_augmentation=True,  # SR boost (random crop+rescale + color jitter)
                            # Action normalization ablation toggle:
                            #   None       -> raw 14-D joint qpos (cosmos-DROID behavior, default)
                            #   "meanstd"  -> per-joint z-score (FastWAM-style; stats in
                            #                 datasets/stats/robotwin_lerobot_stats.json)
                            # Override on the CLI without editing this file, e.g.:
                            #   ...dataloader_train.dataloader.datasets.robotwin.dataset.action_normalization=meanstd
                            # The eval server must pass the matching --action-normalization to denormalize.
                            action_normalization=None,
                            # FastWAM-style video temporal ablation toggle:
                            #   False -> Cosmos-DROID behavior, 33 video frames (0..32)
                            #   True  -> video frames 0,4,8,...,32; actions stay 33 rows
                            # If enabling this, also override:
                            #   model.config.tokenizer.encode_exact_durations=[9]
                            downsample_video_frames=False,
                            # Automatically use precomputed concat-view videos created by
                            # scripts/preprocess/preprocess_robotwin_offline_concat.py when present.
                            # Kept disabled during online augmentation by default because the
                            # augmentation path intentionally jitters each raw camera before concat.
                            use_offline_concat=True,
                            use_offline_concat_with_augmentation=False,
                            viewpoint="concat_view",  # head 256x320 (top) + L/R wrists 128x160 each (bottom)
                            resolution="384x320",  # exact RoboTwin concat bucket; no square padding
                            max_action_dim="${model.config.max_action_dim}",
                            cfg_dropout_rate=0.1,
                            tokenizer_config="${model.config.vlm_config.tokenizer}",
                        ),
                    ),
                ),
            ),
        ),
        dataloader_val=None,
        upload_reproducible_setup=False,
    ),
    flags={"allow_objects": True},
)


# Register both no-padding RoboTwin buckets for RF shift + VAE chunk sizing
# (320x384 for 320x240 cameras, 640x736 for 640x480 cameras). The dataset/eval
# pick one via the ``resolution`` knob; the 33-frame encode duration stays exact.
# To switch to 736x640, also set the dataset ``resolution`` below and ``model.config.resolution``.
action_policy_robotwin_nano["model"]["config"]["rectified_flow_training_config"]["shift"]["384x320"] = 5
action_policy_robotwin_nano["model"]["config"]["rectified_flow_training_config"]["shift"]["736x640"] = 5
action_policy_robotwin_nano["model"]["config"]["tokenizer"]["encode_chunk_frames"]["384x320"] = 24
action_policy_robotwin_nano["model"]["config"]["tokenizer"]["encode_chunk_frames"]["736x640"] = 24
action_policy_robotwin_nano["model"]["config"]["resolution"] = "384x320"


# chunk_length=32 -> 33 observation frames; pin the VAE encode duration to match.
# Set post-construction so it lands on the deep-copied NANO_MODEL_CONFIG.tokenizer.
action_policy_robotwin_nano["model"]["config"]["tokenizer"]["encode_exact_durations"] = [33]


# Uncap the packed-sequence length (NANO default 45056 truncates long windows). -1
# processes the full vision sequence per step; does not change the per-token loss.
action_policy_robotwin_nano["model"]["config"]["max_num_tokens_after_packing"] = -1


# Weight the vision flow-matching loss 10x (NANO default 1.0), balancing it against
# the action loss (action_loss_weight=10) so both heads train at comparable magnitude.
action_policy_robotwin_nano["model"]["config"]["rectified_flow_training_config"]["loss_scale"] = 10.0


# ---------------------------------------------------------------------------
# ``action_policy_robotwin_nano_goal`` — goal-image conditioned variant.
#
# The policy is additionally conditioned on a GOAL IMAGE (last frame of the
# training episode; terminal frame of the rule-based expert at eval time),
# injected through the REASONER branch: the frozen Qwen3-VL vision tower
# encodes the goal frame and its embeddings are scattered at ``<|image_pad|>``
# placeholder positions inside the caption — the generation tower reads them
# via the existing joint attention. Trainable parameter set is unchanged
# (``keys_to_select`` above); the vision tower and reasoner stay frozen.
#
# Switches (override per train script):
#   ...datasets.robotwin.dataset.cond        = text_cond | goal_frame_cond | text_goal_frame_cond
#   ...datasets.robotwin.dataset.goal_layout = concat | cam_high
# The eval server must run with the SAME cond/goal_layout (--cond auto reads
# them from the checkpoint's saved config).
# ---------------------------------------------------------------------------
action_policy_robotwin_nano_goal = copy.deepcopy(action_policy_robotwin_nano)
action_policy_robotwin_nano_goal["job"]["name"] = "action_policy_robotwin_nano_goal"
# Build the Qwen3-VL vision tower on the reasoner (frozen: "visual" is not in
# the optimizer keys_to_select allowlist).
action_policy_robotwin_nano_goal["model"]["config"]["vlm_config"]["model_instance"]["config"]["include_visual"] = True
# The base Cosmos3-Nano DCP carries no visual.* keys — skip them on warm start;
# OmniMoTModel.load_pretrained_model_if_needed seeds the tower from the
# original Qwen3-VL safetensors (vlm_config.model_name) instead. Mid-run
# resumes restore visual.* from the trained DCP.
action_policy_robotwin_nano_goal["checkpoint"]["keys_to_skip_loading"] = list(
    action_policy_robotwin_nano_goal["checkpoint"]["keys_to_skip_loading"]
) + ["visual"]
# Goal-conditioning defaults (train scripts override cond per variant).
_goal_dataset = action_policy_robotwin_nano_goal["dataloader_train"]["dataloader"]["datasets"]["robotwin"]["dataset"]
_goal_dataset["cond"] = "text_goal_frame_cond"
_goal_dataset["goal_layout"] = "concat"


for _item in [action_policy_robotwin_nano, action_policy_robotwin_nano_goal]:
    _name = [k for k, v in globals().items() if v is _item][0]
    cs.store(group="experiment", package="_global_", name=_name, node=_item)
