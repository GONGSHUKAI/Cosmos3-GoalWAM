#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Debug / visualize a RoboTwin *training* batch exactly as the trainer sees it.

This builds the same map-style dataset the ``action_policy_robotwin_nano`` recipe
uses (``get_action_robotwin_sft_dataset`` with the recipe's knobs: concat_view,
384x320 no-padding bucket, ``use_state``, image augmentation, optional action
normalization), streams a batch through the SAME episode-shuffle iterable +
``custom_collate_fn`` the dataloader uses, prints the shape/dtype of every element
in the collated batch, and writes the per-sample training video(s) (the 3-camera
concat the model is actually trained on) as mp4 next to this file.

Run from the repo root inside the cosmos venv::

    LD_LIBRARY_PATH= PYTHONPATH=. \
    ROBOTWIN_ROOT=/pfs/pfs-7jnepv/shukaigong/data/robotwin2.0/place_a2b_left/aloha-agilex_combined_550_lerobot_v3.0 \
        .venv/bin/python debug_train_dataloader_robotwin.py --batch-size 4

``--action-normalization <mode>``: default ``none``, raw qpos, the cosmos-DROID behavior. ``meanstd`` uses z-score normalization
``--no-text``: to skip building the Qwen tokenizer (avoids the HF download when you only want video/action shapes).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from torch.utils.data import IterableDataset

from cosmos_framework.data.vfm.action.action_processing import ActionProcessingRecord
from cosmos_framework.data.vfm.action.datasets.action_sft_dataset import get_action_robotwin_sft_dataset
from cosmos_framework.data.vfm.joint_dataloader import custom_collate_fn
from cosmos_framework.data.vfm.sequence_packing import SequencePlan
from cosmos_framework.tools.visualize.video import save_img_or_video

_DEFAULT_ROOT = "/pfs/pfs-7jnepv/shukaigong/data/robotwin2.0/place_a2b_left/aloha-agilex_combined_550_lerobot_v3.0"
# Mirrors model.config.vlm_config.tokenizer in nano_model_config.py.
_TOKENIZER_CONFIG = {
    "_target_": "cosmos_framework.configs.base.defaults.vlm.create_qwen2_tokenizer_with_download",
    "config_variant": "hf",
    "pretrained_model_name": "Qwen/Qwen3-VL-8B-Instruct",
}


def _describe(value: object) -> str:
    """One-line shape/dtype/type summary for a single collated batch element."""
    if isinstance(value, torch.Tensor):
        return f"Tensor shape={tuple(value.shape)} dtype={value.dtype}"
    if isinstance(value, SequencePlan):
        return f"SequencePlan {value.as_dict()}"
    if isinstance(value, ActionProcessingRecord):
        norm = type(value.action_normalizer).__name__ if value.action_normalizer is not None else None
        return f"ActionProcessingRecord(raw_action_dim={value.raw_action_dim}, action_normalizer={norm})"
    if isinstance(value, list):
        inner = ", ".join(_describe(v) for v in value[:1])
        suffix = ", ..." if len(value) > 1 else ""
        return f"list(len={len(value)}) [{inner}{suffix}]"
    if isinstance(value, (str, int, float, bool)) or value is None:
        return f"{type(value).__name__}={value!r}"
    return f"{type(value).__name__}={value!r}"


def _print_batch(batch: dict) -> None:
    print("\n================ collated train batch ================")
    for key in sorted(batch):
        print(f"  {key:32s}: {_describe(batch[key])}")
    print("======================================================\n")


def _save_videos(batch: dict, out_dir: Path, fps: int) -> None:
    videos = batch.get("video")
    if not isinstance(videos, list):
        raise RuntimeError(f"expected 'video' to be a list of [C,T,H,W] tensors, got {type(videos)}")

    saved: list[str] = []
    for i, video in enumerate(videos):
        assert isinstance(video, torch.Tensor) and video.ndim == 4, (
            f"sample {i} video must be [C,T,H,W], got {type(video)}"
        )
        path = out_dir / f"debug_robotwin_sample{i}"
        save_img_or_video(video, str(path), fps=fps)  # [C,T,H,W] uint8 -> <path>.mp4
        saved.append(f"{path}.mp4")
        c, t, h, w = video.shape
        print(f"  sample {i}: video [C={c},T={t},H={h},W={w}] {video.dtype} -> {path}.mp4")

    # Optional batch overview: concat samples side-by-side (width) into one clip.
    shapes = {tuple(v.shape) for v in videos}
    if len(shapes) == 1 and len(videos) > 1:
        grid = torch.cat(list(videos), dim=-1)  # [C,T,H,W*B]
        grid_path = out_dir / "debug_robotwin_batch_grid"
        save_img_or_video(grid, str(grid_path), fps=fps)
        saved.append(f"{grid_path}.mp4")
        print(f"  batch grid: {tuple(grid.shape)} -> {grid_path}.mp4")

    print(f"\nsaved {len(saved)} mp4(s) under {out_dir}")


def _action_summary(batch: dict) -> None:
    actions = batch.get("action")
    if not isinstance(actions, list) or not actions or not isinstance(actions[0], torch.Tensor):
        return
    a0 = actions[0].float()
    rad = batch.get("raw_action_dim")
    raw_dim = None
    if isinstance(rad, list) and rad and isinstance(rad[0], torch.Tensor):
        raw_dim = int(rad[0].item())
    print("------------ action[0] summary ------------")
    print(f"  shape={tuple(a0.shape)} raw_action_dim={raw_dim} (channels >= raw_dim are zero padding)")
    real = a0[:, : raw_dim] if raw_dim else a0
    print(f"  real-channel min={real.min().item():.4f} max={real.max().item():.4f} mean={real.mean().item():.4f}")
    print(f"  per-dim abs-max (first {real.shape[-1]}): {[round(x, 3) for x in real.abs().amax(0).tolist()]}")
    print("-------------------------------------------")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default=os.environ.get("ROBOTWIN_ROOT", _DEFAULT_ROOT),
                   help="RoboTwin lerobot dataset dir (default: $ROBOTWIN_ROOT or the place_a2b_left v3.0 path)")
    p.add_argument("--batch-size", type=int, default=4, help="number of samples to collate into the debug batch")
    p.add_argument("--action-normalization", default="none", choices=["none", "meanstd", "minmax", "quantile"],
                   help="match the training ablation; 'none' = raw qpos (default)")
    p.add_argument("--resolution", default="384x320", help="concat bucket / resolution (matches the recipe)")
    p.add_argument("--fps", type=int, default=30, help="fps for the saved mp4(s)")
    p.add_argument("--no-shuffle", action="store_true",
                   help="use the deterministic map-style order instead of the training episode-shuffle stream")
    p.add_argument("--no-augmentation", action="store_true", help="disable train-time image augmentation")
    p.add_argument("--no-text", action="store_true", help="skip building the Qwen tokenizer (no text_token_ids)")
    p.add_argument("--out-dir", default=str(Path(__file__).resolve().parent),
                   help="directory to write the mp4(s) (default: this script's directory)")
    args = p.parse_args()

    root = Path(args.root)
    if not (root / "meta" / "info.json").exists():
        raise SystemExit(f"ERROR: {root}/meta/info.json not found; set --root or ROBOTWIN_ROOT correctly.")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    action_normalization = None if args.action_normalization == "none" else args.action_normalization

    tokenizer_config = None
    if not args.no_text:
        try:
            from cosmos_framework.utils.lazy_config import instantiate as _li  # noqa: F401  (probe import)
            tokenizer_config = dict(_TOKENIZER_CONFIG)
        except Exception as exc:  # pragma: no cover
            print(f"[warn] could not prepare tokenizer config ({exc}); continuing with --no-text behavior.")
            tokenizer_config = None

    print(
        f"[debug] root={root}\n"
        f"[debug] batch_size={args.batch_size} resolution={args.resolution} "
        f"action_normalization={action_normalization} shuffle={not args.no_shuffle} "
        f"augmentation={not args.no_augmentation} text={tokenizer_config is not None}"
    )

    # Build the dataset with the SAME knobs as action_policy_robotwin_nano.py.
    try:
        dataset = get_action_robotwin_sft_dataset(
            root=str(root),
            fps=30.0,
            chunk_length=32,
            mode="policy",
            use_state=True,
            iterable_shuffle=not args.no_shuffle,
            episode_shuffle_seed=42,
            use_image_augmentation=not args.no_augmentation,
            action_normalization=action_normalization,
            viewpoint="concat_view",
            resolution=args.resolution,
            max_action_dim=64,
            cfg_dropout_rate=0.1,
            tokenizer_config=tokenizer_config,
        )
    except Exception as exc:
        if tokenizer_config is not None:
            print(f"[warn] dataset build failed ({exc}); retrying without the tokenizer (--no-text).")
            dataset = get_action_robotwin_sft_dataset(
                root=str(root), fps=30.0, chunk_length=32, mode="policy", use_state=True,
                iterable_shuffle=not args.no_shuffle, episode_shuffle_seed=42,
                use_image_augmentation=not args.no_augmentation, action_normalization=action_normalization,
                viewpoint="concat_view", resolution=args.resolution, max_action_dim=64,
                cfg_dropout_rate=0.1, tokenizer_config=None,
            )
        else:
            raise

    # Pull a batch the same way the trainer streams it.
    if isinstance(dataset, IterableDataset):
        it = iter(dataset)
        samples = [next(it) for _ in range(args.batch_size)]
    else:
        n = min(args.batch_size, len(dataset))
        samples = [dataset[i] for i in range(n)]
    print(f"[debug] pulled {len(samples)} sample(s)")

    batch = custom_collate_fn(samples)
    _print_batch(batch)
    _action_summary(batch)
    _save_videos(batch, out_dir, fps=args.fps)


if __name__ == "__main__":
    main()
