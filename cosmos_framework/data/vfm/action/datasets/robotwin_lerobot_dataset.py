# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Minimal RoboTwin (aloha-agilex) LeRobot dataset for Cosmos Action.

Mirrors :class:`DROIDLeRobotDataset` but for RoboTwin's dual-arm aloha-agilex
embodiment:

  * 14D absolute joint action ``[left_arm(6), left_gripper(1), right_arm(6),
    right_gripper(1)]`` (RoboTwin ``joint_action/vector``). Raw (un-normalized) by
    default; set ``action_normalization="meanstd"`` for the z-score ablation
    (per-joint stats in ``stats/robotwin_lerobot_stats.json``).
  * ``use_state=True`` prepends the initial observed 14D state -> ``(chunk+1, 14)``.
  * Optional FastWAM-style video-only 4x temporal downsampling keeps frames
    ``0,4,8,...,32`` while the action tensor remains ``chunk+1`` rows.
  * 3 cameras -> ``concat_view``: head (``cam_high``) full-width on top;
    left/right wrists resized to half size and concatenated on the bottom row.
    The resulting canvas is then resized to the selected no-padding RoboTwin
    bucket (default ``384x320``; ``736x640`` is available for 640x480 cameras).

The data is lerobot v3.0 (aggregated ``data/chunk-*/file-*.parquet`` + per-camera
aggregated mp4 with per-episode ``from_timestamp`` offsets in ``meta/episodes``),
so :class:`ActionBaseDataset` loads ``_info``/``_episodes``/``_tasks``/``_rows``
natively. RoboTwin single-task splits are small (~80k frames), so we index the
base ``_rows`` directly rather than the compact-array path DROID uses for its
18M-frame shard. No keep-ranges filter and no ee_pose layout.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from lerobot.datasets.video_utils import decode_video_frames

from cosmos_framework.data.vfm.action.action_normalization import load_action_stats
from cosmos_framework.data.vfm.action.action_spec import ActionSpec, Gripper, Joint, build_action_spec
from cosmos_framework.data.vfm.action.datasets.base_dataset import ActionBaseDataset
from cosmos_framework.data.vfm.action.domain_utils import get_domain_id

PoseConvention = Literal["backward_framewise"]
Viewpoint = Literal["concat_view"]

# RoboTwin camera -> lerobot video key (set by convert_robotwin_to_lerobot.py:
# head_camera -> cam_high, left_camera -> cam_left_wrist, right_camera -> cam_right_wrist).
_IMAGE_FEATURES = {
    "head": "observation.images.cam_high",
    "left": "observation.images.cam_left_wrist",
    "right": "observation.images.cam_right_wrist",
}
# 14D joint vector lives in a single ``action`` column; the per-frame observed
# state is an identical 14D ``observation.state`` column (the RoboTwin converter
# sets both = ``joint_action/vector``). Gripper is NOT flipped (unlike DROID): we
# emit the raw vector so it feeds RoboTwin's joint interface directly at eval time.
_ACTION_FEATURE = "action"
_STATE_FEATURE = "observation.state"

_NORMALIZER_PATH = Path(__file__).parent / "stats/robotwin_lerobot_stats.json"

# Fixed no-padding RoboTwin concat target canvases used for both training and eval.
# Keys match ``VIDEO_RES_SIZE_INFO`` entries; sizes are (H, W).
_CONCAT_TARGET_HW_BY_RESOLUTION = {
    "384x320": (384, 320),
    "736x640": (736, 640),
}


class RoboTwinLeRobotDataset(ActionBaseDataset):
    """RoboTwin aloha-agilex Action dataset (14D absolute joint, raw/un-normalized)."""

    def __init__(
        self,
        root: str,
        fps: float = 30.0,
        chunk_length: int = 32,
        mode: str = "policy",
        pose_convention: PoseConvention = "backward_framewise",
        tolerance_s: float = 2e-4,
        viewpoint: Viewpoint = "concat_view",
        use_state: bool = True,
        action_normalization: str | None = None,
        use_image_augmentation: bool = False,
        target_resolution: str = "384x320",
        video_downsample_factor: int = 1,
    ) -> None:
        if viewpoint != "concat_view":
            raise NotImplementedError("RoboTwinLeRobotDataset only supports concat_view.")

        root_path = Path(root)
        info = json.loads((root_path / "meta" / "info.json").read_text())
        self._is_lerobot_v21 = str(info.get("codebase_version", "")).startswith("v2.")

        # 14D joint values are raw absolute angles -> disable normalization at the base level.
        if self._is_lerobot_v21:
            self._init_lerobot_v21(
                root=root_path,
                info=info,
                fps=fps,
                chunk_length=chunk_length,
                mode=mode,
                pose_convention=pose_convention,
                tolerance_s=tolerance_s,
                viewpoint=viewpoint,
                action_normalization=action_normalization,
            )
        else:
            super().__init__(
                root=root,
                domain_name="robotwin",
                fps=fps,
                chunk_length=chunk_length,
                mode=mode,
                pose_convention=pose_convention,
                tolerance_s=tolerance_s,
                viewpoint=viewpoint,
                action_normalization=action_normalization,
            )

        self._use_state = bool(use_state)
        self._use_image_augmentation = bool(use_image_augmentation)
        self._video_downsample_factor = int(video_downsample_factor)
        if self._video_downsample_factor < 1:
            raise ValueError(f"video_downsample_factor must be >= 1, got {self._video_downsample_factor}")
        if self._chunk_length % self._video_downsample_factor != 0:
            raise ValueError(
                f"chunk_length={self._chunk_length} must be divisible by "
                f"video_downsample_factor={self._video_downsample_factor}"
            )
        if target_resolution not in _CONCAT_TARGET_HW_BY_RESOLUTION:
            raise ValueError(
                f"Unsupported RoboTwin target_resolution={target_resolution!r}; "
                f"expected one of {sorted(_CONCAT_TARGET_HW_BY_RESOLUTION)}"
            )
        self._target_resolution = target_resolution
        self._target_hw = _CONCAT_TARGET_HW_BY_RESOLUTION[target_resolution]
        self._image_augmentor: T.Compose | None = None

        # Episode boundaries over the base ``_rows`` (sorted by global ``index``; v3.0
        # orders frames by episode, so episodes are contiguous blocks). Valid windows
        # per episode = max(0, length - chunk_length); a window never crosses episodes.
        if not self._is_lerobot_v21:
            self._row_episode = np.asarray([int(r["episode_index"]) for r in self._rows], dtype=np.int64)
            assert np.all(np.diff(self._row_episode) >= 0), "episode_index is not contiguous after sorting by index"
            ep_vals, ep_starts, ep_counts = np.unique(self._row_episode, return_index=True, return_counts=True)
            self._ep_vals = ep_vals.astype(np.int64)
            self._ep_starts = ep_starts.astype(np.int64)
            self._valid_cum = np.cumsum(np.maximum(0, ep_counts - self._chunk_length)).astype(np.int64)

    def _init_lerobot_v21(
        self,
        *,
        root: Path,
        info: dict[str, Any],
        fps: float,
        chunk_length: int,
        mode: str,
        pose_convention: PoseConvention,
        tolerance_s: float,
        viewpoint: Viewpoint,
        action_normalization: str | None,
    ) -> None:
        if pose_convention != "backward_framewise":
            raise NotImplementedError(f"{type(self).__name__} only supports backward_framewise pose deltas.")

        self._fps = float(fps)
        self._dt = 1.0 / self._fps
        self._chunk_length = int(chunk_length)
        self._sample_stride = 1
        self._mode = mode
        self._pose_convention = pose_convention
        self._tolerance_s = float(tolerance_s)
        self._viewpoint = viewpoint
        self._domain_name = "robotwin"
        self._domain_id = get_domain_id("robotwin")
        self._action_normalization = action_normalization
        self._norm_stats: dict[str, torch.Tensor] | None = None
        self._root = root
        self._info = info
        self._chunks_size = int(info.get("chunks_size", 1000))
        self._rows: list[dict[str, Any]] = []
        self._tasks: dict[int, str] = {}

        episodes: list[dict[str, Any]] = []
        episodes_path = root / "meta" / "episodes.jsonl"
        with episodes_path.open() as f:
            for line in f:
                row = json.loads(line)
                episode_index = int(row["episode_index"])
                length = int(row["length"])
                chunk_idx = episode_index // self._chunks_size
                data_path = root / info["data_path"].format(
                    episode_chunk=chunk_idx,
                    episode_index=episode_index,
                    chunk_index=chunk_idx,
                    file_index=episode_index,
                    episode_file=episode_index,
                )
                episodes.append(
                    {
                        "episode_index": episode_index,
                        "episode_chunk": chunk_idx,
                        "data/chunk_index": chunk_idx,
                        "length": length,
                        "tasks": [str(t) for t in row.get("tasks", [])],
                        "data_path": data_path,
                    }
                )

        self._episodes = {int(ep["episode_index"]): ep for ep in episodes}
        self._ep_records = episodes
        ep_counts = np.asarray([int(ep["length"]) for ep in episodes], dtype=np.int64)
        self._valid_cum = np.cumsum(np.maximum(0, ep_counts - self._chunk_length)).astype(np.int64)

    @property
    def action_dim(self) -> int:
        return 14

    def _action_spec(self) -> ActionSpec:
        return build_action_spec(
            Joint(n=6, prefix="left"),
            Gripper(prefix="left"),
            Joint(n=6, prefix="right"),
            Gripper(prefix="right"),
        )

    @classmethod
    def _stats_path(cls) -> Path:
        # Only consulted when action_normalization is not None. The default recipe
        # uses raw joint values (None). For the normalization ablation
        # (action_normalization="meanstd"), the bundled stats/robotwin_lerobot_stats.json
        # holds the 14-D action mean/std/min/max for place_a2b_left (550 episodes);
        # point this elsewhere for a different RoboTwin dataset.
        return _NORMALIZER_PATH

    @classmethod
    def load_action_stats(cls) -> dict[str, torch.Tensor]:
        """Load RoboTwin action stats.

        The bundled stats file uses the standard Cosmos action format
        (top-level ``mean`` / ``std``).  For convenience, this also accepts a
        FastWAM-style ``action/default/global_*`` JSON if the path is replaced
        for another RoboTwin dataset.
        """
        raw = load_action_stats(str(cls._stats_path()))
        if raw:
            return {key: torch.from_numpy(value).float() for key, value in raw.items()}

        path = cls._stats_path()
        with path.open("r") as f:
            stats_json = json.load(f)
        action_default = stats_json.get("action", {}).get("default", {})
        if not action_default:
            raise ValueError(f"No supported RoboTwin action stats found in {path}")
        key_map = {
            "mean": "global_mean",
            "std": "global_std",
            "min": "global_min",
            "max": "global_max",
            "q01": "global_q01",
            "q99": "global_q99",
        }
        return {
            key: torch.tensor(action_default[source_key], dtype=torch.float32)
            for key, source_key in key_map.items()
            if source_key in action_default
        }

    def __len__(self) -> int:
        return int(self._valid_cum[-1]) if self._valid_cum.size else 0

    def __getitem__(self, idx: int) -> dict[str, Any]:
        mode = self._choose_mode()
        idx = int(idx)
        # Map the flat sample index to a within-episode frame window.
        ep = int(np.searchsorted(self._valid_cum, idx, side="right"))
        prev = int(self._valid_cum[ep - 1]) if ep > 0 else 0
        offset = idx - prev

        if self._is_lerobot_v21:
            episode = self._ep_records[ep]
            observation_rows = self._load_lerobot_v21_rows(episode, int(offset), self._chunk_length + 1)
            task_candidates = episode.get("tasks", [])
            ai_caption = random.choice(task_candidates) if task_candidates else str(observation_rows[0]["task_index"])
        else:
            start = int(self._ep_starts[ep]) + offset
            episode_index = int(self._ep_vals[ep])
            episode = self._episodes[episode_index]

            # chunk+1 frames: row[0] is the initial observed state, rows[1:] the action chunk.
            observation_rows = self._rows[start : start + self._chunk_length + 1]
            task = self._tasks[int(observation_rows[0]["task_index"])]
            ai_caption = random.choice(task.split(" | "))

        video = self._load_concat_video(episode, observation_rows)
        raw_action = self._build_joint_action(observation_rows)

        result = self._build_result(
            mode=mode,
            video=video,
            action=raw_action,
            ai_caption=ai_caption,
            additional_view_description=(
                "The top row is from the head camera looking at the dual-arm robot and workspace. "
                "The bottom row contains two horizontally concatenated wrist-camera views, "
                "left-arm wrist on the left and right-arm wrist on the right."
            ),
        )
        if self._video_downsample_factor > 1:
            result["conditioning_fps"] = torch.tensor(self._fps / self._video_downsample_factor, dtype=torch.float32)
            result["action_fps"] = torch.tensor(self._fps, dtype=torch.float32)
        return result

    def _load_lerobot_v21_rows(self, episode: dict[str, Any], start: int, length: int) -> list[dict[str, Any]]:
        table = pq.read_table(
            episode["data_path"],
            columns=[_STATE_FEATURE, _ACTION_FEATURE, "timestamp", "frame_index", "episode_index", "index", "task_index"],
        )
        return table.slice(start, length).to_pylist()

    def _build_joint_action(self, observation_rows: list[dict[str, Any]]) -> torch.Tensor:
        """14D absolute joint action over the chunk. The window is ``chunk+1`` frames:
        ``row[0]`` is the initial observed state (prepended when ``use_state``), and
        ``rows[1:]`` are the ``chunk`` commanded actions. No gripper flip, no normalization."""
        action_rows = observation_rows[1:]
        action = np.asarray([r[_ACTION_FEATURE] for r in action_rows], dtype=np.float32)  # [chunk, 14]
        if self._use_state:
            initial_state = np.asarray(observation_rows[0][_STATE_FEATURE], dtype=np.float32)[None, :]  # [1, 14]
            action = np.concatenate([initial_state, action], axis=0)  # [chunk + 1, 14]
        return torch.from_numpy(action).float()

    def _load_concat_video(
        self,
        episode: dict[str, Any],
        observation_rows: list[dict[str, Any]],
    ) -> torch.Tensor:
        video_rows = observation_rows[:: self._video_downsample_factor]
        timestamps = [float(row["timestamp"]) for row in video_rows]
        frames_by_view = {
            name: decode_video_frames(
                self._video_path(episode, video_key),
                [float(episode.get(f"videos/{video_key}/from_timestamp", 0.0)) + ts for ts in timestamps],
                self._tolerance_s,
            )
            for name, video_key in _IMAGE_FEATURES.items()
        }

        head = frames_by_view["head"]
        left = frames_by_view["left"]
        right = frames_by_view["right"]

        if self._use_image_augmentation:
            # Random crop+rescale (spatial jitter) + color jitter, BEFORE the concat.
            # All three views are stacked so one sampled set of params is applied
            # uniformly across every frame and view (temporally + cross-view consistent),
            # while each __getitem__ resamples.
            if self._image_augmentor is None:
                _, _, h, w = head.shape
                self._image_augmentor = T.Compose(
                    [
                        T.RandomCrop((int(h * 0.95), int(w * 0.95))),
                        T.Resize((h, w), antialias=True),
                        T.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5, hue=0.08),
                    ]
                )
            n, m = head.shape[0], head.shape[0] + left.shape[0]
            combined = self._image_augmentor(torch.cat([head, left, right], dim=0))
            head, left, right = combined[:n], combined[n:m], combined[m:]

        _, _, h_h, w_h = head.shape
        half_h, half_w = h_h // 2, w_h // 2
        left = F.interpolate(left, size=(half_h, half_w), mode="bilinear", align_corners=False)
        right = F.interpolate(right, size=(half_h, half_w), mode="bilinear", align_corners=False)
        bottom = torch.cat([left, right], dim=-1)
        concat = torch.cat([head, bottom], dim=-2)
        if concat.shape[-2:] != self._target_hw:
            concat = F.interpolate(concat, size=self._target_hw, mode="bilinear", align_corners=False)
        return concat

    def _video_path(self, episode: dict[str, Any], video_key: str) -> Path:
        if not self._is_lerobot_v21:
            return super()._video_path(episode, video_key)

        episode_index = int(episode["episode_index"])
        chunk_idx = int(episode.get("episode_chunk", episode_index // self._chunks_size))
        rel = self._info["video_path"].format(
            video_key=video_key,
            chunk_index=chunk_idx,
            file_index=episode_index,
            episode_chunk=chunk_idx,
            episode_file=episode_index,
            episode_index=episode_index,
        )
        return self._root / rel

    def get_shuffle_blocks(self) -> list[tuple[int, int]]:
        """Per-episode flat-index blocks ``(start, length)`` over valid windows.
        ``ActionIterableShuffleDataset`` shuffles the ORDER of these blocks and shards
        them disjointly across ranks, while keeping windows *within* a block sequential."""
        blocks: list[tuple[int, int]] = []
        prev = 0
        for c in np.asarray(self._valid_cum).tolist():
            c = int(c)
            if c > prev:
                blocks.append((prev, c - prev))
            prev = c
        return blocks
