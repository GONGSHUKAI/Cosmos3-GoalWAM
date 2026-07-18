#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Precompute RoboTwin concat-view videos and register them in LeRobot v3 metadata.

This script adds a derived video feature to an existing RoboTwin LeRobot v3 root:

    observation.images.concat_view_384x320

The source camera videos remain untouched. The derived videos are written under
``videos/{feature_key}/chunk-XXX/file-YYY.mp4`` and the episode parquet files are
updated with matching ``videos/{feature_key}/...`` columns.

cd /path/to/cosmos-framework
source .venv/bin/activate

python scripts/preprocess/preprocess_robotwin_offline_concat.py \
--root /path/to/robotwin_lerobot_v3.0

This saves the derived concat video back into that same dataset root. It will create:

videos/observation.images.concat_view_384x320/chunk-000/file-000.mp4

and update the dataset metadata in-place. The script’s default target is 384x320 at scripts/preprocess/preprocess_robotwin_offline_concat.py:303, writes videos
at scripts/preprocess/preprocess_robotwin_offline_concat.py:271, and updates metadata at scripts/preprocess/preprocess_robotwin_offline_concat.py:291.

Optional dry run first:

python scripts/preprocess/preprocess_robotwin_offline_concat.py \
--root /path/to/robotwin_lerobot_v3.0 \
--dry-run

If you later want to regenerate the concat mp4:

python scripts/preprocess/preprocess_robotwin_offline_concat.py \
--root /path/to/robotwin_lerobot_v3.0 \
--overwrite

"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SOURCE_VIDEO_KEYS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)
CONCAT_FEATURE_PREFIX = "observation.images.concat_view"
TARGET_HW_BY_RESOLUTION = {
    "384x320": (384, 320),
    "736x640": (736, 640),
}
VIDEO_INFO = {
    "video.is_depth_map": False,
    "video.fps": 30.0,
    "video.codec": "mp4v",
    "video.pix_fmt": "yuv420p",
    "has_audio": False,
}


def _feature_key(resolution: str) -> str:
    return f"{CONCAT_FEATURE_PREFIX}_{resolution}"


def _load_info(root: Path) -> dict[str, Any]:
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing LeRobot info file: {info_path}")
    with info_path.open("r") as f:
        info = json.load(f)
    if not str(info.get("codebase_version", "")).startswith("v3."):
        raise ValueError(f"Expected LeRobot v3 metadata, got codebase_version={info.get('codebase_version')!r}")
    if "video_path" not in info:
        raise ValueError("meta/info.json does not contain video_path")
    features = info.get("features", {})
    missing = [key for key in SOURCE_VIDEO_KEYS if key not in features]
    if missing:
        raise ValueError(f"Missing source video features in info.json: {missing}")
    return info


def _write_json(path: Path, data: dict[str, Any], *, backup: bool) -> None:
    if backup and path.exists():
        backup_path = path.with_suffix(path.suffix + ".bak")
        if not backup_path.exists():
            shutil.copy2(path, backup_path)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w") as f:
        json.dump(data, f, indent=4)
        f.write("\n")
    tmp_path.replace(path)


def _video_path(root: Path, info: dict[str, Any], video_key: str, chunk_index: int, file_index: int) -> Path:
    rel = info["video_path"].format(
        video_key=video_key,
        chunk_index=chunk_index,
        file_index=file_index,
        episode_chunk=chunk_index,
        episode_file=file_index,
        episode_index=file_index,
    )
    return root / rel


def _open_capture(path: Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    return cap


def _frame_count(cap: cv2.VideoCapture) -> int:
    return int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))


def _source_fps(caps: tuple[cv2.VideoCapture, cv2.VideoCapture, cv2.VideoCapture], fallback: float) -> float:
    fps_values = [float(cap.get(cv2.CAP_PROP_FPS)) for cap in caps]
    valid = [fps for fps in fps_values if fps > 0]
    if not valid:
        return fallback
    return valid[0]


def _resize(frame: Any, size_hw: tuple[int, int]) -> Any:
    height, width = size_hw
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)


def _compose_frame(
    head_bgr: Any,
    left_bgr: Any,
    right_bgr: Any,
    target_hw: tuple[int, int],
) -> Any:
    head_h, head_w = head_bgr.shape[:2]
    half_hw = (head_h // 2, head_w // 2)
    left_small = _resize(left_bgr, half_hw)
    right_small = _resize(right_bgr, half_hw)
    bottom = cv2.hconcat([left_small, right_small])
    concat = cv2.vconcat([head_bgr, bottom])
    if concat.shape[:2] != target_hw:
        concat = _resize(concat, target_hw)
    return concat


def _fourcc(codec: str) -> int:
    if codec == "mp4v":
        return cv2.VideoWriter_fourcc(*"mp4v")
    if codec == "avc1":
        return cv2.VideoWriter_fourcc(*"avc1")
    raise ValueError(f"Unsupported codec={codec!r}; expected 'mp4v' or 'avc1'")


@dataclass(frozen=True)
class _EpisodeSegment:
    episode_index: int
    length: int
    source_paths: tuple[Path, Path, Path]
    source_from_timestamps: tuple[float, float, float]


@dataclass(frozen=True)
class _ConcatVideoJob:
    idx: int
    total: int
    output_path: Path
    target_hw: tuple[int, int]
    fps: float
    codec: str
    whole_source_paths: tuple[Path, Path, Path] | None = None
    segments: tuple[_EpisodeSegment, ...] = ()


class _FfmpegRawVideoReader:
    def __init__(self, path: Path, width: int, height: int) -> None:
        self.path = path
        self.width = width
        self.height = height
        self._frame_bytes = width * height * 3
        cmd = [
            "ffmpeg",
            "-v",
            "error",
            "-c:v",
            "libdav1d",
            "-i",
            str(path),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-",
        ]
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        if self._proc.stdout is None:
            raise RuntimeError(f"Could not open ffmpeg stdout for {path}")
        self._stdout: BinaryIO = self._proc.stdout

    def read(self) -> tuple[bool, Any]:
        raw = self._stdout.read(self._frame_bytes)
        if len(raw) != self._frame_bytes:
            return False, None
        return True, np.frombuffer(raw, dtype=np.uint8).reshape((self.height, self.width, 3))

    def release(self) -> None:
        self._stdout.close()
        self._proc.wait()


class _SequentialVideoReader:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.cap = _open_capture(path)
        self.reader: _FfmpegRawVideoReader | None = None
        self.current_frame = 0

    def _ensure_ffmpeg_reader(self) -> _FfmpegRawVideoReader:
        if self.reader is None:
            width = int(round(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
            height = int(round(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            if width <= 0 or height <= 0:
                raise RuntimeError(f"Could not determine video size for ffmpeg fallback: {self.path}")
            self.cap.release()
            self.reader = _FfmpegRawVideoReader(self.path, width, height)
        return self.reader

    def reopen(self) -> None:
        self.release()
        self.cap = _open_capture(self.path)
        self.reader = None
        self.current_frame = 0

    def read(self) -> tuple[bool, Any]:
        if self.reader is None:
            ok, frame = self.cap.read()
            if not ok:
                ok, frame = self._ensure_ffmpeg_reader().read()
        else:
            ok, frame = self.reader.read()
        if ok:
            self.current_frame += 1
        return ok, frame

    def advance_to(self, target_frame: int) -> None:
        if target_frame < self.current_frame:
            self.reopen()
        while self.current_frame < target_frame:
            ok, _frame = self.read()
            if not ok:
                raise RuntimeError(
                    f"Failed advancing {self.path} to frame {target_frame}; stopped at {self.current_frame}"
                )

    def release(self) -> None:
        self.cap.release()
        if self.reader is not None:
            self.reader.release()


def _make_ffmpeg_readers(
    source_paths: tuple[Path, Path, Path],
    caps: tuple[cv2.VideoCapture, cv2.VideoCapture, cv2.VideoCapture],
) -> tuple[_FfmpegRawVideoReader, _FfmpegRawVideoReader, _FfmpegRawVideoReader]:
    readers = []
    for path, cap in zip(source_paths, caps, strict=True):
        width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        if width <= 0 or height <= 0:
            raise RuntimeError(f"Could not determine video size for ffmpeg fallback: {path}")
        cap.release()
        readers.append(_FfmpegRawVideoReader(path, width, height))
    return tuple(readers)  # type: ignore[return-value]


def _write_concat_video_job(job: _ConcatVideoJob) -> tuple[int, int, Path, int, bool]:
    if job.output_path.exists():
        return job.idx, job.total, job.output_path, 0, True
    if job.whole_source_paths is not None:
        frames = _write_concat_video(
            source_paths=job.whole_source_paths,
            output_path=job.output_path,
            target_hw=job.target_hw,
            fps_fallback=job.fps,
            codec=job.codec,
        )
    else:
        frames = _write_concat_video_segments(
            segments=job.segments,
            output_path=job.output_path,
            target_hw=job.target_hw,
            fps=job.fps,
            codec=job.codec,
        )
    return job.idx, job.total, job.output_path, frames, False


def _write_concat_video(
    source_paths: tuple[Path, Path, Path],
    output_path: Path,
    target_hw: tuple[int, int],
    fps_fallback: float,
    codec: str,
) -> int:
    caps = tuple(_open_capture(path) for path in source_paths)
    writer: cv2.VideoWriter | None = None
    frames_written = 0
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp.mp4")
    try:
        counts = [_frame_count(cap) for cap in caps]
        frame_count = min(counts)
        if frame_count <= 0:
            raise RuntimeError(f"No frames found in source videos: {source_paths}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fps = _source_fps(caps, fps_fallback)
        target_h, target_w = target_hw
        writer = cv2.VideoWriter(str(tmp_path), _fourcc(codec), fps, (target_w, target_h))
        if not writer.isOpened() and codec != "mp4v":
            writer.release()
            writer = cv2.VideoWriter(str(tmp_path), _fourcc("mp4v"), fps, (target_w, target_h))
        if not writer.isOpened():
            raise RuntimeError(f"Could not open VideoWriter for {tmp_path}")

        readers: tuple[_FfmpegRawVideoReader, _FfmpegRawVideoReader, _FfmpegRawVideoReader] | None = None
        for frame_idx in range(frame_count):
            if readers is None:
                ok_head, head = caps[0].read()
                ok_left, left = caps[1].read()
                ok_right, right = caps[2].read()
                if frame_idx == 0 and not (ok_head and ok_left and ok_right):
                    readers = _make_ffmpeg_readers(source_paths, caps)
                    ok_head, head = readers[0].read()
                    ok_left, left = readers[1].read()
                    ok_right, right = readers[2].read()
            else:
                ok_head, head = readers[0].read()
                ok_left, left = readers[1].read()
                ok_right, right = readers[2].read()
            if not (ok_head and ok_left and ok_right):
                raise RuntimeError(f"Failed reading source frame {frame_idx} from {source_paths}")
            writer.write(_compose_frame(head, left, right, target_hw))
            frames_written += 1
    finally:
        for cap in caps:
            cap.release()
        if "readers" in locals() and readers is not None:
            for reader in readers:
                reader.release()
        if writer is not None:
            writer.release()

    tmp_path.replace(output_path)
    return frames_written


def _write_concat_video_segments(
    segments: tuple[_EpisodeSegment, ...],
    output_path: Path,
    target_hw: tuple[int, int],
    fps: float,
    codec: str,
) -> int:
    if not segments:
        raise RuntimeError(f"No episode segments for {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    target_h, target_w = target_hw
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp.mp4")
    writer = cv2.VideoWriter(str(tmp_path), _fourcc(codec), fps, (target_w, target_h))
    if not writer.isOpened() and codec != "mp4v":
        writer.release()
        writer = cv2.VideoWriter(str(tmp_path), _fourcc("mp4v"), fps, (target_w, target_h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {tmp_path}")

    readers: dict[Path, _SequentialVideoReader] = {}
    frames_written = 0
    try:
        for segment in segments:
            segment_readers = []
            for source_path, source_from_timestamp in zip(
                segment.source_paths, segment.source_from_timestamps, strict=True
            ):
                reader = readers.get(source_path)
                if reader is None:
                    reader = _SequentialVideoReader(source_path)
                    readers[source_path] = reader
                reader.advance_to(int(round(source_from_timestamp * fps)))
                segment_readers.append(reader)

            for frame_offset in range(segment.length):
                frames = []
                for reader in segment_readers:
                    ok, frame = reader.read()
                    if not ok:
                        raise RuntimeError(
                            f"Failed reading episode={segment.episode_index} frame_offset={frame_offset} "
                            f"from {reader.path}"
                        )
                    frames.append(frame)
                writer.write(_compose_frame(frames[0], frames[1], frames[2], target_hw))
                frames_written += 1
    finally:
        for reader in readers.values():
            reader.release()
        writer.release()

    tmp_path.replace(output_path)
    return frames_written


def _copy_episode_video_columns(row: dict[str, Any], source_key: str, dest_key: str) -> None:
    for suffix in ("chunk_index", "file_index", "from_timestamp", "to_timestamp"):
        source_col = f"videos/{source_key}/{suffix}"
        dest_col = f"videos/{dest_key}/{suffix}"
        if source_col not in row:
            raise KeyError(f"Episode metadata is missing source column {source_col!r}")
        row[dest_col] = row[source_col]


def _update_episode_metadata(root: Path, dest_key: str, *, dry_run: bool, backup: bool) -> int:
    episode_paths = sorted((root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    if not episode_paths:
        raise FileNotFoundError(f"No episode parquet files found under {root / 'meta' / 'episodes'}")

    updated_rows = 0
    for path in episode_paths:
        table = pq.read_table(path)
        rows = table.to_pylist()
        for row in rows:
            _copy_episode_video_columns(row, SOURCE_VIDEO_KEYS[0], dest_key)
        updated_rows += len(rows)
        if dry_run:
            continue
        if backup:
            backup_path = path.with_suffix(path.suffix + ".bak")
            if not backup_path.exists():
                shutil.copy2(path, backup_path)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        pq.write_table(pa.Table.from_pylist(rows), tmp_path)
        tmp_path.replace(path)
    return updated_rows


def _update_info(
    root: Path,
    info: dict[str, Any],
    dest_key: str,
    target_hw: tuple[int, int],
    *,
    codec: str,
    dry_run: bool,
    backup: bool,
) -> None:
    target_h, target_w = target_hw
    features = info.setdefault("features", {})
    features[dest_key] = {
        "dtype": "video",
        "video_info": dict(VIDEO_INFO, **{"video.fps": float(info.get("fps", 30.0)), "video.codec": codec}),
        "shape": [target_h, target_w, 3],
        "names": ["height", "width", "channel"],
    }
    if not dry_run:
        _write_json(root / "meta" / "info.json", info, backup=backup)


def _iter_video_jobs(
    root: Path,
    info: dict[str, Any],
    dest_key: str,
    target_hw: tuple[int, int],
    *,
    codec: str,
) -> list[_ConcatVideoJob]:
    episode_paths = sorted((root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    rows = [row for path in episode_paths for row in pq.read_table(path).to_pylist()]
    if not rows:
        return []

    source_indices = {
        source_key: {
            (
                int(row[f"videos/{source_key}/chunk_index"]),
                int(row[f"videos/{source_key}/file_index"]),
            )
            for row in rows
        }
        for source_key in SOURCE_VIDEO_KEYS
    }
    aligned = all(
        len(
            {
                (
                    int(row[f"videos/{source_key}/chunk_index"]),
                    int(row[f"videos/{source_key}/file_index"]),
                )
                for source_key in SOURCE_VIDEO_KEYS
            }
        )
        == 1
        and max(float(row[f"videos/{source_key}/from_timestamp"]) for source_key in SOURCE_VIDEO_KEYS)
        - min(float(row[f"videos/{source_key}/from_timestamp"]) for source_key in SOURCE_VIDEO_KEYS)
        < 1e-4
        and max(float(row[f"videos/{source_key}/to_timestamp"]) for source_key in SOURCE_VIDEO_KEYS)
        - min(float(row[f"videos/{source_key}/to_timestamp"]) for source_key in SOURCE_VIDEO_KEYS)
        < 1e-4
        for row in rows
    )
    fps = float(info.get("fps", 30.0))

    jobs: list[_ConcatVideoJob] = []
    if aligned:
        for idx, (chunk_idx, file_idx) in enumerate(sorted(source_indices[SOURCE_VIDEO_KEYS[0]]), start=1):
            source_paths = tuple(
                _video_path(root, info, source_key, chunk_idx, file_idx) for source_key in SOURCE_VIDEO_KEYS
            )
            output_path = _video_path(root, info, dest_key, chunk_idx, file_idx)
            jobs.append(
                _ConcatVideoJob(
                    idx=idx,
                    total=len(source_indices[SOURCE_VIDEO_KEYS[0]]),
                    output_path=output_path,
                    target_hw=target_hw,
                    fps=fps,
                    codec=codec,
                    whole_source_paths=source_paths,
                )
            )
        return jobs

    grouped_rows: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            int(row[f"videos/{SOURCE_VIDEO_KEYS[0]}/chunk_index"]),
            int(row[f"videos/{SOURCE_VIDEO_KEYS[0]}/file_index"]),
        )
        grouped_rows.setdefault(key, []).append(row)

    for idx, ((chunk_idx, file_idx), group) in enumerate(sorted(grouped_rows.items()), start=1):
        group = sorted(group, key=lambda row: int(row["episode_index"]))
        segments = []
        for row in group:
            source_paths = tuple(
                _video_path(
                    root,
                    info,
                    source_key,
                    int(row[f"videos/{source_key}/chunk_index"]),
                    int(row[f"videos/{source_key}/file_index"]),
                )
                for source_key in SOURCE_VIDEO_KEYS
            )
            from_timestamps = tuple(
                float(row[f"videos/{source_key}/from_timestamp"]) for source_key in SOURCE_VIDEO_KEYS
            )
            segments.append(
                _EpisodeSegment(
                    episode_index=int(row["episode_index"]),
                    length=int(row["length"]),
                    source_paths=source_paths,
                    source_from_timestamps=from_timestamps,
                )
            )
        output_path = _video_path(root, info, dest_key, chunk_idx, file_idx)
        jobs.append(
            _ConcatVideoJob(
                idx=idx,
                total=len(grouped_rows),
                output_path=output_path,
                target_hw=target_hw,
                fps=fps,
                codec=codec,
                segments=tuple(segments),
            )
        )
    return jobs


def preprocess(args: argparse.Namespace) -> None:
    root = Path(args.root).expanduser().resolve()
    info = _load_info(root)
    target_hw = TARGET_HW_BY_RESOLUTION[args.resolution]
    dest_key = args.dest_key or _feature_key(args.resolution)
    jobs = _iter_video_jobs(root, info, dest_key, target_hw, codec=args.codec)
    if args.limit is not None:
        jobs = jobs[: args.limit]

    print(f"[robotwin-offline-concat] root={root}")
    print(f"[robotwin-offline-concat] feature={dest_key} resolution={args.resolution} target_hw={target_hw}")
    print(f"[robotwin-offline-concat] videos={len(jobs)} overwrite={args.overwrite} dry_run={args.dry_run}")

    if not args.metadata_only:
        write_jobs = []
        for job in jobs:
            if job.whole_source_paths is not None:
                source_paths = list(job.whole_source_paths)
            else:
                source_paths = [path for segment in job.segments for path in segment.source_paths]
            missing = sorted({path for path in source_paths if not path.exists()})
            if missing:
                raise FileNotFoundError(f"Missing source video(s) for {job.output_path}: {missing}")
            if job.output_path.exists() and not args.overwrite:
                print(f"[{job.idx}/{job.total}] skip existing {job.output_path}")
                continue
            if args.dry_run:
                print(f"[{job.idx}/{job.total}] would write {job.output_path}")
                continue
            if job.output_path.exists():
                job.output_path.unlink()
            write_jobs.append(job)
        if args.workers <= 1:
            for job in write_jobs:
                idx, total, output_path, frames, _skipped = _write_concat_video_job(job)
                print(f"[{idx}/{total}] wrote {output_path} frames={frames}")
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                futures = [executor.submit(_write_concat_video_job, job) for job in write_jobs]
                for future in as_completed(futures):
                    idx, total, output_path, frames, skipped = future.result()
                    action = "skip existing" if skipped else "wrote"
                    suffix = "" if skipped else f" frames={frames}"
                    print(f"[{idx}/{total}] {action} {output_path}{suffix}")

    _update_info(root, info, dest_key, target_hw, codec=args.codec, dry_run=args.dry_run, backup=not args.no_backup)
    updated_rows = _update_episode_metadata(root, dest_key, dry_run=args.dry_run, backup=not args.no_backup)
    print(f"[robotwin-offline-concat] metadata rows updated={updated_rows}")


def parse_args() -> argparse.Namespace:
    data_base = Path(os.environ.get("ROBOTWIN_DATA_ROOT", Path.cwd() / "data"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default=str(data_base / "robotwin2.0/place_a2b_left/aloha-agilex_combined_550_lerobot_v3.0"),
        help="RoboTwin LeRobot v3 dataset root.",
    )
    parser.add_argument(
        "--resolution",
        choices=sorted(TARGET_HW_BY_RESOLUTION),
        default="384x320",
        help="Offline concat target resolution.",
    )
    parser.add_argument("--dest-key", default=None, help="Override output video feature key.")
    parser.add_argument("--codec", choices=("mp4v", "avc1"), default="mp4v", help="OpenCV VideoWriter codec.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing offline concat videos.")
    parser.add_argument("--metadata-only", action="store_true", help="Only update info/episode metadata.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned work without writing videos or metadata.")
    parser.add_argument("--no-backup", action="store_true", help="Do not create .bak metadata backups.")
    parser.add_argument("--workers", type=int, default=1, help="Number of concat videos to generate in parallel.")
    parser.add_argument(
        "--limit", type=int, default=None, help="Process only the first N video files for smoke testing."
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    preprocess(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
