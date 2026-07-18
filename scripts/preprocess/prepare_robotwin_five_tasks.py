#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Prepare a five-task RoboTwin aloha-agilex LeRobot v3 dataset.

This helper mirrors the local place_a2b_left workflow:

1. Unzip each task's clean/randomized aloha-agilex archives if needed.
2. Build one raw combined RoboTwin directory with contiguous episode indices.
3. Convert that raw directory to LeRobot v2.1 with RoboTwin's converter.
4. Convert the v2.1 dataset to LeRobot v3.0 with LeRobot's v30 converter.

The offline concat video feature is intentionally handled by
``scripts/preprocess/preprocess_robotwin_offline_concat.py`` after this script
finishes, matching the existing training path.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from lerobot.datasets.v30 import convert_dataset_v21_to_v30 as v30_converter

TASKS = (
    "hanging_mug",
    "move_stapler_pad",
    "open_microwave",
    "stack_bowls_three",
    "turn_switch",
)
SPLITS = ("aloha-agilex_clean_50", "aloha-agilex_randomized_500")
RAW_COMBINED_NAME = "aloha-agilex_combined_2750"
V21_NAME = f"{RAW_COMBINED_NAME}_lerobot_v2.1"
V30_NAME = f"{RAW_COMBINED_NAME}_lerobot_v3.0"


def _episode_id(path: Path) -> int:
    return int(path.stem.replace("episode", ""))


def _run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _unzip_archives(data_root: Path, *, force: bool) -> None:
    for task in TASKS:
        task_root = data_root / task
        for split in SPLITS:
            archive = task_root / f"{split}.zip"
            out_dir = task_root / split
            if not archive.exists():
                raise FileNotFoundError(f"Missing archive: {archive}")
            if out_dir.exists() and not force:
                print(f"[skip] existing {out_dir}", flush=True)
                continue
            if out_dir.exists() and force:
                shutil.rmtree(out_dir)
            _run(["unzip", "-q", str(archive), "-d", str(task_root)])


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _build_raw_combined(data_root: Path, output_root: Path, *, overwrite: bool) -> Path:
    raw_root = output_root / RAW_COMBINED_NAME
    if raw_root.exists():
        if not overwrite:
            print(f"[skip] existing raw combined dataset: {raw_root}", flush=True)
            return raw_root
        shutil.rmtree(raw_root)

    (raw_root / "data").mkdir(parents=True)
    (raw_root / "instructions").mkdir()

    manifest: list[dict[str, Any]] = []
    next_episode = 0
    for task in TASKS:
        task_root = data_root / task
        for split in SPLITS:
            split_root = task_root / split
            data_dir = split_root / "data"
            instructions_dir = split_root / "instructions"
            if not data_dir.is_dir() or not instructions_dir.is_dir():
                raise FileNotFoundError(f"Missing raw RoboTwin data/instructions under {split_root}")

            h5_files = sorted(data_dir.glob("episode*.hdf5"), key=_episode_id)
            if not h5_files:
                raise FileNotFoundError(f"No episode*.hdf5 files under {data_dir}")

            for h5_path in h5_files:
                old_episode = _episode_id(h5_path)
                instruction_path = instructions_dir / f"episode{old_episode}.json"
                if not instruction_path.exists():
                    raise FileNotFoundError(f"Missing instruction file: {instruction_path}")

                _link_or_copy(h5_path, raw_root / "data" / f"episode{next_episode}.hdf5")
                _link_or_copy(instruction_path, raw_root / "instructions" / f"episode{next_episode}.json")
                manifest.append(
                    {
                        "episode_index": next_episode,
                        "task": task,
                        "split": split,
                        "source_episode_index": old_episode,
                        "source_hdf5": str(h5_path),
                        "source_instruction": str(instruction_path),
                    }
                )
                next_episode += 1

    with (raw_root / "source_manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    print(f"[OK] raw combined episodes={next_episode} root={raw_root}", flush=True)
    return raw_root


def _convert_raw_to_v21(raw_root: Path, v21_root: Path, converter_path: Path, workers: int, *, overwrite: bool) -> None:
    if v21_root.exists():
        if not overwrite:
            print(f"[skip] existing LeRobot v2.1 dataset: {v21_root}", flush=True)
            return
        shutil.rmtree(v21_root)

    _run(
        [
            "python",
            str(converter_path),
            "--src",
            str(raw_root),
            "--dst",
            str(v21_root),
            "--robot-type",
            "aloha-agilex",
            "--workers",
            str(workers),
        ]
    )


def _convert_v21_to_v30(v21_root: Path, output_root: Path, *, overwrite: bool) -> Path:
    v30_root = output_root / V30_NAME
    if v30_root.exists():
        if not overwrite:
            print(f"[skip] existing LeRobot v3.0 dataset: {v30_root}", flush=True)
            return v30_root
        shutil.rmtree(v30_root)

    temp_repo_id = V21_NAME
    temp_root_parent = output_root / ".lerobot_v30_work"
    temp_root = temp_root_parent / temp_repo_id
    if temp_root_parent.exists():
        shutil.rmtree(temp_root_parent)
    temp_root_parent.mkdir(parents=True)
    shutil.copytree(v21_root, temp_root, copy_function=os.link)

    v30_converter.convert_dataset(
        repo_id=temp_repo_id,
        root=temp_root_parent,
        push_to_hub=False,
        force_conversion=True,
    )
    shutil.move(str(temp_root), str(v30_root))
    shutil.rmtree(temp_root_parent)
    print(f"[OK] LeRobot v3.0 root={v30_root}", flush=True)
    return v30_root


def parse_args() -> argparse.Namespace:
    data_base = Path(os.environ.get("ROBOTWIN_DATA_ROOT", Path.cwd() / "data"))
    robotwin_dir = Path(os.environ.get("ROBOTWIN_DIR", Path.cwd() / "external" / "RoboTwin"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=data_base / "robotwin2.0")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=data_base / "robotwin2.0" / "5_tasks_combined",
    )
    parser.add_argument(
        "--robotwin-converter",
        type=Path,
        default=robotwin_dir / "script" / "convert_robotwin_to_lerobot.py",
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--force-unzip", action="store_true")
    parser.add_argument("--overwrite-raw", action="store_true")
    parser.add_argument("--overwrite-v21", action="store_true")
    parser.add_argument("--overwrite-v30", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    _unzip_archives(args.data_root, force=args.force_unzip)
    raw_root = _build_raw_combined(args.data_root, args.output_root, overwrite=args.overwrite_raw)
    v21_root = args.output_root / V21_NAME
    _convert_raw_to_v21(raw_root, v21_root, args.robotwin_converter, args.workers, overwrite=args.overwrite_v21)
    _convert_v21_to_v30(v21_root, args.output_root, overwrite=args.overwrite_v30)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
