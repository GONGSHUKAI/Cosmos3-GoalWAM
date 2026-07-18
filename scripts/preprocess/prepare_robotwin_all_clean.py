#!/usr/bin/env python3
"""Prepare all RoboTwin aloha-agilex clean episodes as one LeRobot v3 dataset.

The output directory itself is a Cosmos-trainable LeRobot v3 root. Intermediate
raw and LeRobot v2.1 datasets are kept in a separate work directory so the
pipeline can resume after interruption.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd
from lerobot.datasets.v30 import convert_dataset_v21_to_v30 as v30_converter

CLEAN_SPLIT = "aloha-agilex_clean_50"
EXPECTED_TASKS = 50
EXPECTED_EPISODES_PER_TASK = 50
EXPECTED_TOTAL_EPISODES = EXPECTED_TASKS * EXPECTED_EPISODES_PER_TASK
RAW_NAME = "aloha-agilex_all_clean_2500"
V21_NAME = f"{RAW_NAME}_lerobot_v2.1"


def _episode_id(path: Path) -> int:
    return int(path.stem.removeprefix("episode"))


def _run(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def _discover_tasks(data_root: Path) -> list[str]:
    tasks = sorted(path.parent.name for path in data_root.glob(f"*/{CLEAN_SPLIT}.zip"))
    if len(tasks) != EXPECTED_TASKS:
        raise ValueError(f"Expected {EXPECTED_TASKS} clean task archives under {data_root}, found {len(tasks)}")
    return tasks


def _unzip_archives(data_root: Path, tasks: list[str], *, force: bool) -> None:
    for index, task in enumerate(tasks, start=1):
        task_root = data_root / task
        archive = task_root / f"{CLEAN_SPLIT}.zip"
        split_root = task_root / CLEAN_SPLIT
        if split_root.exists() and not force:
            print(f"[{index}/{len(tasks)}] skip existing {split_root}", flush=True)
            continue
        if split_root.exists():
            shutil.rmtree(split_root)
        _run(["unzip", "-q", str(archive), "-d", str(task_root)])


def _validate_raw_splits(data_root: Path, tasks: list[str]) -> None:
    for task in tasks:
        split_root = data_root / task / CLEAN_SPLIT
        data_files = sorted((split_root / "data").glob("episode*.hdf5"), key=_episode_id)
        instruction_files = sorted((split_root / "instructions").glob("episode*.json"), key=_episode_id)
        if len(data_files) != EXPECTED_EPISODES_PER_TASK or len(instruction_files) != EXPECTED_EPISODES_PER_TASK:
            raise ValueError(
                f"{task}: expected {EXPECTED_EPISODES_PER_TASK} data and instruction files, "
                f"found {len(data_files)} and {len(instruction_files)}"
            )
        data_ids = [_episode_id(path) for path in data_files]
        instruction_ids = [_episode_id(path) for path in instruction_files]
        if data_ids != instruction_ids:
            raise ValueError(f"{task}: episode IDs differ between data and instructions")


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _build_raw_combined(data_root: Path, work_root: Path, tasks: list[str], *, overwrite: bool) -> Path:
    raw_root = work_root / RAW_NAME
    if raw_root.exists() and not overwrite:
        manifest_path = raw_root / "source_manifest.json"
        if manifest_path.exists() and len(json.loads(manifest_path.read_text())) == EXPECTED_TOTAL_EPISODES:
            print(f"[skip] existing raw combined dataset: {raw_root}", flush=True)
            return raw_root
        raise ValueError(f"Incomplete existing raw dataset; rerun with --overwrite-raw: {raw_root}")
    if raw_root.exists():
        shutil.rmtree(raw_root)

    (raw_root / "data").mkdir(parents=True)
    (raw_root / "instructions").mkdir()
    manifest: list[dict[str, Any]] = []
    next_episode = 0
    for task_index, task in enumerate(tasks):
        split_root = data_root / task / CLEAN_SPLIT
        h5_files = sorted((split_root / "data").glob("episode*.hdf5"), key=_episode_id)
        for h5_path in h5_files:
            source_episode = _episode_id(h5_path)
            instruction_path = split_root / "instructions" / f"episode{source_episode}.json"
            _link_or_copy(h5_path, raw_root / "data" / f"episode{next_episode}.hdf5")
            _link_or_copy(instruction_path, raw_root / "instructions" / f"episode{next_episode}.json")
            manifest.append(
                {
                    "episode_index": next_episode,
                    "benchmark_task_index": task_index,
                    "benchmark_task": task,
                    "split": CLEAN_SPLIT,
                    "source_episode_index": source_episode,
                    "source_hdf5": str(h5_path),
                    "source_instruction": str(instruction_path),
                }
            )
            next_episode += 1

    if next_episode != EXPECTED_TOTAL_EPISODES:
        raise ValueError(f"Expected {EXPECTED_TOTAL_EPISODES} combined episodes, built {next_episode}")
    (raw_root / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[OK] raw combined episodes={next_episode} root={raw_root}", flush=True)
    return raw_root


def _convert_raw_to_v21(raw_root: Path, v21_root: Path, converter_path: Path, workers: int, *, overwrite: bool) -> None:
    if v21_root.exists() and not overwrite:
        info = json.loads((v21_root / "meta" / "info.json").read_text())
        if info.get("total_episodes") == EXPECTED_TOTAL_EPISODES:
            print(f"[skip] existing LeRobot v2.1 dataset: {v21_root}", flush=True)
            return
        raise ValueError(f"Incomplete existing v2.1 dataset; rerun with --overwrite-v21: {v21_root}")
    if v21_root.exists():
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


def _convert_v21_to_v30(v21_root: Path, output_root: Path, work_root: Path, *, overwrite: bool) -> None:
    if output_root.exists() and not overwrite:
        info = json.loads((output_root / "meta" / "info.json").read_text())
        if info.get("codebase_version") == "v3.0" and info.get("total_episodes") == EXPECTED_TOTAL_EPISODES:
            print(f"[skip] existing LeRobot v3.0 dataset: {output_root}", flush=True)
            return
        raise ValueError(f"Incomplete existing v3.0 dataset; rerun with --overwrite-v30: {output_root}")
    if output_root.exists():
        shutil.rmtree(output_root)

    conversion_parent = work_root / ".lerobot_v30_conversion"
    conversion_root = conversion_parent / V21_NAME
    if conversion_parent.exists():
        shutil.rmtree(conversion_parent)
    conversion_parent.mkdir(parents=True)
    shutil.copytree(v21_root, conversion_root, copy_function=os.link)
    v30_converter.convert_dataset(
        repo_id=V21_NAME,
        root=conversion_parent,
        push_to_hub=False,
        force_conversion=True,
    )
    shutil.move(str(conversion_root), str(output_root))
    shutil.rmtree(conversion_parent)
    print(f"[OK] LeRobot v3.0 root={output_root}", flush=True)


def _fix_tasks_index(output_root: Path) -> None:
    tasks_path = output_root / "meta" / "tasks.parquet"
    tasks = pd.read_parquet(tasks_path)
    tasks.index.name = "task"
    tasks.to_parquet(tasks_path)
    print(f"[OK] tasks.parquet index name={tasks.index.name!r} rows={len(tasks)}", flush=True)


def _write_action_stats(output_root: Path) -> Path:
    stats_path = output_root / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text())["action"]
    action_stats = {key: stats[key] for key in ("mean", "std", "min", "max")}
    output_path = output_root / "action_stats.json"
    output_path.write_text(json.dumps(action_stats, indent=2) + "\n")
    print(f"[OK] action normalization stats={output_path}", flush=True)
    return output_path


def parse_args() -> argparse.Namespace:
    data_base = Path(os.environ.get("ROBOTWIN_DATA_ROOT", Path.cwd() / "data"))
    robotwin_dir = Path(os.environ.get("ROBOTWIN_DIR", Path.cwd() / "external" / "RoboTwin"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=data_base / "robotwin2.0")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=data_base / "robotwin2.0_clean2rand_lerobot_v3.0",
    )
    parser.add_argument(
        "--work-root",
        type=Path,
        default=data_base / ".robotwin2.0_clean2rand_work",
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
    tasks = _discover_tasks(args.data_root)
    args.work_root.mkdir(parents=True, exist_ok=True)
    _unzip_archives(args.data_root, tasks, force=args.force_unzip)
    _validate_raw_splits(args.data_root, tasks)
    raw_root = _build_raw_combined(args.data_root, args.work_root, tasks, overwrite=args.overwrite_raw)
    v21_root = args.work_root / V21_NAME
    _convert_raw_to_v21(raw_root, v21_root, args.robotwin_converter, args.workers, overwrite=args.overwrite_v21)
    _convert_v21_to_v30(v21_root, args.output_root, args.work_root, overwrite=args.overwrite_v30)
    _fix_tasks_index(args.output_root)
    _write_action_stats(args.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
