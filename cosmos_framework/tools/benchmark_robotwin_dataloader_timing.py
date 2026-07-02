# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Benchmark RoboTwin online LeRobot dataloader preprocessing paths.

This intentionally avoids model/checkpoint/optimizer setup. It instantiates the
same configured ``dataloader_train`` as the SFT recipe, keeps the recipe's
``num_workers`` and ``prefetch_factor``, and reports the timing metadata emitted
by the dataset and packing dataloader.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

import torch
import torch.distributed as dist

from cosmos_framework.configs.toml_config.sft_config import load_experiment_from_toml
from cosmos_framework.utils.lazy_config import instantiate

CASES = {
    "robotwin_full_aug_on_b32": (False, True, "[33]"),
    "robotwin_full_aug_off_b32": (False, False, "[33]"),
    "robotwin_downsample_aug_on_b32": (True, True, "[9]"),
    "robotwin_downsample_aug_off_b32": (True, False, "[9]"),
}
_DIST_INIT_TEMP_DIR: tempfile.TemporaryDirectory | None = None


def _bind_local_cuda_device() -> None:
    if not torch.cuda.is_available():
        return
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)


def _as_float(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return 0.0
        return float(value.detach().cpu().float().sum().item())
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def _as_int(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return 0
        return int(value.detach().cpu().reshape(-1)[0].item())
    if isinstance(value, int):
        return int(value)
    if isinstance(value, list):
        return sum(_as_int(item) for item in value)
    return 0


def _infer_num_samples(batch: dict[str, Any]) -> int:
    explicit = _as_int(batch.get("_num_samples"))
    if explicit:
        return explicit
    for key in ("video", "images", "action", "sequence_plan", "dataset_name"):
        value = batch.get(key)
        if isinstance(value, list):
            return len(value)
        if isinstance(value, torch.Tensor) and value.dim() > 0:
            return int(value.shape[0])
    return 0


def _flatten_step_times(value: Any) -> dict[str, float]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return {str(key): _as_float(val) for key, val in value.items()}
    if isinstance(value, list):
        totals: dict[str, float] = defaultdict(float)
        for item in value:
            for key, val in _flatten_step_times(item).items():
                totals[key] += val
        return dict(totals)
    return {}


def _case_names(value: str) -> list[str]:
    if value == "all":
        return list(CASES)
    names = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [name for name in names if name not in CASES]
    if unknown:
        raise ValueError(f"Unknown case(s): {unknown}. Valid cases: {sorted(CASES)}")
    return names


def _init_distributed_if_needed() -> None:
    if not dist.is_available() or dist.is_initialized():
        return
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="gloo", init_method="env://")
        return

    global _DIST_INIT_TEMP_DIR
    _DIST_INIT_TEMP_DIR = tempfile.TemporaryDirectory(prefix="robotwin_dataloader_bench_dist_")
    store_path = os.path.join(_DIST_INIT_TEMP_DIR.name, "store")
    dist.init_process_group(backend="gloo", init_method=f"file://{store_path}", rank=0, world_size=1)


def _rank() -> int:
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def _world_size() -> int:
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


def _gather(summary: dict[str, Any]) -> list[dict[str, Any]]:
    if dist.is_available() and dist.is_initialized():
        gathered: list[dict[str, Any] | None] = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, summary)
        return [item for item in gathered if item is not None]
    return [summary]


def _format_steps(step_totals: dict[str, float], samples: int) -> str:
    if not step_totals:
        return "none"
    denom = max(1, samples)
    return " ".join(f"{key}={value / denom:.4f}s/sample" for key, value in sorted(step_totals.items()))


def _build_overrides(
    name: str, downsample: bool, augment: bool, encode_durations: str, args: argparse.Namespace
) -> list[str]:
    return [
        f"job.name={name}",
        "job.wandb_mode=disabled",
        "dataloader_train.max_samples_per_batch=32",
        "dataloader_train.dataloader.datasets.robotwin.dataset.emit_timing=true",
        f"dataloader_train.dataloader.datasets.robotwin.dataset.use_image_augmentation={str(augment).lower()}",
        f"dataloader_train.dataloader.datasets.robotwin.dataset.downsample_video_frames={str(downsample).lower()}",
        f"model.config.tokenizer.encode_exact_durations={encode_durations}",
        *args.extra_override,
    ]


def run_case(name: str, args: argparse.Namespace) -> dict[str, Any]:
    downsample, augment, encode_durations = CASES[name]
    if _rank() == 0:
        print(
            f"[benchmark] start case={name} downsample={downsample} augmentation={augment} "
            f"max_samples_per_batch=32 batches={args.batches} warmup={args.warmup_batches}",
            file=sys.stderr,
            flush=True,
        )
    overrides = _build_overrides(name, downsample, augment, encode_durations, args)
    config = load_experiment_from_toml(args.sft_toml, extra_overrides=overrides)
    dataloader = instantiate(config.dataloader_train)
    iterator = iter(dataloader)

    wait_times: list[float] = []
    worker_batch_time = 0.0
    worker_timed_steps = 0.0
    worker_unattributed = 0.0
    from_workers = 0
    from_buffer = 0
    buffer_size = 0
    num_batches = 0
    num_samples = 0
    step_totals: dict[str, float] = defaultdict(float)

    for i in range(args.warmup_batches + args.batches):
        t0 = time.monotonic()
        batch = next(iterator)
        elapsed = time.monotonic() - t0
        if i < args.warmup_batches:
            continue
        measured_index = i - args.warmup_batches + 1
        wait_times.append(elapsed)
        num_batches += 1
        num_samples += _infer_num_samples(batch)
        worker_batch_time += _as_float(batch.get("_worker_batch_time"))
        worker_timed_steps += _as_float(batch.get("_worker_aug_time"))
        worker_unattributed += _as_float(batch.get("_worker_io_time"))
        from_workers += _as_int(batch.get("_from_workers"))
        from_buffer += _as_int(batch.get("_from_buffer"))
        buffer_size += _as_int(batch.get("_buffer_size"))
        for key, value in _flatten_step_times(batch.get("_worker_aug_step_times")).items():
            step_totals[key] += value
        if _rank() == 0 and args.progress_every > 0 and measured_index % args.progress_every == 0:
            print(
                f"[benchmark] case={name} rank0_batch={measured_index}/{args.batches} "
                f"wait={elapsed:.3f}s samples={_infer_num_samples(batch)}",
                file=sys.stderr,
                flush=True,
            )

    summary = {
        "case": name,
        "rank": _rank(),
        "world_size": _world_size(),
        "batches": num_batches,
        "samples": num_samples,
        "wait_time": sum(wait_times),
        "wait_min": min(wait_times) if wait_times else 0.0,
        "wait_max": max(wait_times) if wait_times else 0.0,
        "worker_batch_time": worker_batch_time,
        "worker_timed_steps": worker_timed_steps,
        "worker_unattributed": worker_unattributed,
        "from_workers": from_workers,
        "from_buffer": from_buffer,
        "buffer_size": buffer_size,
        "step_totals": dict(step_totals),
    }
    return summary


def summarize(case: str, summaries: Iterable[dict[str, Any]]) -> str:
    items = list(summaries)
    batches = sum(int(item["batches"]) for item in items)
    samples = sum(int(item["samples"]) for item in items)
    wait_time = sum(float(item["wait_time"]) for item in items)
    wait_rank_avg = [float(item["wait_time"]) / max(1, int(item["batches"])) for item in items if item["batches"]]
    worker_batch_time = sum(float(item["worker_batch_time"]) for item in items)
    worker_timed_steps = sum(float(item["worker_timed_steps"]) for item in items)
    worker_unattributed = sum(float(item["worker_unattributed"]) for item in items)
    from_workers = sum(int(item["from_workers"]) for item in items)
    from_buffer = sum(int(item["from_buffer"]) for item in items)
    buffer_size = sum(int(item["buffer_size"]) for item in items)
    step_totals: dict[str, float] = defaultdict(float)
    for item in items:
        for key, value in item["step_totals"].items():
            step_totals[str(key)] += float(value)

    denom_batches = max(1, batches)
    denom_samples = max(1, samples)
    return (
        f"{case}: "
        f"wait={wait_time / denom_batches:.3f}s/batch "
        f"wait_rank_min={(min(wait_rank_avg) if wait_rank_avg else 0.0):.3f}s/batch "
        f"wait_rank_max={(max(wait_rank_avg) if wait_rank_avg else 0.0):.3f}s/batch "
        f"worker_sample={worker_batch_time / denom_samples:.4f}s/sample "
        f"worker_timed_steps={worker_timed_steps / denom_samples:.4f}s/sample "
        f"worker_unattributed={worker_unattributed / denom_samples:.4f}s/sample "
        f"samples={samples} batches={batches} "
        f"from_workers={from_workers} from_buffer={from_buffer} "
        f"avg_buffer={buffer_size / denom_batches:.1f} "
        f"steps[{_format_steps(step_totals, samples)}]"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft-toml", default="examples/toml/sft_config/action_policy_robotwin.toml")
    parser.add_argument("--cases", default="all", help="Comma-separated case names or 'all'.")
    parser.add_argument("--batches", type=int, default=20, help="Measured batches per case.")
    parser.add_argument("--warmup-batches", type=int, default=2, help="Warmup batches skipped per case.")
    parser.add_argument("--progress-every", type=int, default=1, help="Print rank-0 progress every N measured batches.")
    parser.add_argument(
        "extra_override",
        nargs="*",
        help="Additional Hydra overrides appended after the benchmark overrides.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _bind_local_cuda_device()
    _init_distributed_if_needed()
    try:
        for case in _case_names(args.cases):
            local = run_case(case, args)
            gathered = _gather(local)
            if _rank() == 0:
                print(summarize(case, gathered), flush=True)
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
        if _DIST_INIT_TEMP_DIR is not None:
            _DIST_INIT_TEMP_DIR.cleanup()


if __name__ == "__main__":
    main()
