# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any

import torch
import torch.distributed as dist
import wandb

from cosmos_framework.model._base import ImaginaireModel
from cosmos_framework.utils import distributed, log
from cosmos_framework.utils.callback import Callback


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        return float(value.detach().cpu().float().sum().item())
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        return int(value.detach().cpu().reshape(-1)[0].item())
    if isinstance(value, int):
        return int(value)
    if isinstance(value, list):
        total = 0
        found = False
        for item in value:
            item_value = _as_int(item)
            if item_value is not None:
                total += item_value
                found = True
        return total if found else None
    return None


def _flatten_step_times(value: Any) -> dict[str, float]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return {str(k): float(v) for k, v in value.items() if isinstance(v, (int, float))}
    if isinstance(value, list):
        out: dict[str, float] = defaultdict(float)
        for item in value:
            for key, val in _flatten_step_times(item).items():
                out[key] += val
        return dict(out)
    return {}


class DataloaderSpeedCallback(Callback):
    """Log dataloader wait time and worker-side sample timing.

    ``timer/dataloader_train`` already measures the main-process blocking time
    around ``next(dataloader)``. This callback also consumes worker timing keys
    produced by action/VLM datasets so data stalls can be split into decode,
    augmentation, transform, and collate/packing overhead.
    """

    def __init__(
        self,
        every_n: int = 100,
        step_size: int = 1,
        log_to_wandb: bool = True,
        save_s3: bool = False,
    ) -> None:
        super().__init__()
        self.every_n = int(every_n)
        self.step_size = int(step_size)
        self.log_to_wandb = bool(log_to_wandb)
        self.save_s3 = bool(save_s3)
        self._load_start: float | None = None
        self._reset()

    def _reset(self) -> None:
        self._num_batches = 0
        self._num_samples = 0
        self._wait_time = 0.0
        self._worker_batch_time = 0.0
        self._worker_aug_time = 0.0
        self._worker_io_time = 0.0
        self._from_workers = 0
        self._from_buffer = 0
        self._buffer_size = 0
        self._step_times: dict[str, float] = defaultdict(float)

    def on_before_dataloading(self, iteration: int = 0) -> None:
        self._load_start = time.monotonic()

    def on_after_dataloading(self, iteration: int = 0) -> None:
        if self._load_start is not None:
            self._wait_time += time.monotonic() - self._load_start
            self._load_start = None

    def on_training_step_start(self, model: ImaginaireModel, data: dict[str, torch.Tensor], iteration: int = 0) -> None:
        self._num_batches += 1
        self._num_samples += int(_as_int(data.get("_num_samples")) or 0)
        self._worker_batch_time += float(_as_float(data.get("_worker_batch_time")) or 0.0)
        self._worker_aug_time += float(_as_float(data.get("_worker_aug_time")) or 0.0)
        self._worker_io_time += float(_as_float(data.get("_worker_io_time")) or 0.0)
        self._from_workers += int(_as_int(data.get("_from_workers")) or 0)
        self._from_buffer += int(_as_int(data.get("_from_buffer")) or 0)
        self._buffer_size += int(_as_int(data.get("_buffer_size")) or 0)
        for key, value in _flatten_step_times(data.get("_worker_aug_step_times")).items():
            self._step_times[key] += value

    def _gather(self) -> list[dict[str, Any]]:
        local = {
            "num_batches": self._num_batches,
            "num_samples": self._num_samples,
            "wait_time": self._wait_time,
            "worker_batch_time": self._worker_batch_time,
            "worker_aug_time": self._worker_aug_time,
            "worker_io_time": self._worker_io_time,
            "from_workers": self._from_workers,
            "from_buffer": self._from_buffer,
            "buffer_size": self._buffer_size,
            "step_times": dict(self._step_times),
        }
        if dist.is_available() and dist.is_initialized():
            gathered: list[dict[str, Any] | None] = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(gathered, local)
            return [item for item in gathered if item is not None]
        return [local]

    def on_training_step_end(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        if iteration <= 0 or iteration % self.every_n != 0:
            return

        gathered = self._gather()
        if not distributed.is_rank0():
            self._reset()
            return

        total_batches = sum(int(item["num_batches"]) for item in gathered)
        total_samples = sum(int(item["num_samples"]) for item in gathered)
        total_wait = sum(float(item["wait_time"]) for item in gathered)
        wait_per_batch_by_rank = [
            float(item["wait_time"]) / max(1, int(item["num_batches"]))
            for item in gathered
            if int(item["num_batches"]) > 0
        ]
        min_wait = min(wait_per_batch_by_rank) if wait_per_batch_by_rank else 0.0
        max_wait = max(wait_per_batch_by_rank) if wait_per_batch_by_rank else 0.0
        total_worker_batch = sum(float(item["worker_batch_time"]) for item in gathered)
        total_worker_aug = sum(float(item["worker_aug_time"]) for item in gathered)
        total_worker_io = sum(float(item["worker_io_time"]) for item in gathered)
        total_from_workers = sum(int(item["from_workers"]) for item in gathered)
        total_from_buffer = sum(int(item["from_buffer"]) for item in gathered)
        total_buffer_size = sum(int(item["buffer_size"]) for item in gathered)
        step_totals: dict[str, float] = defaultdict(float)
        for item in gathered:
            for key, value in item["step_times"].items():
                step_totals[str(key)] += float(value)

        denom_batches = max(1, total_batches)
        denom_samples = max(1, total_samples)
        msg = (
            f"{iteration} : dataloader_speed "
            f"wait={total_wait / denom_batches:.3f}s/batch "
            f"wait_rank_min={min_wait:.3f}s/batch "
            f"wait_rank_max={max_wait:.3f}s/batch "
            f"worker_sample={total_worker_batch / denom_samples:.4f}s/sample "
            f"worker_timed_steps={total_worker_aug / denom_samples:.4f}s/sample "
            f"worker_unattributed={total_worker_io / denom_samples:.4f}s/sample "
            f"samples={total_samples} batches={total_batches} "
            f"from_workers={total_from_workers} from_buffer={total_from_buffer} "
            f"avg_buffer={total_buffer_size / denom_batches:.1f}"
        )
        if step_totals:
            step_msg = " ".join(
                f"{key}={value / denom_samples:.4f}s/sample" for key, value in sorted(step_totals.items())
            )
            msg = f"{msg} steps[{step_msg}]"
        log.info(msg, rank0_only=True)

        if self.log_to_wandb and wandb.run:
            info = {
                "dataloader/wait_s_per_batch": total_wait / denom_batches,
                "dataloader/wait_rank_min_s_per_batch": min_wait,
                "dataloader/wait_rank_max_s_per_batch": max_wait,
                "dataloader/worker_s_per_sample": total_worker_batch / denom_samples,
                "dataloader/worker_timed_steps_s_per_sample": total_worker_aug / denom_samples,
                "dataloader/worker_unattributed_s_per_sample": total_worker_io / denom_samples,
                "dataloader/samples": total_samples,
                "dataloader/batches": total_batches,
                "dataloader/from_workers": total_from_workers,
                "dataloader/from_buffer": total_from_buffer,
                "dataloader/avg_buffer_size": total_buffer_size / denom_batches,
            }
            for key, value in step_totals.items():
                info[f"dataloader/step_{key}_s_per_sample"] = value / denom_samples
            wandb.log(info, step=iteration)

        self._reset()
