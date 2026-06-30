# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Plain-TCP inference server for the Cosmos3-Nano-Policy-RoboTwin action policy.

Runs in the cosmos `.venv` (py3.13). A RoboTwin eval client (py3.10 conda env)
connects over a tiny length-prefixed JSON+base64 protocol — so the client needs
only socket+numpy, no openpi / torch. This mirrors the in-process FastWAM
`fastwam_policy`, but the model lives in this separate process because the
RoboTwin conda env (torch 2.4.1 / sapien) cannot coexist with cosmos (torch 2.10).

The server reproduces the RoboTwin training preprocessing EXACTLY (see
`RoboTwinLeRobotDataset` + the `action_policy_robotwin_nano` experiment):
3-camera concat_view (head full-width on top; left/right wrists resized to half
and concatenated on the bottom), resized to the no-padding RoboTwin bucket
(`resolution="384x320"`, or `736x640` for 640x480 cameras), 14-D state row
prepended, `domain_name="robotwin"`, `mode="policy"`, video `[3,33,H,W]` with
frame0 = the current observation and the rest generated. No gripper flip. Action
normalization mirrors training: with `--action-normalization none` (default for
raw-qpos checkpoints) the 14-D joint output is returned unchanged; with
`--action-normalization meanstd` (or `auto`, which reads the trained setting from
the config) the model output is denormalized back to raw qpos using the bundled
RoboTwin action stats.

Protocol (length-prefixed: 4-byte big-endian length + JSON body):
  request  {"cmd": "ping"}                                  -> {"ok": true}
  request  {"cmd": "reset"}                                 -> {"ok": true}
  request  {"cmd": "infer", "prompt": str,
            "head"/"left"/"right": <ndarray HxWx3 uint8>,
            "state": <ndarray [14] float>}                  -> {"action": <ndarray [32,14]>}
ndarrays are encoded as {"__ndarray__": <base64 raw bytes>, "shape": [...], "dtype": str}.

Example:
  PYTHONPATH=. python -m cosmos_framework.scripts.action_policy_server_robotwin \
    --checkpoint-path <.../action_policy_robotwin/checkpoints/iter_000001000> \
    --experiment action_policy_robotwin_nano --port 9876
"""

from cosmos_framework.inference.common.init import init_script

init_script()

import argparse
import base64
import json
from pathlib import Path
import socket
import socketserver
import threading
import time
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

from cosmos_framework.data.vfm.action.action_normalization import denormalize_action, load_action_stats
from cosmos_framework.data.vfm.action.datasets.robotwin_lerobot_dataset import RoboTwinLeRobotDataset
from cosmos_framework.data.vfm.action.domain_utils import get_domain_id
from cosmos_framework.data.vfm.action.transforms import ActionTransformPipeline
from cosmos_framework.inference.args import OmniSetupArgs, OmniSetupOverrides
from cosmos_framework.inference.inference import OmniInference
from cosmos_framework.inference.common.init import init_output_dir
from cosmos_framework.scripts.action_policy_server_robolab import (
    _build_data_batch_from_sample,
    _ensure_rgb_uint8_image,
    _load_training_config,
    _resize_rgb_uint8,
    _resolve_checkpoint_path,
    _validate_checkpoint,
)
from cosmos_framework.scripts.action_policy_server_utils import (
    DEFAULT_FALLBACK_OUTPUT_DIR,
    disable_runtime_ema_for_frozen_config,
    get_local_ip,
    maybe_init_distributed,
)
from cosmos_framework.utils import log

_DOMAIN_NAME = "robotwin"
_ACTION_DIM = 14  # [left_arm6, left_grip, right_arm6, right_grip]
# No-padding RoboTwin concat target canvases (H, W), keyed by resolution; must
# match ``RoboTwinLeRobotDataset._CONCAT_TARGET_HW_BY_RESOLUTION``.
_CONCAT_TARGET_HW_BY_RESOLUTION = {
    "384x320": (384, 320),
    "736x640": (736, 640),
}
_CONCAT_VIEW_DESCRIPTION = (
    "The top row is from the head camera looking at the dual-arm robot and workspace. "
    "The bottom row contains two horizontally concatenated wrist-camera views, "
    "left-arm wrist on the left and right-arm wrist on the right."
)
_DEFAULT_OUTPUT_DIR = DEFAULT_FALLBACK_OUTPUT_DIR / "robotwin"


def _infer_training_config_file_from_checkpoint(checkpoint_path: str) -> str | None:
    """Return the saved training config for a local raw-DCP iter dir, if present."""
    if "://" in checkpoint_path:
        return None
    path = Path(checkpoint_path).expanduser().absolute()

    if (path / "model").is_dir() and path.name.startswith("iter_"):
        iter_dir = path
    elif path.name == "model" and path.parent.name.startswith("iter_"):
        iter_dir = path.parent
    elif path.name.startswith("iter_"):
        iter_dir = path
    else:
        return None

    run_dir = iter_dir.parent.parent
    config_file = run_dir / "config.yaml"
    return str(config_file) if config_file.is_file() else None


def _load_saved_training_config(config_file: str | None) -> Any | None:
    if config_file is None:
        return None
    try:
        return OmegaConf.load(config_file)
    except Exception as exc:
        log.warning(f"[robotwin-policy-server] could not load saved training config {config_file}: {exc}")
        return None


# --------------------------------------------------------------------------- #
# tiny ndarray-aware JSON wire codec (mirrors RoboTwin eval_policy_client.py)  #
# --------------------------------------------------------------------------- #
def _encode(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return {
            "__ndarray__": base64.b64encode(np.ascontiguousarray(obj).tobytes()).decode("ascii"),
            "shape": list(obj.shape),
            "dtype": str(obj.dtype),
        }
    if isinstance(obj, dict):
        return {k: _encode(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_encode(v) for v in obj]
    return obj


def _decode(obj: Any) -> Any:
    if isinstance(obj, dict):
        if "__ndarray__" in obj:
            buf = base64.b64decode(obj["__ndarray__"])
            return np.frombuffer(buf, dtype=np.dtype(obj["dtype"])).reshape(obj["shape"]).copy()
        return {k: _decode(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_decode(v) for v in obj]
    return obj


def _read_action_normalization_from_config(training_config: Any) -> str | None:
    """Recursively search the training config's dataloader_train for ``action_normalization``."""
    if training_config is None:
        return None

    def _search(obj: Any) -> str | None:
        items = None
        if isinstance(obj, dict):
            items = obj.items()
        elif OmegaConf.is_config(obj):
            try:
                items = OmegaConf.to_container(obj, resolve=False).items()
            except Exception:
                items = None
        if items is None:
            return None
        for key, val in items:
            if key == "action_normalization":
                return "none" if val is None else str(val)
            found = _search(val)
            if found is not None:
                return found
        return None

    dl_train = None
    if isinstance(training_config, dict):
        dl_train = training_config.get("dataloader_train")
    else:
        dl_train = getattr(training_config, "dataloader_train", None)
    if dl_train is None:
        return None
    return _search(dl_train)


def _read_bool_from_dataloader_config(training_config: Any, key_name: str) -> bool | None:
    """Recursively search ``dataloader_train`` for a boolean dataset knob."""
    if training_config is None:
        return None

    def _search(obj: Any) -> bool | None:
        items = None
        if isinstance(obj, dict):
            items = obj.items()
        elif OmegaConf.is_config(obj):
            try:
                items = OmegaConf.to_container(obj, resolve=False).items()
            except Exception:
                items = None
        if items is None:
            return None
        for key, val in items:
            if key == key_name:
                if isinstance(val, bool):
                    return val
                if isinstance(val, str):
                    lowered = val.strip().lower()
                    if lowered in {"1", "true", "yes", "y"}:
                        return True
                    if lowered in {"0", "false", "no", "n"}:
                        return False
                return bool(val)
            found = _search(val)
            if found is not None:
                return found
        return None

    dl_train = training_config.get("dataloader_train") if isinstance(training_config, dict) else getattr(
        training_config, "dataloader_train", None
    )
    if dl_train is None:
        return None
    return _search(dl_train)


class RoboTwinPolicyService:
    """Loads the trained RoboTwin checkpoint and serves action chunks."""

    def __init__(self, args: argparse.Namespace) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for OmniMoTModel inference.")
        checkpoint_path = _resolve_checkpoint_path(args.checkpoint_path, hf_revision="main")
        _validate_checkpoint(checkpoint_path, allow_dcp_checkpoint=True)
        maybe_init_distributed()

        setup_overrides: dict[str, Any] = {
            "checkpoint_path": checkpoint_path,
            "output_dir": args.output_dir or str(_DEFAULT_OUTPUT_DIR),
            "sampler": args.sampler,
        }
        saved_config_file = _infer_training_config_file_from_checkpoint(checkpoint_path)
        if saved_config_file is not None:
            setup_overrides["config_file"] = saved_config_file
            log.info(f"[robotwin-policy-server] using saved training config: {saved_config_file}")
        elif args.experiment:
            setup_overrides["experiment"] = args.experiment

        # The training TOML set the VAE path + disabled pretrained sub-weight fetches at
        # launch; those don't apply when rebuilding a config for inference. The frozen
        # ViT / diffusion / action weights come from the trained checkpoint itself.
        exp_overrides = [
            "model.config.vlm_config.pretrained_weights.enabled=False",
            "model.config.diffusion_expert_config.load_weights_from_pretrained=False",
        ]
        if args.vae_path:
            exp_overrides.append(f"model.config.tokenizer.vae_path={args.vae_path}")
        setup_overrides["experiment_overrides"] = exp_overrides
        setup_args: OmniSetupArgs = OmniSetupOverrides.model_validate(setup_overrides).build_setup()
        init_output_dir(setup_args.output_dir)
        setup_args = disable_runtime_ema_for_frozen_config(setup_args)

        log.info(f"[robotwin-policy-server] loading model: checkpoint={checkpoint_path!r} experiment={args.experiment!r}")
        pipe = OmniInference.create(setup_args)
        self.model = pipe.model
        self.model.eval()

        # Build the SAME transform training used: ActionTransformPipeline with the
        # model's VLM tokenizer config (text tokenization requires it) + eval cfg_dropout=0.
        # Raw DCP training outputs save config.yaml without the structured ``_type`` key,
        # so prefer the OmegaConf-loaded YAML when it was found above.
        training_config = _load_saved_training_config(saved_config_file)
        if training_config is None:
            training_config = _load_training_config(pipe.setup_args, checkpoint_path)
        self.resolution = str(args.resolution)
        if self.resolution not in _CONCAT_TARGET_HW_BY_RESOLUTION:
            raise ValueError(
                f"Unsupported --resolution={self.resolution!r}; "
                f"expected one of {sorted(_CONCAT_TARGET_HW_BY_RESOLUTION)}"
            )
        self._target_hw = _CONCAT_TARGET_HW_BY_RESOLUTION[self.resolution]
        self.conditioning_fps = float(args.conditioning_fps)
        self.action_chunk_size = int(args.action_chunk_size)
        self.video_downsample_factor = self._resolve_video_downsample_factor(
            args.downsample_video_frames,
            int(args.video_downsample_factor),
            training_config,
        )
        if self.action_chunk_size % self.video_downsample_factor != 0:
            raise ValueError(
                f"--action-chunk-size={self.action_chunk_size} must be divisible by "
                f"video_downsample_factor={self.video_downsample_factor}"
            )
        self.video_fps = self.conditioning_fps / float(self.video_downsample_factor)
        tok_cfg = self._extract_tokenizer_config(training_config)
        self._transform = ActionTransformPipeline(
            tokenizer_config=tok_cfg,
            cfg_dropout_rate=0.0,
            action_video_downsample_factor=self.video_downsample_factor,
            max_action_dim=64,
            append_viewpoint_info=True,
            append_duration_fps_timestamps=True,
            append_resolution_info=True,
            append_idle_frames=False,
        )
        self.guidance = float(args.guidance)
        self.num_steps = int(args.num_steps)
        self.shift = float(args.shift)
        self.seed = int(args.seed)
        self._domain_id = get_domain_id(_DOMAIN_NAME)
        self._lock = threading.Lock()

        # Action denormalization. The training dataset normalizes action TARGETS in
        # RoboTwinLeRobotDataset._build_result, so the model emits NORMALIZED actions;
        # we must invert that here to return raw qpos. "auto" reads the method from the
        # training config's dataloader (so a checkpoint trained with raw qpos -> "none",
        # and one trained with meanstd -> "meanstd").
        self.action_normalization = self._resolve_action_normalization(
            args.action_normalization, training_config
        )
        self._norm_stats: dict[str, torch.Tensor] | None = None
        if self.action_normalization != "none":
            self._norm_stats = self._load_action_stats(args.action_stats_path)

        log.info(
            f"[robotwin-policy-server] ready domain={_DOMAIN_NAME} id={self._domain_id} res={self.resolution} "
            f"chunk={self.action_chunk_size} video_downsample_factor={self.video_downsample_factor} "
            f"video_fps={self.video_fps} action_fps={self.conditioning_fps} guidance={self.guidance} "
            f"num_steps={self.num_steps} shift={self.shift} action_normalization={self.action_normalization}"
        )

    @staticmethod
    def _resolve_action_normalization(requested: str, training_config: Any) -> str:
        """Resolve the action normalization method, reading 'auto' from the config."""
        requested = str(requested).lower()
        if requested != "auto":
            return requested
        configured = _read_action_normalization_from_config(training_config)
        if configured is None:
            log.warning(
                "[robotwin-policy-server] could not read action_normalization from the training config; "
                "assuming raw qpos (none). Pass --action-normalization explicitly to override."
            )
            return "none"
        return str(configured).lower()

    @staticmethod
    def _resolve_video_downsample_factor(
        requested: str,
        factor: int,
        training_config: Any,
    ) -> int:
        requested = str(requested).lower()
        if factor < 1:
            raise ValueError(f"--video-downsample-factor must be >= 1, got {factor}")
        if requested in {"1", "true", "yes", "y"}:
            return factor
        if requested in {"0", "false", "no", "n"}:
            return 1
        if requested != "auto":
            raise ValueError(f"Unsupported --downsample-video-frames={requested!r}")
        configured = _read_bool_from_dataloader_config(training_config, "downsample_video_frames")
        if configured is None:
            log.warning(
                "[robotwin-policy-server] could not read downsample_video_frames from the training config; "
                "assuming full 33-frame video path. Pass --downsample-video-frames explicitly to override."
            )
            return 1
        return factor if configured else 1

    def _load_action_stats(self, stats_path: str | None) -> dict[str, torch.Tensor]:
        if stats_path:
            raw = load_action_stats(stats_path)
            if raw:
                stats = {k: torch.from_numpy(v).float() for k, v in raw.items()}
            else:
                with open(stats_path, "r") as f:
                    stats_json = json.load(f)
                action_default = stats_json.get("action", {}).get("default", {})
                key_map = {
                    "mean": "global_mean",
                    "std": "global_std",
                    "min": "global_min",
                    "max": "global_max",
                    "q01": "global_q01",
                    "q99": "global_q99",
                }
                stats = {
                    key: torch.tensor(action_default[source_key], dtype=torch.float32)
                    for key, source_key in key_map.items()
                    if source_key in action_default
                }
                if not stats:
                    raise ValueError(f"No supported action stats found in {stats_path}")
        else:
            # Fall back to the dataset's bundled stats (same file used in training).
            stats = RoboTwinLeRobotDataset.load_action_stats()
        log.info(
            f"[robotwin-policy-server] loaded action stats for denormalization "
            f"(method={self.action_normalization}, keys={sorted(stats)}, dim={next(iter(stats.values())).shape[0]})"
        )
        return stats

    @staticmethod
    def _extract_tokenizer_config(training_config: Any) -> dict | None:
        if training_config is None:
            raise RuntimeError(
                "Could not load training config to recover the VLM tokenizer (text tokenization needs it). "
                "Pass --experiment action_policy_robotwin_nano and ensure ROBOTWIN_ROOT is set."
            )
        node = training_config.model.config.vlm_config.tokenizer
        return OmegaConf.to_container(node, resolve=True) if OmegaConf.is_config(node) else dict(node)

    def _concat_view(self, head: np.ndarray, left: np.ndarray, right: np.ndarray) -> torch.Tensor:
        """Reproduce RoboTwinLeRobotDataset._load_concat_video: head full-width on
        top; left/right wrists resized to half and concatenated on the bottom; the
        whole canvas is then resized to the no-padding target. Returns [3,H,W] uint8."""
        h_h, w_h = head.shape[:2]
        half = (h_h // 2, w_h // 2)
        left = _resize_rgb_uint8(left, half)
        right = _resize_rgb_uint8(right, half)
        bottom = np.concatenate([left, right], axis=1)  # [h//2, w, 3]
        concat = np.concatenate([head, bottom], axis=0)  # [h + h//2, w, 3]
        if concat.shape[:2] != self._target_hw:
            concat = _resize_rgb_uint8(concat, self._target_hw)
        return torch.from_numpy(concat).permute(2, 0, 1).contiguous()  # [3,H,W]

    def _build_sample(self, obs: dict[str, Any]) -> dict[str, Any]:
        prompt = obs.get("prompt")
        if not isinstance(prompt, str):
            raise ValueError("'prompt' must be a string")
        head = _ensure_rgb_uint8_image(obs["head"], "head")
        left = _ensure_rgb_uint8_image(obs["left"], "left")
        right = _ensure_rgb_uint8_image(obs["right"], "right")
        state = np.asarray(obs["state"], dtype=np.float32).reshape(-1)
        if state.shape[0] != _ACTION_DIM:
            raise ValueError(f"'state' must have {_ACTION_DIM} dims, got {state.shape}")

        concat = self._concat_view(head, left, right)  # [3,Hc,Wc] uint8
        _, hc, wc = concat.shape
        t_frames = self.action_chunk_size // self.video_downsample_factor + 1
        video = torch.zeros((3, t_frames, hc, wc), dtype=torch.uint8)
        video[:, 0] = concat  # current obs; rest is generated

        action = torch.zeros((self.action_chunk_size + 1, _ACTION_DIM), dtype=torch.float32)
        state_tensor = torch.from_numpy(state)
        if self.action_normalization != "none":
            assert self._norm_stats is not None
            state_tensor = self._normalize_action(state_tensor)
        action[0] = state_tensor  # use_state: row0 = initial 14-D state, in training action space

        sample = {
            "ai_caption": prompt,
            "video": video,
            "action": action,
            "conditioning_fps": torch.tensor(self.video_fps, dtype=torch.float32),
            "action_fps": torch.tensor(self.conditioning_fps, dtype=torch.float32),
            "mode": "policy",
            "domain_id": torch.tensor(self._domain_id, dtype=torch.long),
            "viewpoint": "concat_view",
            "additional_view_description": _CONCAT_VIEW_DESCRIPTION,
        }
        return self._transform(sample, self.resolution)

    def _normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        assert self._norm_stats is not None
        stats = {key: value.to(device=action.device, dtype=action.dtype) for key, value in self._norm_stats.items()}
        if self.action_normalization == "meanstd":
            return ((action - stats["mean"]) / stats["std"].clamp(min=1e-8)).clamp(-5.0, 5.0)
        if self.action_normalization == "minmax":
            lo, hi = stats["min"], stats["max"]
            return 2.0 * (action - lo) / (hi - lo).clamp(min=1e-8) - 1.0
        if self.action_normalization == "quantile":
            q01, q99 = stats["q01"], stats["q99"]
            return 2.0 * (action - q01) / (q99 - q01).clamp(min=1e-8) - 1.0
        raise ValueError(f"Unsupported action normalization for conditioning state: {self.action_normalization!r}")

    def infer(self, obs: dict[str, Any]) -> np.ndarray:
        sample = self._build_sample(obs)
        data_batch = _build_data_batch_from_sample(sample)
        with self._lock:
            with torch.inference_mode():
                samples = self.model.generate_samples_from_batch(
                    data_batch,
                    guidance=self.guidance,
                    seed=[self.seed],
                    num_steps=self.num_steps,
                    shift=self.shift,
                )
        action = samples["action"][0][:, :_ACTION_DIM]  # [33,14]
        action = action[1:]  # drop the state row -> [32,14]
        action = action.detach().cpu().float()
        if self.action_normalization != "none":
            assert self._norm_stats is not None
            action = denormalize_action(action, self.action_normalization, self._norm_stats)  # [32,14] raw qpos
        return action.numpy()  # NO gripper flip


def _make_handler(service: RoboTwinPolicyService) -> type[socketserver.BaseRequestHandler]:
    def _recv_n(sock, n: int) -> bytes:
        buf = b""
        while len(buf) < n:
            chunk = sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("socket closed")
            buf += chunk
        return buf

    class Handler(socketserver.BaseRequestHandler):
        def handle(self) -> None:
            sock = self.request
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            while True:
                try:
                    header = _recv_n(sock, 4)
                except (ConnectionError, OSError):
                    return
                length = int.from_bytes(header, "big")
                body = _recv_n(sock, length)
                t_body_received = time.perf_counter()
                req = _decode(json.loads(body.decode("utf-8")))
                t_req_decoded = time.perf_counter()
                cmd = req.get("cmd", "infer")
                timing: dict[str, float] | None = None
                try:
                    if cmd == "ping":
                        resp: dict[str, Any] = {"ok": True}
                    elif cmd == "reset":
                        resp = {"ok": True}
                    elif cmd == "infer":
                        t_infer_start = time.perf_counter()
                        action = service.infer(req)
                        t_action_ready = time.perf_counter()
                        timing = {
                            "server_request_decode_s": t_req_decoded - t_body_received,
                            "server_obs_to_action_s": t_action_ready - t_infer_start,
                            "server_request_to_action_s": t_action_ready - t_body_received,
                        }
                        resp = {"action": action, "_server_timing": timing}
                    else:
                        resp = {"error": f"unknown cmd {cmd!r}"}
                except Exception as exc:  # noqa: BLE001
                    log.error(f"[robotwin-policy-server] infer error: {exc}")
                    resp = {"error": str(exc)}
                payload = json.dumps(_encode(resp)).encode("utf-8")
                sock.sendall(len(payload).to_bytes(4, "big") + payload)

    return Handler


class _ThreadingTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def serve(args: argparse.Namespace) -> None:
    service = RoboTwinPolicyService(args)
    server = _ThreadingTCPServer((args.host, int(args.port)), _make_handler(service))
    local_ip = get_local_ip()
    log.info(f"[robotwin-policy-server] serving on tcp://{local_ip}:{int(args.port)} (bind {args.host})")
    server.serve_forever()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint-path", required=True, help="Trained RoboTwin DCP dir (…/checkpoints/iter_<N>) or safetensors dir / HF repo")
    p.add_argument("--experiment", default="action_policy_robotwin_nano", help="Hydra experiment for DCP config rebuild")
    p.add_argument("--vae-path", default="/pfs/pfs-7jnepv/shukaigong/weights/Wan2.2-TI2V-5B/Wan2.2_VAE.pth",
                   help="Local Wan2.2 VAE .pth (overrides the experiment's default registry path; required for --experiment loading)")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=9876)
    p.add_argument("--resolution", default="384x320")
    p.add_argument("--conditioning-fps", type=float, default=30.0)
    p.add_argument("--action-chunk-size", type=int, default=32)
    p.add_argument(
        "--downsample-video-frames",
        default="auto",
        choices=["auto", "true", "false", "1", "0", "yes", "no", "y", "n"],
        help=(
            "Whether to use the FastWAM-style 4x video-only downsampling at eval. "
            "'auto' reads downsample_video_frames from the training config."
        ),
    )
    p.add_argument(
        "--video-downsample-factor",
        type=int,
        default=4,
        help="Video-only temporal downsample factor used when --downsample-video-frames is true.",
    )
    p.add_argument("--guidance", type=float, default=3.0)
    p.add_argument("--num-steps", type=int, default=4)
    p.add_argument("--shift", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sampler", default="unipc")
    p.add_argument("--output-dir", default=None)
    p.add_argument(
        "--action-normalization",
        default="auto",
        choices=["auto", "none", "meanstd", "minmax", "quantile"],
        help=(
            "How to denormalize predicted actions back to raw qpos. Must match the value the "
            "checkpoint was TRAINED with. 'auto' reads it from the training config "
            "(none/meanstd); 'none' returns the raw model output unchanged."
        ),
    )
    p.add_argument(
        "--action-stats-path",
        default=None,
        help=(
            "Path to the action stats JSON used for denormalization. Defaults to the dataset's "
            "bundled stats/robotwin_lerobot_stats.json. Only used when normalization != none."
        ),
    )
    args = p.parse_args()
    serve(args)


if __name__ == "__main__":
    main()
