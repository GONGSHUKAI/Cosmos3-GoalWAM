# RoboTwin eval client for Cosmos3-Nano-Policy-RoboTwin.
#
# Runs in the RoboTwin conda env (py3.10). The Cosmos model lives in a SEPARATE
# process (cosmos .venv, py3.13) because the Cosmos and SAPIEN/Curobo dependency
# stacks are intentionally isolated. This module is a thin socket client to
# cosmos_framework.scripts.action_policy_server_robotwin — no torch/cosmos here,
# only socket + numpy. Mirrors FastWAM's in-process fastwam_policy contract.
#
# Interface (RoboTwin script/eval_policy.py): encode_obs / get_model / eval / reset_model.

import base64
import json
import os
import socket
from collections import deque
from typing import Any

import numpy as np

_ACTION_DIM = 14  # [left_arm6, left_grip, right_arm6, right_grip] (qpos)


def _parse_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


# --------------------------------------------------------------------------- #
# wire codec (must match action_policy_server_robotwin.py _encode/_decode)     #
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


def _recv_n(sock: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("cosmos policy server closed the connection")
        buf += chunk
    return buf


class CosmosPolicyClient:
    """Socket client plus eval-time action postprocessing."""

    def __init__(
        self,
        host: str,
        port: int,
        replan_steps: int,
        gripper_hysteresis: bool = False,
        timeout: float = 600.0,
        goal_cond: bool = False,
        goal_source: str = "oracle",
    ) -> None:
        self.host = host
        self.port = int(port)
        self.replan_steps = int(max(1, min(replan_steps, 32)))
        self.gripper_hysteresis = bool(gripper_hysteresis)
        self.timeout = float(timeout)
        # Goal-image conditioning (server trained with cond=goal_frame_cond /
        # text_goal_frame_cond): forward the oracle goal frames — the terminal
        # observation of the rule-based expert, handed over per episode by
        # script/eval_policy.py via set_episode_ref_obs.
        self.goal_cond = bool(goal_cond)
        self.goal_source = str(goal_source).lower()
        if self.goal_source == "generated":
            raise NotImplementedError(
                "goal_source=generated: plug a goal-image generator here (generate the goal frame "
                "from the current observation + instruction); only goal_source=oracle is implemented."
            )
        if self.goal_source != "oracle":
            raise ValueError(f"Unsupported goal_source={self.goal_source!r}")
        self._goal_obs: dict[str, np.ndarray] | None = None
        self.pending: deque[np.ndarray] = deque()
        self.sock: socket.socket | None = None
        self._gripper_seen_closed = [False, False]
        self._gripper_force_open = [False, False]
        self._connect()

    def _connect(self) -> None:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(self.timeout)
        s.connect((self.host, self.port))
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock = s
        # sanity ping
        if not self._call({"cmd": "ping"}).get("ok"):
            raise RuntimeError("cosmos policy server ping failed")
        print(
            f"[cosmos_policy] connected to {self.host}:{self.port} "
            f"replan_steps={self.replan_steps} gripper_hysteresis={self.gripper_hysteresis}",
            flush=True,
        )

    def _call(self, req: dict[str, Any]) -> dict[str, Any]:
        assert self.sock is not None
        payload = json.dumps(_encode(req)).encode("utf-8")
        self.sock.sendall(len(payload).to_bytes(4, "big") + payload)
        length = int.from_bytes(_recv_n(self.sock, 4), "big")
        resp = _decode(json.loads(_recv_n(self.sock, length).decode("utf-8")))
        if isinstance(resp, dict) and resp.get("error"):
            raise RuntimeError(f"cosmos policy server error: {resp['error']}")
        return resp

    def set_episode_ref_obs(self, observation: dict[str, Any]) -> None:
        """Store the oracle goal (expert terminal observation) for this episode.

        Called by script/eval_policy.py after the successful expert seed-check.
        Raw uint8 frames only — the server owns layout/resizing (concat vs
        cam_high) so it always matches how the checkpoint was trained.
        """
        if not self.goal_cond:
            return
        obs = observation["observation"]
        self._goal_obs = {
            "goal_head": np.asarray(obs["head_camera"]["rgb"], dtype=np.uint8),
            "goal_left": np.asarray(obs["left_camera"]["rgb"], dtype=np.uint8),
            "goal_right": np.asarray(obs["right_camera"]["rgb"], dtype=np.uint8),
        }
        print("[cosmos_policy] oracle goal frames captured for this episode", flush=True)

    def infer(self, obs: dict[str, Any], instruction: str) -> np.ndarray:
        req = {
            "cmd": "infer",
            "prompt": str(instruction),
            "head": np.asarray(obs["head"], dtype=np.uint8),
            "left": np.asarray(obs["left"], dtype=np.uint8),
            "right": np.asarray(obs["right"], dtype=np.uint8),
            "state": np.asarray(obs["state"], dtype=np.float32).reshape(-1),
        }
        if self.goal_cond:
            if self._goal_obs is None:
                raise RuntimeError(
                    "goal_cond=true but no goal frames were captured — script/eval_policy.py must "
                    "call set_episode_ref_obs with the expert terminal observation before the rollout "
                    "(is the expert seed-check enabled?)"
                )
            req.update(self._goal_obs)
        action = np.asarray(self._call(req)["action"], dtype=np.float32)  # [32,14]
        assert action.ndim == 2 and action.shape[-1] == _ACTION_DIM, action.shape
        if action.shape[1] >= 14:
            print(
                "[cosmos_policy] chunk gripper "
                f"left=min:{action[:, 6].min():.3f} max:{action[:, 6].max():.3f} "
                f"right=min:{action[:, 13].min():.3f} max:{action[:, 13].max():.3f}",
                flush=True,
            )
        return action

    def apply_gripper_hysteresis(self, task_env: Any, action: np.ndarray) -> np.ndarray:
        if not self.gripper_hysteresis or action.shape[0] < 14:
            return action
        try:
            current_grippers = [
                float(task_env.robot.get_left_gripper_val()),
                float(task_env.robot.get_right_gripper_val()),
            ]
        except Exception:
            current_grippers = [None, None]
        for arm_i, dim in enumerate((6, 13)):
            if current_grippers[arm_i] is not None and current_grippers[arm_i] <= 0.2:
                self._gripper_seen_closed[arm_i] = True
            if self._gripper_seen_closed[arm_i] and action[dim] >= 0.8:
                self._gripper_force_open[arm_i] = True
            if self._gripper_force_open[arm_i]:
                action[dim] = 1.0
        return action

    def reset(self) -> None:
        self.pending.clear()
        self._gripper_seen_closed = [False, False]
        self._gripper_force_open = [False, False]
        self._goal_obs = None  # per-episode oracle goal; re-set via set_episode_ref_obs
        try:
            self._call({"cmd": "reset"})
        except Exception as exc:  # noqa: BLE001 — reset is best-effort
            print(f"[cosmos_policy] reset notify failed (ignored): {exc}", flush=True)


# --------------------------------------------------------------------------- #
# RoboTwin policy interface                                                    #
# --------------------------------------------------------------------------- #
def encode_obs(observation):
    """RoboTwin obs dict -> {head,left,right rgb (HxWx3 uint8), state (14,)}."""
    obs = observation["observation"]
    return {
        "head": obs["head_camera"]["rgb"],
        "left": obs["left_camera"]["rgb"],
        "right": obs["right_camera"]["rgb"],
        "state": np.asarray(observation["joint_action"]["vector"], dtype=np.float32),
    }


def get_model(usr_args):
    host = os.environ.get("COSMOS_SERVER_HOST") or usr_args.get("server_host", "127.0.0.1")
    port = _parse_int(os.environ.get("COSMOS_SERVER_PORT") or usr_args.get("server_port"), 9876)
    replan_steps = _parse_int(os.environ.get("COSMOS_POLICY_REPLAN_STEPS") or usr_args.get("replan_steps"), 8)
    gripper_hysteresis = _parse_bool(
        os.environ.get("COSMOS_GRIPPER_HYSTERESIS") or usr_args.get("gripper_hysteresis"),
        default=False,
    )
    # Goal-image conditioning: must match the server's cond (goal-trained
    # checkpoints require goal frames; text_cond servers ignore them).
    goal_cond = _parse_bool(
        os.environ.get("COSMOS_GOAL_COND") or usr_args.get("goal_cond"),
        default=False,
    )
    goal_source = os.environ.get("COSMOS_GOAL_SOURCE") or usr_args.get("goal_source", "oracle")
    return CosmosPolicyClient(
        host=host,
        port=port,
        replan_steps=replan_steps,
        gripper_hysteresis=gripper_hysteresis,
        goal_cond=goal_cond,
        goal_source=goal_source,
    )


def eval(TASK_ENV, model, observation):
    """One inference per call; execute up to replan_steps actions (receding horizon).
    The eval_policy.py outer loop fetches a fresh obs and re-calls us each round."""
    instruction = TASK_ENV.get_instruction()
    obs = encode_obs(observation)
    action_chunk = model.infer(obs, instruction)  # [32,14]
    n_exec = min(model.replan_steps, action_chunk.shape[0])
    for i in range(n_exec):
        action = model.apply_gripper_hysteresis(TASK_ENV, action_chunk[i].copy())
        TASK_ENV.take_action(action, action_type="qpos")  # [left_arm6,left_grip,right_arm6,right_grip]
        if getattr(TASK_ENV, "eval_success", False):
            return


def reset_model(model):
    model.reset()
