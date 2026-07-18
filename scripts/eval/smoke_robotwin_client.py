# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Standalone smoke test for action_policy_server_robotwin (no RoboTwin sim).

Sends one synthetic observation (3 RGB frames + a 14-D state + a prompt) to the
running server and prints the returned action chunk shape — proves the model path
end to end. Pure socket+numpy, runs in either env.

  python scripts/eval/smoke_robotwin_client.py --host 127.0.0.1 --port 9876
"""

import argparse
import base64
import json
import socket
from typing import Any

import numpy as np


def _encode(o: Any) -> Any:
    if isinstance(o, np.ndarray):
        return {"__ndarray__": base64.b64encode(np.ascontiguousarray(o).tobytes()).decode("ascii"),
                "shape": list(o.shape), "dtype": str(o.dtype)}
    if isinstance(o, dict):
        return {k: _encode(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_encode(v) for v in o]
    return o


def _decode(o: Any) -> Any:
    if isinstance(o, dict):
        if "__ndarray__" in o:
            return np.frombuffer(base64.b64decode(o["__ndarray__"]), dtype=np.dtype(o["dtype"])).reshape(o["shape"]).copy()
        return {k: _decode(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_decode(v) for v in o]
    return o


def _recv_n(s: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        c = s.recv(n - len(buf))
        if not c:
            raise ConnectionError("server closed")
        buf += c
    return buf


def _call(s: socket.socket, req: dict) -> dict:
    p = json.dumps(_encode(req)).encode()
    s.sendall(len(p).to_bytes(4, "big") + p)
    n = int.from_bytes(_recv_n(s, 4), "big")
    return _decode(json.loads(_recv_n(s, n).decode()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9876)
    ap.add_argument("--height", type=int, default=240)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument(
        "--cond",
        default="text_cond",
        choices=["text_cond", "goal_frame_cond", "text_goal_frame_cond"],
        help="Match the server's cond; goal modes attach synthetic goal_head/left/right frames.",
    )
    args = ap.parse_args()

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((args.host, args.port))
    assert _call(s, {"cmd": "ping"}).get("ok"), "ping failed"
    print("ping OK")

    h, w = args.height, args.width
    obs = {
        "cmd": "infer",
        "prompt": "Grab the toy car and place it to the left of the phone.",
        "head": (np.random.rand(h, w, 3) * 255).astype(np.uint8),
        "left": (np.random.rand(h, w, 3) * 255).astype(np.uint8),
        "right": (np.random.rand(h, w, 3) * 255).astype(np.uint8),
        "state": np.zeros(14, dtype=np.float32),
    }
    if args.cond in ("goal_frame_cond", "text_goal_frame_cond"):
        # Oracle goal frames = expert terminal obs in real eval; synthetic here.
        obs["goal_head"] = (np.random.rand(h, w, 3) * 255).astype(np.uint8)
        obs["goal_left"] = (np.random.rand(h, w, 3) * 255).astype(np.uint8)
        obs["goal_right"] = (np.random.rand(h, w, 3) * 255).astype(np.uint8)
    resp = _call(s, obs)
    if "error" in resp:
        raise SystemExit(f"server error: {resp['error']}")
    action = np.asarray(resp["action"], dtype=np.float32)
    print(f"action shape={action.shape} dtype={action.dtype} finite={np.isfinite(action).all()}")
    print("first step:", np.round(action[0], 3))
    assert action.shape == (32, 14), action.shape
    assert np.isfinite(action).all()
    print("SMOKE OK")


if __name__ == "__main__":
    main()
