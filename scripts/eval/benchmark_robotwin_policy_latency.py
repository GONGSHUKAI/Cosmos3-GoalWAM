#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Benchmark RoboTwin policy-server action inference latency.

This is a pure TCP client for ``action_policy_server_robotwin.py``. It sends
synthetic RoboTwin-shaped observations and times infer calls without launching
the RoboTwin simulator, so the measured latency is policy-server latency plus
localhost protocol overhead.
"""

from __future__ import annotations

import argparse
import base64
import json
import socket
import statistics
import time
from typing import Any

import numpy as np


def _encode(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return {
            "__ndarray__": base64.b64encode(np.ascontiguousarray(obj).tobytes()).decode("ascii"),
            "shape": list(obj.shape),
            "dtype": str(obj.dtype),
        }
    if isinstance(obj, dict):
        return {key: _encode(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_encode(value) for value in obj]
    return obj


def _decode(obj: Any) -> Any:
    if isinstance(obj, dict):
        if "__ndarray__" in obj:
            return np.frombuffer(base64.b64decode(obj["__ndarray__"]), dtype=np.dtype(obj["dtype"])).reshape(
                obj["shape"]
            ).copy()
        return {key: _decode(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_decode(value) for value in obj]
    return obj


def _recv_n(sock: socket.socket, n: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < n:
        chunk = sock.recv(n - len(chunks))
        if not chunk:
            raise ConnectionError("server closed the socket")
        chunks.extend(chunk)
    return bytes(chunks)


def _call(sock: socket.socket, request: dict[str, Any]) -> tuple[dict[str, Any], dict[str, float]]:
    t0 = time.perf_counter()
    payload = json.dumps(_encode(request)).encode("utf-8")
    t1 = time.perf_counter()

    sock.sendall(len(payload).to_bytes(4, "big") + payload)
    header = _recv_n(sock, 4)
    n = int.from_bytes(header, "big")
    body = _recv_n(sock, n)
    t2 = time.perf_counter()

    response = _decode(json.loads(body.decode("utf-8")))
    t3 = time.perf_counter()
    return response, {
        "client_encode_s": t1 - t0,
        "payload_roundtrip_s": t2 - t1,
        "client_decode_s": t3 - t2,
        "client_total_s": t3 - t0,
    }


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    arr = np.asarray(values, dtype=np.float64)
    return float(np.percentile(arr, pct))


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "mean_s": float(statistics.fmean(values)),
        "median_s": float(statistics.median(values)),
        "p90_s": _percentile(values, 90),
        "p95_s": _percentile(values, 95),
        "min_s": float(min(values)),
        "max_s": float(max(values)),
        "iters": float(len(values)),
    }


def _make_observation(args: argparse.Namespace) -> dict[str, Any]:
    rng = np.random.default_rng(args.seed)
    h, w = args.height, args.width
    state = np.zeros(14, dtype=np.float32)
    state[6] = 1.0
    state[13] = 1.0
    return {
        "cmd": "infer",
        "prompt": args.prompt,
        "head": rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8),
        "left": rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8),
        "right": rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8),
        "state": state,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--prompt",
        default="Pick up the object and place it into the target area.",
    )
    parser.add_argument(
        "--label",
        default="robotwin_policy",
        help="Free-form label included in the printed JSON summary.",
    )
    args = parser.parse_args()

    obs = _make_observation(args)
    with socket.create_connection((args.host, args.port), timeout=30.0) as sock:
        sock.settimeout(None)
        ping, _ = _call(sock, {"cmd": "ping"})
        if not ping.get("ok"):
            raise RuntimeError(f"ping failed: {ping}")

        for i in range(args.warmup):
            response, timing = _call(sock, obs)
            if "error" in response:
                raise RuntimeError(f"server error during warmup {i}: {response['error']}")
            action = np.asarray(response["action"], dtype=np.float32)
            if action.shape != (32, 14) or not np.isfinite(action).all():
                raise RuntimeError(f"invalid action during warmup {i}: shape={action.shape}")
            server_timing = response.get("_server_timing", {})
            server_obs_to_action = server_timing.get("server_obs_to_action_s", float("nan"))
            print(
                f"[warmup {i + 1}/{args.warmup}] "
                f"server_obs_to_action_s={server_obs_to_action:.6f} "
                f"payload_roundtrip_s={timing['payload_roundtrip_s']:.6f}"
            )

        timings: list[dict[str, float]] = []
        server_timings: list[dict[str, float]] = []
        for i in range(args.iters):
            response, timing = _call(sock, obs)
            if "error" in response:
                raise RuntimeError(f"server error during iter {i}: {response['error']}")
            action = np.asarray(response["action"], dtype=np.float32)
            if action.shape != (32, 14) or not np.isfinite(action).all():
                raise RuntimeError(f"invalid action during iter {i}: shape={action.shape}")
            server_timing = response.get("_server_timing", {})
            if "server_obs_to_action_s" not in server_timing:
                raise RuntimeError("server response did not include _server_timing.server_obs_to_action_s")
            timings.append(timing)
            server_timings.append(server_timing)
            print(
                f"[iter {i + 1}/{args.iters}] "
                f"server_obs_to_action_s={server_timing['server_obs_to_action_s']:.6f} "
                f"payload_roundtrip_s={timing['payload_roundtrip_s']:.6f} "
                f"client_total_s={timing['client_total_s']:.6f}"
            )

    payload_roundtrip = [item["payload_roundtrip_s"] for item in timings]
    client_total = [item["client_total_s"] for item in timings]
    client_encode = [item["client_encode_s"] for item in timings]
    client_decode = [item["client_decode_s"] for item in timings]
    server_obs_to_action = [item["server_obs_to_action_s"] for item in server_timings]
    server_request_to_action = [item["server_request_to_action_s"] for item in server_timings]
    server_request_decode = [item["server_request_decode_s"] for item in server_timings]
    result = {
        "label": args.label,
        "host": args.host,
        "port": args.port,
        "image_shape": [args.height, args.width, 3],
        "warmup": args.warmup,
        "server_obs_to_action": _summary(server_obs_to_action),
        "server_request_to_action": _summary(server_request_to_action),
        "server_request_decode_mean_s": float(statistics.fmean(server_request_decode)),
        "payload_roundtrip": _summary(payload_roundtrip),
        "client_total": _summary(client_total),
        "client_encode_mean_s": float(statistics.fmean(client_encode)),
        "client_decode_mean_s": float(statistics.fmean(client_decode)),
        "approx_action_chunks_per_second_server_obs_to_action": float(1.0 / statistics.fmean(server_obs_to_action)),
        "approx_action_chunks_per_second_payload_roundtrip": float(1.0 / statistics.fmean(payload_roundtrip)),
    }
    print("[benchmark_summary]")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
