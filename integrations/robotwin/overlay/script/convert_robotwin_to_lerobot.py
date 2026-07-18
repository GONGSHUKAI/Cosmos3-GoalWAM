"""Convert raw RoboTwin (HDF5 + JPEG-in-HDF5 + JSON instructions) to LeRobot v2.1.

Input layout (e.g. ./place_a2b_left/):
    data/episode{N}.hdf5
        joint_action/vector: (T, 14)  -> action & observation.state
        observation/{head,left,right}_camera/rgb: (T,) JPEG bytes  -> 3 video tracks
    instructions/episode{N}.json
        {"seen": [str, ...], "unseen": [...]}  -> first "seen" entry is the task

Output layout (LeRobot v2.1, 30 fps):
    data/chunk-000/episode_{N:06d}.parquet
    videos/chunk-000/observation.images.cam_high/episode_{N:06d}.mp4
    videos/chunk-000/observation.images.cam_left_wrist/episode_{N:06d}.mp4
    videos/chunk-000/observation.images.cam_right_wrist/episode_{N:06d}.mp4
    meta/info.json
    meta/tasks.jsonl
    meta/episodes.jsonl
    meta/episodes_stats.jsonl

The 3 RoboTwin cams map to FastWAM's robotwin.yaml shape_meta:
    head_camera  -> cam_high
    left_camera  -> cam_left_wrist
    right_camera -> cam_right_wrist
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from PIL import Image

CAM_MAP = {
    "head_camera": "cam_high",
    "left_camera": "cam_left_wrist",
    "right_camera": "cam_right_wrist",
}
LEROBOT_CAM_KEYS = [f"observation.images.{v}" for v in CAM_MAP.values()]
FPS = 30


def _decode_jpegs(group: h5py.Dataset) -> list[np.ndarray]:
    out = []
    for raw in group[:]:
        img = Image.open(io.BytesIO(raw.tobytes() if hasattr(raw, "tobytes") else raw)).convert("RGB")
        out.append(np.asarray(img))
    return out


def _encode_video(frames: list[np.ndarray], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    h, w, _ = frames[0].shape
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{w}x{h}",
        "-r",
        str(FPS),
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-vf",
        f"scale={w}:{h}",
        "-an",
        str(out_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for frame in frames:
        proc.stdin.write(np.ascontiguousarray(frame).tobytes())
    proc.stdin.close()
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"ffmpeg failed (rc={rc}) for {out_path}")


def _episode_stats(values: np.ndarray) -> dict:
    """LeRobot v2.1 episodes_stats.jsonl entry — values shape: [T, D] for vectors, [T, C, H, W] for images."""
    if values.ndim == 2:
        T = values.shape[0]
        return {
            "min": values.min(axis=0).tolist(),
            "max": values.max(axis=0).tolist(),
            "mean": values.mean(axis=0).tolist(),
            "std": values.std(axis=0).tolist(),
            "count": [T],
        }
    if values.ndim == 1:
        T = values.shape[0]
        return {
            "min": [float(values.min())],
            "max": [float(values.max())],
            "mean": [float(values.mean())],
            "std": [float(values.std())],
            "count": [T],
        }
    raise ValueError(f"unexpected shape {values.shape}")


def _image_zero_stats(T: int) -> dict:
    """Per-channel zero stats — matches the format used by other RoboTwin LeRobot dumps."""
    return {
        "min": [[[0.0]], [[0.0]], [[0.0]]],
        "max": [[[0.0]], [[0.0]], [[0.0]]],
        "mean": [[[0.0]], [[0.0]], [[0.0]]],
        "std": [[[0.0]], [[0.0]], [[0.0]]],
        "count": [T],
    }


def _build_features(image_h: int, image_w: int, action_dim: int, state_dim: int) -> dict:
    feats: dict[str, dict] = {}
    for cam_key in LEROBOT_CAM_KEYS:
        feats[cam_key] = {
            "dtype": "video",
            "video_info": {
                "video.is_depth_map": False,
                "video.fps": float(FPS),
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "has_audio": False,
            },
            "shape": [image_h, image_w, 3],
            "names": ["height", "width", "channel"],
        }
    feats["observation.state"] = {"dtype": "float32", "shape": [state_dim], "names": None}
    feats["action"] = {"dtype": "float32", "shape": [action_dim], "names": None}
    feats["episode_index"] = {"dtype": "int64", "shape": [1], "names": None}
    feats["frame_index"] = {"dtype": "int64", "shape": [1], "names": None}
    feats["index"] = {"dtype": "int64", "shape": [1], "names": None}
    feats["task_index"] = {"dtype": "int64", "shape": [1], "names": None}
    feats["timestamp"] = {"dtype": "float32", "shape": [1], "names": None}
    return feats


def _process_one_episode(args_tuple):
    src_dir, out_dir, ep_idx, global_index_start, task_index, image_h, image_w = args_tuple
    src_dir = Path(src_dir)
    out_dir = Path(out_dir)
    h5_path = src_dir / "data" / f"episode{ep_idx}.hdf5"

    with h5py.File(h5_path, "r") as f:
        action_vec = np.asarray(f["joint_action/vector"][:], dtype=np.float32)  # [T, 14]
        T = action_vec.shape[0]
        cam_frames: dict[str, list[np.ndarray]] = {}
        for hdf_cam, lerobot_cam in CAM_MAP.items():
            grp = f[f"observation/{hdf_cam}/rgb"]
            cam_frames[lerobot_cam] = _decode_jpegs(grp)
            assert len(cam_frames[lerobot_cam]) == T, f"frames/action mismatch for ep{ep_idx} cam {lerobot_cam}"
            fh, fw, _ = cam_frames[lerobot_cam][0].shape
            assert fh == image_h and fw == image_w, (
                f"cam size mismatch {fh}x{fw} vs {image_h}x{image_w} ep{ep_idx} cam {lerobot_cam}"
            )

    chunk = ep_idx // 1000
    for lerobot_cam, frames in cam_frames.items():
        out_video = (
            out_dir
            / "videos"
            / f"chunk-{chunk:03d}"
            / f"observation.images.{lerobot_cam}"
            / f"episode_{ep_idx:06d}.mp4"
        )
        _encode_video(frames, out_video)

    df = pd.DataFrame(
        {
            "observation.state": [action_vec[i] for i in range(T)],
            "action": [action_vec[i] for i in range(T)],
            "episode_index": np.full(T, ep_idx, dtype=np.int64),
            "frame_index": np.arange(T, dtype=np.int64),
            "index": np.arange(global_index_start, global_index_start + T, dtype=np.int64),
            "task_index": np.full(T, task_index, dtype=np.int64),
            "timestamp": (np.arange(T, dtype=np.float32) / FPS).astype(np.float32),
        }
    )
    out_parquet = out_dir / "data" / f"chunk-{chunk:03d}" / f"episode_{ep_idx:06d}.parquet"
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_parquet, index=False)

    state_stats = _episode_stats(action_vec)
    action_stats = _episode_stats(action_vec)
    img_stats = _image_zero_stats(T)
    return ep_idx, T, state_stats, action_stats, img_stats


def _load_task_for_episode(src_dir: Path, ep_idx: int) -> str:
    p = src_dir / "instructions" / f"episode{ep_idx}.json"
    if not p.exists():
        return "Robot manipulation task."
    with p.open() as f:
        d = json.load(f)
    seen = d.get("seen") or d.get("unseen") or []
    if not seen:
        return "Robot manipulation task."
    return str(seen[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, required=True, help="raw RoboTwin dir, e.g. ./place_a2b_left")
    parser.add_argument("--dst", type=str, required=True, help="LeRobot dst dir, e.g. ./data/place_a2b_left_lerobot")
    parser.add_argument(
        "--single-task",
        action="store_true",
        help="debug only: use one task string for all episodes; "
        "by default each episode gets its own instruction/task_index",
    )
    parser.add_argument("--robot-type", type=str, default="aloha-agilex")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.single_task:
        print("[!] --single-task is for debugging only; per-episode instructions will be collapsed.")

    src = Path(args.src).resolve()
    dst = Path(args.dst).resolve()
    if dst.exists():
        print(f"[!] dst exists, removing: {dst}")
        shutil.rmtree(dst)
    (dst / "meta").mkdir(parents=True, exist_ok=True)

    h5_files = sorted((src / "data").glob("episode*.hdf5"), key=lambda p: int(p.stem.replace("episode", "")))
    if args.max_episodes is not None:
        h5_files = h5_files[: args.max_episodes]
    ep_indices = [int(p.stem.replace("episode", "")) for p in h5_files]
    print(f"[i] found {len(ep_indices)} episodes in {src}")

    # Probe one episode for image dimensions.
    with h5py.File(h5_files[0], "r") as f:
        sample_raw = f["observation/head_camera/rgb"][0]
        sample_img = Image.open(io.BytesIO(sample_raw.tobytes()))
        image_w, image_h = sample_img.size  # PIL: (W, H)
        action_dim = int(f["joint_action/vector"].shape[1])
    print(f"[i] image HxW = {image_h}x{image_w}, action_dim = {action_dim}")

    # Build tasks.jsonl
    if args.single_task:
        task_str = _load_task_for_episode(src, ep_indices[0])
        tasks = [task_str]
        ep_to_task_index = {ep: 0 for ep in ep_indices}
    else:
        seen_tasks: dict[str, int] = {}
        tasks = []
        ep_to_task_index = {}
        for ep in ep_indices:
            t = _load_task_for_episode(src, ep)
            if t not in seen_tasks:
                seen_tasks[t] = len(tasks)
                tasks.append(t)
            ep_to_task_index[ep] = seen_tasks[t]
    with (dst / "meta" / "tasks.jsonl").open("w") as f:
        for i, t in enumerate(tasks):
            f.write(json.dumps({"task_index": i, "task": t}) + "\n")
    print(f"[i] wrote {len(tasks)} task(s)")

    # First pass: gather episode lengths so we can assign global `index` ranges.
    lengths: dict[int, int] = {}
    for h5_path, ep in zip(h5_files, ep_indices):
        with h5py.File(h5_path, "r") as f:
            lengths[ep] = int(f["joint_action/vector"].shape[0])

    global_starts: dict[int, int] = {}
    cum = 0
    for ep in ep_indices:
        global_starts[ep] = cum
        cum += lengths[ep]
    total_frames = cum

    # Process in parallel.
    work = [(str(src), str(dst), ep, global_starts[ep], ep_to_task_index[ep], image_h, image_w) for ep in ep_indices]
    results: dict[int, tuple] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_process_one_episode, w) for w in work]
        for i, fut in enumerate(as_completed(futs)):
            ep, T, s_stats, a_stats, img_stats = fut.result()
            results[ep] = (T, s_stats, a_stats, img_stats)
            if (i + 1) % 25 == 0 or i + 1 == len(futs):
                print(f"  [{i + 1}/{len(futs)}] done")

    # Write episodes.jsonl + episodes_stats.jsonl in episode order.
    with (dst / "meta" / "episodes.jsonl").open("w") as fe, (dst / "meta" / "episodes_stats.jsonl").open("w") as fs:
        for ep in ep_indices:
            T, s_stats, a_stats, img_stats = results[ep]
            task_str = tasks[ep_to_task_index[ep]]
            fe.write(json.dumps({"episode_index": ep, "tasks": [task_str], "length": T}) + "\n")
            stats = {
                "observation.state": s_stats,
                "action": a_stats,
                "episode_index": {"min": [ep], "max": [ep], "mean": [float(ep)], "std": [0.0], "count": [T]},
                "frame_index": {
                    "min": [0],
                    "max": [T - 1],
                    "mean": [(T - 1) / 2.0],
                    "std": [float(np.std(np.arange(T)))],
                    "count": [T],
                },
                "index": {
                    "min": [global_starts[ep]],
                    "max": [global_starts[ep] + T - 1],
                    "mean": [global_starts[ep] + (T - 1) / 2.0],
                    "std": [float(np.std(np.arange(T)))],
                    "count": [T],
                },
                "task_index": {
                    "min": [ep_to_task_index[ep]],
                    "max": [ep_to_task_index[ep]],
                    "mean": [float(ep_to_task_index[ep])],
                    "std": [0.0],
                    "count": [T],
                },
                "timestamp": {
                    "min": [0.0],
                    "max": [(T - 1) / FPS],
                    "mean": [(T - 1) / (2.0 * FPS)],
                    "std": [float(np.std(np.arange(T) / FPS))],
                    "count": [T],
                },
            }
            for cam_key in LEROBOT_CAM_KEYS:
                stats[cam_key] = img_stats
            fs.write(json.dumps({"episode_index": ep, "stats": stats}) + "\n")

    info = {
        "codebase_version": "v2.1",
        "robot_type": args.robot_type,
        "total_episodes": len(ep_indices),
        "total_frames": total_frames,
        "total_tasks": len(tasks),
        "total_videos": len(ep_indices) * len(LEROBOT_CAM_KEYS),
        "total_chunks": (max(ep_indices) // 1000) + 1,
        "chunks_size": 1000,
        "fps": FPS,
        "splits": {"train": f"0:{len(ep_indices)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": _build_features(image_h, image_w, action_dim, action_dim),
    }
    with (dst / "meta" / "info.json").open("w") as f:
        json.dump(info, f, indent=4)

    print(f"[OK] wrote LeRobot dataset: {dst}")
    print(f"     episodes={len(ep_indices)}, frames={total_frames}, tasks={len(tasks)}")


if __name__ == "__main__":
    main()
