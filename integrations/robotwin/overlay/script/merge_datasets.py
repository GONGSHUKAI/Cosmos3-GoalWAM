#!/usr/bin/env python3
"""Merge multiple RoboTwin collected-data folders into one, renumbering episodes.

Each source folder is expected to share the standard RoboTwin layout::

    <source>/
        data/episode<N>.hdf5
        instructions/episode<N>.json
        _traj_data/episode<N>.pkl
        video/episode<N>.mp4
        scene_info.json   # dict keyed "episode_<N>"
        seed.txt          # whitespace-separated seed per episode (positional)

Episodes from every source are concatenated in the order the sources are given
and renumbered 0..(total-1). The merged folder keeps the same structure; per-file
subfolders, ``scene_info.json`` and ``seed.txt`` are all renumbered consistently.

Usage (defaults reproduce the place_a2b_left 50+500 -> 550 merge)::

    python merge_datasets.py
    python merge_datasets.py --sources A B C --output OUT --mode copy
"""

import argparse
import json
import os
import re
import shutil

EPISODE_RE = re.compile(r"^episode(\d+)\.(.+)$")

DATA_ROOT = os.environ.get("ROBOTWIN_DATA_ROOT")
DEFAULT_SOURCES = None
DEFAULT_OUTPUT = None
if DATA_ROOT:
    DEFAULT_SOURCES = [
        os.path.join(DATA_ROOT, "robotwin2.0/place_a2b_left/aloha-agilex_clean_50"),
        os.path.join(DATA_ROOT, "robotwin2.0/place_a2b_left/aloha-agilex_randomized_500"),
    ]
    DEFAULT_OUTPUT = os.path.join(DATA_ROOT, "robotwin2.0/place_a2b_left/aloha-agilex_combined_550")


def discover_episode_subdirs(source):
    """Return {subdir_name: {episode_index: filename}} for episode<N>.<ext> files."""
    subdirs = {}
    for name in sorted(os.listdir(source)):
        path = os.path.join(source, name)
        if not os.path.isdir(path):
            continue
        episodes = {}
        for fname in os.listdir(path):
            m = EPISODE_RE.match(fname)
            if m:
                episodes[int(m.group(1))] = fname
        if episodes:
            subdirs[name] = episodes
    return subdirs


def count_episodes(source, subdirs):
    """Determine episode count for a source, asserting subfolders agree."""
    counts = {name: len(eps) for name, eps in subdirs.items()}
    n = max(counts.values())
    for name, eps in subdirs.items():
        missing = sorted(set(range(n)) - set(eps))
        if missing:
            raise ValueError(f"{source}/{name} missing episodes {missing}")
    return n


def place_file(src_file, dst_file, mode):
    if os.path.exists(dst_file) or os.path.islink(dst_file):
        os.remove(dst_file)
    if mode == "hardlink":
        os.link(src_file, dst_file)
    elif mode == "symlink":
        os.symlink(os.path.abspath(src_file), dst_file)
    elif mode == "copy":
        shutil.copy2(src_file, dst_file)
    else:
        raise ValueError(f"unknown mode {mode}")


def load_seeds(source, n):
    seed_path = os.path.join(source, "seed.txt")
    if not os.path.exists(seed_path):
        return [None] * n
    with open(seed_path) as f:
        seeds = f.read().split()
    if len(seeds) != n:
        raise ValueError(f"{seed_path}: {len(seeds)} seeds for {n} episodes")
    return seeds


def load_scene_info(source):
    p = os.path.join(source, "scene_info.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", nargs="+", default=DEFAULT_SOURCES, help="source folders, concatenated in this order")
    ap.add_argument("--output", default=DEFAULT_OUTPUT, help="merged output folder")
    ap.add_argument(
        "--mode",
        choices=["hardlink", "symlink", "copy"],
        default="hardlink",
        help="how to place episode files (default: hardlink)",
    )
    args = ap.parse_args()

    if not args.sources or not args.output:
        ap.error("pass --sources and --output, or set ROBOTWIN_DATA_ROOT")

    os.makedirs(args.output, exist_ok=True)

    merged_scene = {}
    merged_seeds = []
    all_subdir_names = set()
    offset = 0

    for source in args.sources:
        subdirs = discover_episode_subdirs(source)
        if not subdirs:
            raise ValueError(f"no episode files found under {source}")
        n = count_episodes(source, subdirs)
        all_subdir_names |= set(subdirs)

        for name in subdirs:
            os.makedirs(os.path.join(args.output, name), exist_ok=True)

        for i in range(n):
            new_i = offset + i
            for name, eps in subdirs.items():
                ext = EPISODE_RE.match(eps[i]).group(2)
                src_file = os.path.join(source, name, eps[i])
                dst_file = os.path.join(args.output, name, f"episode{new_i}.{ext}")
                place_file(src_file, dst_file, args.mode)

        scene = load_scene_info(source)
        if scene is not None:
            for i in range(n):
                key = f"episode_{i}"
                if key in scene:
                    merged_scene[f"episode_{offset + i}"] = scene[key]

        merged_seeds.extend(load_seeds(source, n))

        print(f"[merge] {source}: {n} episodes -> {offset}..{offset + n - 1}")
        offset += n

    total = offset

    if merged_scene:
        with open(os.path.join(args.output, "scene_info.json"), "w") as f:
            json.dump(merged_scene, f, indent=4)

    if any(s is not None for s in merged_seeds):
        seeds_out = [s if s is not None else "0" for s in merged_seeds]
        with open(os.path.join(args.output, "seed.txt"), "w") as f:
            f.write(" ".join(seeds_out) + " \n")

    print(f"[merge] done: {total} episodes in {sorted(all_subdir_names)} -> {args.output}")
    print(f"[merge] scene_info entries: {len(merged_scene)}, seeds: {len(merged_seeds)}")


if __name__ == "__main__":
    main()
