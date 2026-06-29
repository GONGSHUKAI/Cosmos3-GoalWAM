# Plan: Cosmos3-Nano-Policy-RoboTwin (DROID-style action policy on RoboTwin place_a2b_left)

## Context

Goal: train Cosmos3-Nano as a joint **future-video + future-action** policy on RoboTwin data, mirroring the
DROID recipe (`Cosmos3-Nano-Policy-DROID`). Start with one task (`place_a2b_left`) as a validation
experiment, but name everything generically (`robotwin`) so more RoboTwin tasks can be added later.

Cosmos has no aloha/RoboTwin dataset class — only DROID (8-D single-arm), robomind_franka (20-D cartesian),
agibot, bridge. So new code is required. The user already converted the data to **lerobot v3.0** at
`/pfs/pfs-7jnepv/shukaigong/data/robotwin2.0/place_a2b_left/aloha-agilex_combined_550_lerobot_v3.0`.

**Verified data facts** (this v3.0 matches the DROID v3.0 schema the base class expects):
- `meta/info.json`: codebase_version v3.0, robot_type aloha-agilex, fps 30, 550 episodes, 82387 frames,
  features `observation.state`[14], `action`[14], cameras `cam_high / cam_left_wrist / cam_right_wrist`.
- `meta/episodes/chunk-000/file-000.parquet` has the fields the base loader needs:
  `videos/<key>/{from_timestamp,chunk_index,file_index}`, `data/{chunk,file}_index`, `length`, `tasks`.
- `data/chunk-000/file-000.parquet`: 550 episodes aggregated; cols observation.state(14)/action(14)/
  episode_index/frame_index/index/task_index/timestamp. In this data **action == observation.state**
  (the converter set both = RoboTwin `joint_action/vector`).
- 14-D action layout = `[left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)]`.

Because the v3.0 layout matches, **`ActionBaseDataset` loads it natively — no edit to `base_dataset.py`.**

## Modeling decisions (mirror DROID, adapt to aloha)

- `domain_name="robotwin"`, new `domain_id=16` (unused id; `num_embodiment_domains=32` allows 0-31).
- `action_space="joint"` → 14-D; `action_dim=14`; `max_action_dim=64` pads 14→64 (model unchanged).
- `_action_spec` = `build_action_spec(Joint(n=6,prefix="left"), Gripper(prefix="left"), Joint(n=6,prefix="right"), Gripper(prefix="right"))` (14-D).
- `use_state=True`: window is chunk+1 frames; row0 = initial `observation.state`, rows[1:] = `action` chunk.
- `action_normalization=None` (raw joint angles, like DROID joint_pos). No stats file needed.
- **No gripper flip** (unlike DROID): output raw 14-D vector so it feeds RoboTwin's joint interface directly at eval time. (Noted as the one convention to revisit when the eval adapter is built.)
- `viewpoint="concat_view"`: `cam_high` (head) full-width on top; `cam_left_wrist` + `cam_right_wrist`
  resized to half and concatenated on the bottom row — same structure as DROID's wrist-top/exterior-bottom.
- `chunk_length=32`, fps=30 → `encode_exact_durations=[33]`; `resolution="384x320"` (RoboTwin concat
  canvas is resized to 384×320 with no square padding).
- Joint video+action: keep `vision_gen=True` + `action_gen=True` (NANO_MODEL_CONFIG) and the DROID overrides
  `loss_scale=10.0` (vision) + `action_loss_weight=10.0`. No keep_ranges filter (RoboTwin has none).

## New files

1. **`cosmos_framework/data/vfm/action/datasets/robotwin_lerobot_dataset.py`** — `RoboTwinLeRobotDataset(ActionBaseDataset)`.
   Closely mirrors `droid_lerobot_dataset.py` but:
   - `_IMAGE_FEATURES = {"head":"observation.images.cam_high","left":"observation.images.cam_left_wrist","right":"observation.images.cam_right_wrist"}`
   - reads base `_rows`/`_episodes`/`_tasks` (82k frames is small — no need for DROID's compact-array optimization);
     groups frames into per-episode windows (`np.unique` on episode_index → valid windows = len-chunk).
   - `_build_joint_action`: 14-D from `action` col rows[1:]; prepend `observation.state` row0 when use_state; no flip.
   - `_load_concat_video`: like DROID's, 3 RoboTwin cams, head-top / wrists-bottom concat; reuses base `_video_path`
     and `decode_video_frames` with `from_timestamp` from meta/episodes (works on the aggregated mp4).
   - `_action_spec`, `action_dim=14`, `_stats_path` (returns a path; never used since normalization=None),
     `get_shuffle_blocks` (per-episode blocks, like DROID).
   - constructor args mirror DROID minus the filter_dict/ee_pose paths (single `joint` action space).

2. **`cosmos_framework/configs/base/experiment/action/posttrain_config/action_policy_robotwin_nano.py`** —
   copy of `action_policy_droid_nano.py` with: `domain`/dataset swapped to RoboTwin, dataloader factory
   `get_action_robotwin_sft_dataset(root="${oc.env:ROBOTWIN_ROOT}", fps=30, chunk_length=32, action_space="joint",
   use_state=True, viewpoint="concat_view", resolution="384x320", max_action_dim=64, iterable_shuffle=True, ...)`,
   `encode_exact_durations=[33]`, `max_num_tokens_after_packing=-1`, `loss_scale=10.0`; registered via the same
   `cs.store(group="experiment", package="_global_", name="action_policy_robotwin_nano", ...)` block.

3. **`examples/toml/sft_config/action_policy_robotwin.toml`** — copy of `action_policy_droid_repro.toml`;
   `[job].experiment="action_policy_robotwin_nano"`, `name="action_policy_robotwin"`, VAE path, parallelism,
   `max_iter`, `save_iter` (same scalars).

4. **`examples/launch_sft_action_policy_robotwin.sh`** — copy of `launch_sft_action_policy_droid.sh`; bridges
   `DATASET_PATH`→`ROBOTWIN_ROOT`, checks `$ROBOTWIN_ROOT/meta/info.json`, sources `_sft_launcher_common.sh`.

5. **`scripts/train_robotwin.sh`** — like `scripts/train_droid.sh`: sets `DATASET_PATH`/`ROBOTWIN_ROOT` to the
   v3.0 dir, `BASE_CHECKPOINT_PATH`, `WAN_VAE_PATH`, wandb env, `EXTRA_TAIL_OVERRIDES` with
   `dataloader_train.max_samples_per_batch=32` and `optimizer.lr=3.5e-5` (no keep_ranges).

## Edited files (all additive, v3.0/DROID paths untouched)

6. **`cosmos_framework/data/vfm/action/domain_utils.py`** — add `"robotwin": 16` to `EMBODIMENT_TO_DOMAIN_ID`
   and `"robotwin": 14` to `EMBODIMENT_TO_RAW_ACTION_DIM` (required: base `get_domain_id` raises if missing).

7. **`cosmos_framework/data/vfm/action/datasets/__init__.py`** — import + `__all__` export `RoboTwinLeRobotDataset`.

8. **`cosmos_framework/data/vfm/action/datasets/action_sft_dataset.py`** — add `get_action_robotwin_sft_dataset(...)`
   factory mirroring `get_action_droid_sft_dataset` (instantiate `RoboTwinLeRobotDataset` + `ActionTransformPipeline`
   + `ActionSFTDataset` + optional `ActionIterableShuffleDataset`); no filter_dict args.

9. **`cosmos_framework/configs/base/config.py`** — add the import line that triggers `cs.store()` registration of
   `action_policy_robotwin_nano` (next to the DROID import).

## Verification (end-to-end smoke)

1. Import/registration sanity: `python -c "import cosmos_framework.configs.base.config"` then confirm the
   experiment resolves (dryrun): `PYTHONPATH=. python -m cosmos_framework.scripts.train
   --sft-toml=examples/toml/sft_config/action_policy_robotwin.toml --dryrun -- job.wandb_mode=disabled`
   with `ROBOTWIN_ROOT`/`BASE_CHECKPOINT_PATH`/`WAN_VAE_PATH` exported. Should print the config and exit 0.
2. Dataset unit check: instantiate `RoboTwinLeRobotDataset(root=<v3.0>, ...)`, pull `ds[0]`, assert
   `video` shape `[3,33,H,W]` uint8, `action` shape `[33,14]`, caption non-empty.
3. 10-iter smoke on 8×B20Z: `EXTRA_TAIL_OVERRIDES="job.wandb_mode=disabled trainer.max_iter=10
   checkpoint.save_iter=10 dataloader_train.max_samples_per_batch=32 model.config.compile.enabled=False"
   bash examples/launch_sft_action_policy_robotwin.sh` — expect loss printed, no crash, checkpoint at iter_10.
4. Then real run via `scripts/train_robotwin.sh`.

## Notes / deferred
- Gripper-flip and `resolution` are the two knobs most likely to need revisiting after first results.

---
---

# PART 2 — RoboTwin closed-loop eval infrastructure (Cosmos3-Nano-Policy-RoboTwin)

## Context

We can now train `Cosmos3-Nano-Policy-RoboTwin` (Part 1). To measure it we need RoboTwin closed-loop eval.
**Hard constraint:** the RoboTwin conda env is **py3.10 / torch 2.4.1+cu121 / sapien 3.0.0b1**; cosmos `.venv`
is **py3.13 / torch 2.10+cu130**. They cannot coexist in one process. FastWAM ran its model *in-process*
inside RoboTwin's eval because FastWAM's deps matched RoboTwin's env — **cosmos cannot**. So eval must be a
**server (cosmos venv) / client (robotwin env) split**, talking over a plain socket.

```
Cosmos inference server (cosmos .venv, py3.13)        RoboTwin eval client (robotwin conda env, py3.10)
  loads trained RoboTwin DCP via OmniInference          script/eval_policy.py drives the sapien sim
  obs(3 imgs + 14D state + prompt) → action[32,14]  ◄── policy/cosmos_policy/deploy_policy.py (socket client)
```

Reference that defines every interface: FastWAM `experiments/robotwin/fastwam_policy/deploy_policy.py`
(in-process analog) and cosmos `action_policy_server_robolab.py` (model-load + transform path to reuse).

## Known interface facts (verified)
- RoboTwin policy = `policy/<name>/{deploy_policy.py (encode_obs/get_model/eval/reset_model), deploy_policy.yml, eval.sh}`;
  `eval_policy.py` resolves `--policy_name` → that module. eval.sh args: `<task> <task_config> <ckpt_setting> <seed> <gpu>`.
- obs from `TASK_ENV.get_obs()`: `observation/{head,left,right}_camera/rgb` (uint8 HxWx3) + `joint_action/vector` [14].
- action back via `TASK_ENV.take_action(action, action_type="qpos")`, qpos = `[left_arm6, left_grip, right_arm6, right_grip]`
  (== our training layout). Language via `TASK_ENV.get_instruction()`. Receding horizon: predict 32, exec `replan_steps`, re-request.
- Server reuse: `OmniInference.create(setup_args)` with `checkpoint_path`+`experiment`+`sampler`
  (`action_policy_server_robolab.py:357-434`); `ActionTransformPipeline`; concat geometry from
  `RoboTwinLeRobotDataset._load_concat_video`.

## Correctness requirement (or scores are meaningless)
Server preprocessing MUST byte-match training: concat_view (head full-width top; left/right wrist resized to
half, concatenated bottom), `resolution="384x320"`, 14-D state row prepended (`use_state`), `domain_name="robotwin"`
(id 16), **no gripper flip**, `action_normalization=None`, `mode="policy"`, video = `[3,33,H,W]` with frame0 =
current obs and the rest zeros (vision is generated). These mirror `RoboTwinLeRobotDataset` + the experiment.

## New files

1. **`cosmos_framework/scripts/action_policy_server_robotwin.py`** (runs in cosmos `.venv`) — minimal TCP server.
   - Loads the trained checkpoint via the robolab path: `OmniInference.create` with `checkpoint_path=<iter_N DCP>`,
     `experiment="action_policy_robotwin_nano"`, `--allow-dcp-checkpoint` semantics (reuse `_resolve_checkpoint_path`
     / `_build_setup_args` logic from `action_policy_server_robolab.py`).
   - `infer(obs)`: build the eval sample mirroring `RoboTwinLeRobotDataset` (concat 3 cams → frame0 of a `[3,33,H,W]`
     video; `[33,14]` action with row0=state; prompt→ai_caption; domain robotwin; viewpoint concat_view), run
     `ActionTransformPipeline` (res 384x320, max_action_dim 64), `model.generate_samples_from_batch(guidance,num_steps,shift,seed)`,
     return `action[:, :14]` with the state/history row dropped. No flip, no denorm.
   - Protocol: length-prefixed JSON+base64-numpy (mirror RoboTwin `eval_policy_client.py` wire format) so the
     client needs only socket+numpy — **no openpi, no torch-heavy deps on the robotwin side**. Commands: `infer`, `reset`, `ping`.
   - CLI: `--checkpoint-path`, `--experiment action_policy_robotwin_nano`, `--port`, `--guidance`, `--num-steps`,
     `--shift`, `--resolution 384x320`, `--seed`.

2. **`RoboTwin/policy/cosmos_policy/deploy_policy.py`** (runs in robotwin env) — mirrors `fastwam_policy`:
   - `get_model(usr_args)` → opens socket to the server (host/port from yml), returns a thin client holding a
     `deque` of pending actions + `replan_steps`.
   - `encode_obs(obs)` → `(head_rgb, left_rgb, right_rgb, vector14)`.
   - `eval(TASK_ENV, model, observation)` → if queue empty: `instruction=TASK_ENV.get_instruction()`, send
     `{prompt,instruction-formatted, head,left,right, state}` → recv `action[32,14]`, push first `replan_steps`;
     pop one, `TASK_ENV.take_action(action, action_type="qpos")`. (Same receding-horizon shape as fastwam.)
   - `reset_model(model)` → clear queue + send `reset`.

3. **`RoboTwin/policy/cosmos_policy/deploy_policy.yml`** — `policy_name/task_name/.../instruction_type` (standard)
   plus `server_host`, `server_port`, `replan_steps` (default 8), `num_inference_steps`, `guidance`.

4. **`RoboTwin/policy/cosmos_policy/eval.sh`** — activates the **robotwin conda env** (not a local .venv), exports
   the Blackwell Vulkan/OIDN vars (FastWAM NOTE §5: `VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json`,
   `NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics`), `cd` to RoboTwin root, runs `script/eval_policy.py
   --config policy/cosmos_policy/deploy_policy.yml --overrides --task_name … --policy_name cosmos_policy`.

5. **`scripts/serve_robotwin_policy.sh`** (cosmos repo) — convenience launcher for file 1 in the cosmos `.venv`
   (sets env, points `--checkpoint-path` at the latest `…/action_policy_robotwin/checkpoints/iter_<N>`).

## Prerequisites / setup (mostly done; verify at build time)
- RoboTwin assets + `task_config/` already downloaded into `/pfs/.../code/RoboTwin` (user ran `_download_assets.sh`).
- Blackwell: confirm `vulkaninfo --summary` shows NVIDIA ICD and bump SAPIEN bundled OIDN to 2.3.3 in the
  robotwin env (FastWAM NOTE §5) — else scores are silently depressed.
- A trained checkpoint exists at `…/action_policy_robotwin/checkpoints/iter_<N>/` (DCP). (`save_iter` now 100.)

## Verification (end-to-end, after a checkpoint exists)
1. Server up (cosmos venv): `bash scripts/serve_robotwin_policy.sh` → logs "ready", `ping` returns.
2. Standalone infer smoke: a tiny client sends one synthetic obs (3 zero images + zero 14-vector + a prompt) →
   server returns `action` shape `[32,14]`, finite. (No sim needed; proves the model path.)
3. Real eval (robotwin env): `cd RoboTwin && bash policy/cosmos_policy/eval.sh place_a2b_left demo_clean iter<N> 0 0`
   for a few episodes (`eval_num_episodes` small) → simulator runs, success-rate file written under RoboTwin eval output.
4. Sanity vs train task: eval `place_a2b_left` (the trained task) first; expect non-trivial success once trained.

## Open knobs / deferred
- `replan_steps` (8), `num_inference_steps` (e.g. 4-10), `guidance`, `shift` — tune after first numbers.
- Gripper convention: trained with **no flip**; if the arm opens/closes inverted in sim, flip here + retrain-free
  by flipping in the client (or fix at train time). This is the most likely first-run bug.
- Multi-task / multi-GPU eval orchestration (FastWAM's run_robotwin_manager analog) — later.
