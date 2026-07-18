# Cosmos3-Nano-RoboTwin-Policy-GoalWAM

This repository contains Cosmos3-Nano action-policy post-training and closed-loop
RoboTwin evaluation, including text baselines and goal-frame-guided GoalWAM
variants.

The current goal-conditioned evaluation uses an **oracle goal frame**: RoboTwin
runs its successful rule-based expert for the selected seed, captures the final
observation, and supplies those terminal camera frames to the policy. Generated
goal images are an extension point but are not implemented.

## Supported experiments

| Scenario | Condition | Training launcher | 50-task evaluation launcher |
| --- | --- | --- | --- |
| Clean + randomized training | Text baseline | `scripts/train/train_robotwin_50tasks_clean+rand.sh` | `scripts/eval/eval_robotwin_50tasks_clean+rand.sh` |
| Clean + randomized training | Text + oracle goal | `scripts/train/train_robotwin_50tasks_clean+rand_text_goal_frame_cond.sh` | `scripts/eval/eval_robotwin_50tasks_clean+rand_text_goal_frame_cond.sh` |
| Clean-to-random | Text baseline | `scripts/train/train_robotwin_50tasks_clean2rand.sh` | `scripts/eval/eval_robotwin_50tasks_clean2rand.sh` |
| Clean-to-random | Oracle goal only | `scripts/train/train_robotwin_50tasks_clean2rand_goal_frame_cond.sh` | `scripts/eval/eval_robotwin_50tasks_clean2rand_goal_frame_cond.sh` |
| Clean-to-random | Text + oracle goal | `scripts/train/train_robotwin_50tasks_clean2rand_text_goal_frame_cond.sh` | `scripts/eval/eval_robotwin_50tasks_clean2rand_text_goal_frame_cond.sh` |

All canonical variants use mean/std action normalization, offline concat-view
videos, and temporal video downsampling to frames `0,4,...,32` while retaining
the full 32-step action chunk.

## Architecture

Cosmos and RoboTwin intentionally use separate Python environments:

```text
Cosmos .venv (Python 3.13 / CUDA 13)
  action_policy_server_robotwin.py
  current cameras + state + text/goal -> action chunk [32, 14]
                    |
                    | length-prefixed TCP + JSON/NumPy codec
                    v
RoboTwin conda env (Python 3.10 / CUDA 12.8)
  SAPIEN simulator + policy/cosmos_policy socket client
```

Do not install RoboTwin into the Cosmos `.venv`. SAPIEN, MPLib, Curobo, and the
Cosmos training stack require different Python/PyTorch combinations.

## 1. Clone and local configuration

```bash
git clone git@github.com:GONGSHUKAI/Cosmos3-GoalWAM.git
cd Cosmos3-GoalWAM
cp config/local.env.example config/local.env
```

Edit `config/local.env` with the paths for the target cluster. The file is
ignored by Git. It is automatically loaded by the tracked training, download,
evaluation, and setup launchers.

Recommended authentication:

```bash
hf auth login
wandb login
```

For non-interactive jobs, `HF_TOKEN` and `WANDB_API_KEY` may instead be placed in
`config/local.env`. Never add them to a tracked shell script.

Important configuration variables:

| Variable | Purpose |
| --- | --- |
| `ROBOTWIN_DATA_ROOT` | Parent directory for RoboTwin raw and LeRobot datasets |
| `COSMOS_WEIGHTS_ROOT` | Parent directory for base, VAE, and exported checkpoints |
| `BASE_CHECKPOINT_PATH` | Cosmos3-Nano DCP checkpoint used for post-training |
| `WAN_VAE_PATH` | `Wan2.2_VAE.pth` used to encode policy videos |
| `ROBOTWIN_DIR` | RoboTwin source checkout; defaults to `external/RoboTwin` on a new cluster |
| `ROBOTWIN_CONDA_ENV` | RoboTwin conda environment name, normally `robotwin` |
| `CONDA_SH` | Path to `etc/profile.d/conda.sh` |
| `IMAGINAIRE_OUTPUT_ROOT` | Optional training output root |

## 2. Build the Cosmos `.venv`

System requirements include Linux, an NVIDIA GPU, CUDA 12.8 or newer, `ffmpeg`,
and `git-lfs`. The current training cluster uses CUDA 13.

```bash
sudo apt-get install -y --no-install-recommends \
  curl ffmpeg git-lfs libx11-dev tree wget

curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"

# CUDA 13 training environment. Use cu128-train on a CUDA 12.8 cluster.
uv sync --all-extras --group=cu130-train
source .venv/bin/activate
export LD_LIBRARY_PATH=

python -c "import cosmos_framework; print('Cosmos import OK')"
```

The dependency lock is `uv.lock`; do not copy the existing `.venv` between
clusters.

## 3. Download model inputs and training data

Download the Cosmos3-Nano base checkpoint and Wan2.2 VAE into the paths selected
in `config/local.env`:

```bash
source config/local.env
mkdir -p "$COSMOS_WEIGHTS_ROOT"

hf download nvidia/Cosmos3-Nano \
  --local-dir "$COSMOS_WEIGHTS_ROOT/Cosmos3-Nano"

hf download Wan-AI/Wan2.2-TI2V-5B Wan2.2_VAE.pth \
  --local-dir "$COSMOS_WEIGHTS_ROOT/Wan2.2-TI2V-5B"

source .venv/bin/activate
export LD_LIBRARY_PATH=
python -m cosmos_framework.scripts.convert_model_to_dcp \
  --checkpoint-path "$COSMOS_WEIGHTS_ROOT/Cosmos3-Nano" \
  -o "$BASE_CHECKPOINT_PATH"
```

The unified clean+random RoboTwin LeRobot dataset can be downloaded with:

```bash
bash scripts/download/download_robotwin_lerobot_v3.0_unified.sh
```

For clean-to-random training, download the official RoboTwin dataset and build
the 50-task clean-only LeRobot v3 dataset:

```bash
bash scripts/download/download_robotwin.sh

source .venv/bin/activate
export LD_LIBRARY_PATH=
python scripts/preprocess/prepare_robotwin_all_clean.py
```

The committed action-statistics JSON files must stay paired with the dataset
used for training. Evaluation uses the same file to denormalize the generated
14-dimensional qpos actions.

## 4. Configure RoboTwin on another cluster

RoboTwin is not vendored into this repository. The integration pins the official
source revision in `integrations/robotwin/UPSTREAM_COMMIT` and applies the local
evaluation patch and policy overlay during setup.

Install the system Vulkan dependencies first:

```bash
sudo apt-get install -y libvulkan1 mesa-vulkan-drivers vulkan-tools
vulkaninfo --summary
```

Create and activate a separate Python 3.10 environment, then run the setup:

```bash
conda create -n robotwin python=3.10.20 -y
conda activate robotwin

bash scripts/setup/setup_robotwin.sh
bash scripts/setup/download_robotwin_assets.sh
bash scripts/setup/verify_robotwin.sh
```

The setup uses `integrations/robotwin/requirements-cu128.txt`, including PyTorch
2.7.1+cu128, torchvision 0.22.1+cu128, SAPIEN 3.0.0b1, MPLib 0.2.1,
PyTorch3D 0.7.8, Curobo 0.7.8, and Warp 1.12.0. It also applies the SAPIEN/MPLib
source fixes and installs OIDN 2.3.3 for Blackwell rendering.

To prepare only the source checkout without installing packages:

```bash
bash scripts/setup/prepare_robotwin_source.sh
```

## 5. Training

Every launcher can be run from any working directory. It loads
`config/local.env`, activates `.venv`, validates the dataset/checkpoint/VAE
paths, determines the process count from `CUDA_VISIBLE_DEVICES`, and calls the
paired `examples/launch_sft_action_policy_robotwin.sh` launcher.

Examples:

```bash
# Text baseline, clean + randomized data.
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  bash scripts/train/train_robotwin_50tasks_clean+rand.sh

# Text + goal-frame GoalWAM, clean + randomized data.
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 GOAL_LAYOUT=concat \
  bash scripts/train/train_robotwin_50tasks_clean+rand_text_goal_frame_cond.sh

# Text baseline, clean-to-random setting.
bash scripts/train/train_robotwin_50tasks_clean2rand.sh

# Goal-frame-only GoalWAM, clean-to-random setting.
bash scripts/train/train_robotwin_50tasks_clean2rand_goal_frame_cond.sh

# Text + goal-frame GoalWAM, clean-to-random setting.
bash scripts/train/train_robotwin_50tasks_clean2rand_text_goal_frame_cond.sh
```

Supported goal layouts:

- `GOAL_LAYOUT=concat`: the same inverted-T three-camera composition used for
  the policy observation.
- `GOAL_LAYOUT=cam_high`: head camera only.

Re-running against the same `IMAGINAIRE_OUTPUT_ROOT` resumes from the latest DCP
checkpoint when the framework finds one.

## 6. Evaluation

Evaluation launchers start one Cosmos policy server in `.venv`, wait for its TCP
port, and then invoke RoboTwin in the separate conda environment.

Single-task examples:

```bash
# Clean+random text baseline.
bash scripts/eval/eval_robotwin_1task_clean+rand.sh \
  /path/to/checkpoints/iter_000032000 place_a2b_left demo_clean 0 0

# Clean2random text + oracle-goal GoalWAM.
bash scripts/eval/eval_robotwin_1task_clean2rand_text_goal_frame_cond.sh \
  /path/to/checkpoints/iter_000006000 place_a2b_left demo_randomized 0 0
```

All-50-task examples:

```bash
# Inspect task assignment and commands without loading a model.
DRY_RUN=1 GPUS=0,1 \
  bash scripts/eval/eval_robotwin_50tasks_clean2rand_text_goal_frame_cond.sh \
  /path/to/checkpoints/iter_000006000 random

# Full run. TEST_NUM controls episodes per task.
GPUS=0,1,2,3,4,5,6,7 TEST_NUM=100 \
  bash scripts/eval/eval_robotwin_50tasks_clean2rand_text_goal_frame_cond.sh \
  /path/to/checkpoints/iter_000006000 random
```

Useful evaluation overrides include `TEST_NUM`, `TASKS`, `GPUS`, `EVAL_SEED`,
`REPLAN_STEPS`, `ROBOTWIN_NUM_STEPS`, `PORT`, and `RUN_SMOKE=1`.

Evaluation output is written under `evaluate_results/` on the Cosmos side and
`eval_result/` under the RoboTwin checkout. Both are runtime artifacts and are
not committed.

## 7. Verification and troubleshooting

Before a long experiment:

```bash
# Shell syntax.
find scripts -name '*.sh' -not -path 'scripts/deprecated/*' -print0 \
  | xargs -0 -n1 bash -n

# Python lint for changed code.
uv run ruff check cosmos_framework scripts integrations

# Synthetic server/client request after starting a server.
python scripts/eval/smoke_robotwin_client.py --port 9876
```

Common problems:

- Clear `LD_LIBRARY_PATH` after activating `.venv`; otherwise PyTorch may import
  incompatible host libraries.
- A checkpoint's `cond`, `goal_layout`, temporal downsampling, and action
  normalization must match evaluation. The server reads them from the saved
  training configuration and fails on incompatible explicit overrides.
- Clean+random and clean-to-random models use different action-statistics files.
- On Blackwell, verify that `vulkaninfo --summary` sees the NVIDIA GPU. The
  RoboTwin policy launcher sets `VK_ICD_FILENAMES` and the setup installs OIDN
  2.3.3 to avoid degraded SAPIEN rendering.
- Do not run multiple evaluations on the same TCP port.

## Repository map

- `cosmos_framework/`: Cosmos training, model, data, checkpoint, and policy
  server implementation.
- `examples/toml/sft_config/`: baseline and goal-frame SFT recipes.
- `scripts/train/`: reproducible training entry points.
- `scripts/eval/`: single-task and 50-task server/client orchestration.
- `scripts/preprocess/`: RoboTwin-to-LeRobot preparation and offline concat
  generation.
- `integrations/robotwin/`: pinned upstream revision, patch, overlay, and
  requirements for the external simulator checkout.
- `docs/`: original Cosmos framework setup, training, inference, and reference
  documentation.

## Upstream projects and licenses

This work builds on the NVIDIA Cosmos framework and the RoboTwin benchmark.
Retain the upstream license headers and consult the respective repositories for
their model, code, dataset, and asset licenses:

- <https://github.com/NVIDIA/cosmos-framework>
- <https://github.com/RoboTwin-Platform/RoboTwin>

The original Cosmos documentation remains available in `docs/setup.md`,
`docs/training.md`, `docs/inference.md`, and `docs/code_structure.md`.
