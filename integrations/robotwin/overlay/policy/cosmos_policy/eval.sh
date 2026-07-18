#!/bin/bash
# RoboTwin closed-loop eval for Cosmos3-Nano-Policy-RoboTwin.
# The Cosmos model is served separately (cosmos .venv): start it first with
#   bash /path/to/cosmos-framework/scripts/eval/serve_robotwin_policy.sh
# then run this in the RoboTwin conda env.
#
# Usage (from RoboTwin root or anywhere):
#   bash policy/cosmos_policy/eval.sh <task_name> <task_config> <ckpt_setting> <seed> <gpu_id>
# e.g.
#   bash policy/cosmos_policy/eval.sh place_a2b_left demo_clean iter1000 0 0

policy_name=${COSMOS_POLICY_NAME:-cosmos_policy}
task_name=${1}
task_config=${2}
ckpt_setting=${3}     # just a label for the eval_result/ dir; the real ckpt is loaded by the server
seed=${4}
gpu_id=${5}

export CUDA_VISIBLE_DEVICES=${gpu_id}

# Blackwell GPU: force the NVIDIA Vulkan ICD + capabilities so SAPIEN doesn't fall
# back to software / mis-select an ICD (see FastWAM NOTE.md §5). Without this the
# sim may run but produce silently degraded renders -> depressed success rates.
export NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export PYTHONUNBUFFERED=1

if [[ -z "${CONDA_SH:-}" ]]; then
  if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: set CONDA_SH to <conda-base>/etc/profile.d/conda.sh" >&2
    exit 1
  fi
  CONDA_SH="$(conda info --base)/etc/profile.d/conda.sh"
fi
source "$CONDA_SH"
# The orchestrating Cosmos shell may already have the repo .venv active. Conda
# activation does not remove that venv from PATH, so python would still resolve to
# cosmos-framework/.venv/bin/python and miss RoboTwin/SAPIEN packages.
if [ -n "${VIRTUAL_ENV:-}" ]; then
  PATH="$(printf '%s' "$PATH" | awk -v RS=: -v ORS=: -v venv="$VIRTUAL_ENV/bin" '$0 != venv {print}' | sed 's/:$//')"
  unset VIRTUAL_ENV
fi
conda activate "${ROBOTWIN_CONDA_ENV:-robotwin}"
echo "[cosmos_policy/eval.sh] python=$(which python)"
python -c "import sapien.core as sapien; print('[cosmos_policy/eval.sh] sapien=' + sapien.__file__)"

cd "$(dirname "$0")/../.."   # -> RoboTwin root

# Optional endpoint override from the orchestrator (cosmos scripts/eval_robotwin.sh).
# Unset -> behaves exactly as before (uses deploy_policy.yml's server_host/port).
endpoint_overrides=()
[ -n "${COSMOS_SERVER_HOST:-}" ] && endpoint_overrides+=(--server_host "${COSMOS_SERVER_HOST}")
[ -n "${COSMOS_SERVER_PORT:-}" ] && endpoint_overrides+=(--server_port "${COSMOS_SERVER_PORT}")
[ -n "${COSMOS_TEST_NUM:-}" ] && endpoint_overrides+=(--test_num "${COSMOS_TEST_NUM}")

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config policy/${policy_name}/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --ckpt_setting ${ckpt_setting} \
    --seed ${seed} \
    --policy_name ${policy_name} \
    --instruction_type unseen \
    "${endpoint_overrides[@]}"
