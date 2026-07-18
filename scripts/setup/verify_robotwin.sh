#!/usr/bin/env bash
set -euo pipefail

COSMOS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$COSMOS_ROOT/scripts/_common.sh"
cosmos_load_local_env
ROBOTWIN_DIR="${ROBOTWIN_DIR:-$COSMOS_ROOT/external/RoboTwin}"

python - <<'PY'
import importlib.metadata as metadata
import sys
import torch

assert sys.version_info[:2] == (3, 10), sys.version
assert torch.version.cuda == "12.8", torch.version.cuda
for name in ("sapien", "mplib", "open3d", "warp-lang", "pytorch3d", "nvidia-curobo"):
    print(f"{name}={metadata.version(name)}")
print(f"python={sys.version.split()[0]} torch={torch.__version__} cuda={torch.version.cuda}")
PY

test -d "$ROBOTWIN_DIR/assets/embodiments"
test -f "$ROBOTWIN_DIR/policy/cosmos_policy/deploy_policy.py"
test -f "$ROBOTWIN_DIR/script/eval_policy.py"
echo "[verify_robotwin] source, environment, overlay, and assets are present"
