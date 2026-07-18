#!/usr/bin/env bash
set -euo pipefail

COSMOS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$COSMOS_ROOT/scripts/_common.sh"
cosmos_load_local_env

if [[ -n "${VIRTUAL_ENV:-}" && "$VIRTUAL_ENV" == "$COSMOS_ROOT/.venv" ]]; then
  echo "ERROR: do not install RoboTwin into the Cosmos .venv." >&2
  echo "Activate a separate Python 3.10 conda environment first." >&2
  exit 1
fi

python - <<'PY'
import sys
if sys.version_info[:2] != (3, 10):
    raise SystemExit(f"RoboTwin requires Python 3.10; active interpreter is {sys.version.split()[0]}")
PY

bash "$COSMOS_ROOT/scripts/setup/prepare_robotwin_source.sh"
ROBOTWIN_DIR="${ROBOTWIN_DIR:-$COSMOS_ROOT/external/RoboTwin}"

python -m pip install --upgrade pip
python -m pip install --no-build-isolation -r "$COSMOS_ROOT/integrations/robotwin/requirements-cu128.txt"

python - <<'PY'
from pathlib import Path
import mplib
import sapien

sapien_loader = Path(sapien.__file__).resolve().parent / "wrapper" / "urdf_loader.py"
text = sapien_loader.read_text()
text = text.replace('with open(urdf_file, "r") as f:', 'with open(urdf_file, "r", encoding="utf-8") as f:')
text = text.replace('srdf_file = urdf_file[:-4] + "srdf"', 'srdf_file = urdf_file[:-4] + ".srdf"')
text = text.replace('with open(srdf_file, "r") as f:', 'with open(srdf_file, "r", encoding="utf-8") as f:')
sapien_loader.write_text(text)

planner = Path(mplib.__file__).resolve().parent / "planner.py"
text = planner.read_text()
text = text.replace(
    "if np.linalg.norm(delta_twist) < 1e-4 or collide or not within_joint_limit:",
    "if np.linalg.norm(delta_twist) < 1e-4 or not within_joint_limit:",
)
planner.write_text(text)
print(f"[setup_robotwin] patched {sapien_loader}")
print(f"[setup_robotwin] patched {planner}")
PY

CUROBO_DIR="$ROBOTWIN_DIR/envs/curobo"
if [[ ! -d "$CUROBO_DIR/.git" ]]; then
  mkdir -p "$(dirname "$CUROBO_DIR")"
  git clone --branch v0.7.8 --depth 1 https://github.com/NVlabs/curobo.git "$CUROBO_DIR"
fi

CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
CUDA_PATH="$CUDA_HOME" \
PATH="$CUDA_HOME/bin:$PATH" \
LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}" \
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0;10.0}" \
CC="${CC:-/usr/bin/gcc}" CXX="${CXX:-/usr/bin/g++}" CUDAHOSTCXX="${CUDAHOSTCXX:-/usr/bin/g++}" \
python -m pip install -e "$CUROBO_DIR" --no-build-isolation

if [[ "${INSTALL_OIDN_FIX:-1}" == "1" ]]; then
  OIDN_VERSION=2.3.3
  OIDN_SHA256=3c385230d9e6f63527ba72f2229594dbac5051674219d72e0044b5d0b841796f
  OIDN_ARCHIVE="${TMPDIR:-/tmp}/oidn-${OIDN_VERSION}.x86_64.linux.tar.gz"
  OIDN_ROOT="${TMPDIR:-/tmp}/oidn-${OIDN_VERSION}.x86_64.linux"
  if [[ ! -f "$OIDN_ARCHIVE" ]]; then
    curl -L --fail --retry 3 \
      "https://github.com/RenderKit/oidn/releases/download/v${OIDN_VERSION}/oidn-${OIDN_VERSION}.x86_64.linux.tar.gz" \
      -o "$OIDN_ARCHIVE"
  fi
  echo "$OIDN_SHA256  $OIDN_ARCHIVE" | sha256sum --check -
  rm -rf "$OIDN_ROOT"
  tar -xzf "$OIDN_ARCHIVE" -C "${TMPDIR:-/tmp}"

  SAPIEN_DIR="$(python - <<'PY'
from pathlib import Path
import sapien
print(Path(sapien.__file__).resolve().parent)
PY
)"
  cp "$OIDN_ROOT/lib/libOpenImageDenoise.so.${OIDN_VERSION}" "$SAPIEN_DIR/oidn_library/"
  cp "$OIDN_ROOT/lib/libOpenImageDenoise_core.so.${OIDN_VERSION}" "$SAPIEN_DIR/oidn_library/"
  cp "$OIDN_ROOT/lib/libOpenImageDenoise_device_cuda.so.${OIDN_VERSION}" "$SAPIEN_DIR/oidn_library/"
  sed -i "s/2\.0\.1/${OIDN_VERSION}/g" "$SAPIEN_DIR/_oidn_tricks.py"
  echo "[setup_robotwin] installed OIDN $OIDN_VERSION into $SAPIEN_DIR"
fi

echo "[setup_robotwin] environment installation complete"
echo "Next: bash scripts/setup/download_robotwin_assets.sh"
