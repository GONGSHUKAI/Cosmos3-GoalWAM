#!/usr/bin/env bash

# Shared helpers for the tracked cluster launchers. This file intentionally does
# not enable `set -e`/`set -u`; the calling script owns its shell policy.

cosmos_load_local_env() {
  local env_file="${COSMOS_LOCAL_ENV:-$COSMOS_ROOT/config/local.env}"
  if [[ -f "$env_file" ]]; then
    # Export plain assignments as well as explicit `export` lines.
    set -a
    # shellcheck disable=SC1090
    source "$env_file"
    set +a
  fi

  export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
  export UV_CACHE_DIR="${UV_CACHE_DIR:-$HOME/.cache/uv}"
  export LD_LIBRARY_PATH=
}

cosmos_activate_venv() {
  local activate="$COSMOS_ROOT/.venv/bin/activate"
  if [[ ! -f "$activate" ]]; then
    echo "ERROR: Cosmos virtual environment not found: $activate" >&2
    echo "Run: uv sync --all-extras --group=cu130-train" >&2
    return 1
  fi
  # shellcheck disable=SC1090
  source "$activate"
  export LD_LIBRARY_PATH=
}

cosmos_require_dir() {
  local name="$1" path="$2"
  if [[ -z "$path" || ! -d "$path" ]]; then
    echo "ERROR: $name directory not found: ${path:-<unset>}" >&2
    return 1
  fi
}

cosmos_require_file() {
  local name="$1" path="$2"
  if [[ -z "$path" || ! -f "$path" ]]; then
    echo "ERROR: $name file not found: ${path:-<unset>}" >&2
    return 1
  fi
}

cosmos_prepare_training_env() {
  : "${BASE_CHECKPOINT_PATH:?Set BASE_CHECKPOINT_PATH in config/local.env or the shell environment}"
  : "${WAN_VAE_PATH:?Set WAN_VAE_PATH in config/local.env or the shell environment}"
  cosmos_require_dir BASE_CHECKPOINT_PATH "$BASE_CHECKPOINT_PATH"
  cosmos_require_file WAN_VAE_PATH "$WAN_VAE_PATH"
}
