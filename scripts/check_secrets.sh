#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PATTERN='hf_[A-Za-z0-9]{20,}|wandb_v1_[A-Za-z0-9_-]{20,}|github_pat_[A-Za-z0-9_]{20,}|ghp_[A-Za-z0-9]{20,}'

matches="$({
  git grep -Il -E "$PATTERN" -- . 2>/dev/null || true
  rg -l --hidden -g '!.git/**' "$PATTERN" . 2>/dev/null || true
} | sort -u)"
if [[ -n "$matches" ]]; then
  echo "ERROR: possible credential found in tracked files:" >&2
  echo "$matches" >&2
  exit 1
fi

echo "[check_secrets] no known credential patterns found in tracked or trackable files"
