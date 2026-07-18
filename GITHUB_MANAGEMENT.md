# GitHub management checklist

The repository keeps the NVIDIA upstream remote as `origin` and the GoalWAM
repository as `myfork`:

```bash
git remote -v
# origin  https://github.com/NVIDIA/cosmos-framework.git
# myfork  git@github.com:GONGSHUKAI/Cosmos3-GoalWAM.git
```

## Before every commit

Never use `git add .` in this repository. Training launchers, local credentials,
large outputs, and external simulator files may coexist in the worktree.

1. Review tracked and untracked changes:

   ```bash
   git status --short
   git diff --check
   git diff --stat
   ```

2. Confirm the local credential file remains ignored:

   ```bash
   git check-ignore -v config/local.env
   ```

3. Scan the files that could be committed:

   ```bash
   git grep -InE 'hf_[A-Za-z0-9]{20,}|wandb_v1_[A-Za-z0-9_-]{20,}|github_pat_|ghp_' -- .
   rg -n --hidden -g '!config/local.env' -g '!external/**' \
     'hf_[A-Za-z0-9]{20,}|wandb_v1_[A-Za-z0-9_-]{20,}|github_pat_|ghp_'
   ```

4. Stage explicit groups only, for example:

   ```bash
   git add .gitignore README.md GITHUB_MANAGEMENT.md
   git add config/local.env.example
   git add cosmos_framework examples
   git add scripts/_common.sh scripts/train scripts/eval scripts/download scripts/preprocess scripts/setup
   git add integrations/robotwin
   ```

5. Inspect exactly what will be committed:

   ```bash
   git diff --cached --stat
   git diff --cached
   ```

## Validation

```bash
find scripts -name '*.sh' -not -path 'scripts/deprecated/*' -print0 \
  | xargs -0 -n1 bash -n
uv run ruff check cosmos_framework scripts integrations
```

For evaluation changes, also run the 50-task manager in dry-run mode and a
synthetic policy client smoke test before pushing.

## Commit and push

Prefer small, reviewable commits:

```bash
git commit -m "security: externalize cluster credentials and paths"
git commit -m "robotwin: add reproducible GoalWAM integration"
git commit -m "docs: document GoalWAM training and evaluation"
git push myfork main
```

For experimental work that should not immediately update `main`:

```bash
git switch -c goalwam-experiment
git push -u myfork goalwam-experiment
```

## Synchronize NVIDIA upstream

Do this only with a clean worktree:

```bash
git fetch origin
git merge origin/main
git push myfork main
```

Resolve and test any conflicts in the RoboTwin policy/data paths before pushing.
