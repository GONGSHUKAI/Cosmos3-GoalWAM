# RoboTwin integration

This directory makes the external RoboTwin dependency reproducible without
vendoring its assets or full source tree into this repository.

- `UPSTREAM_COMMIT` pins the official RoboTwin source revision.
- `patches/` contains the small tracked-source change for configurable episode
  counts and oracle goal-frame capture.
- `overlay/` contains the Cosmos socket client, policy variant configs, and data
  conversion helpers that are not part of upstream RoboTwin.
- `requirements-cu128.txt` records the Python packages verified with the current
  Blackwell/CUDA 12.8 evaluation stack.

Run `bash scripts/setup/setup_robotwin.sh` from the Cosmos repository root after
activating a fresh Python 3.10 conda environment. The script clones RoboTwin to
`external/RoboTwin` by default, applies the patch and overlay, installs the
requirements, builds Curobo v0.7.8, and installs the SAPIEN OIDN 2.3.3 fix.

Set `ROBOTWIN_DIR` to use a different clone destination. The setup script refuses
to replace a dirty or differently pinned clone.
