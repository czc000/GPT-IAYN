# GPT-IAYN Implementation Plan

**Goal:** A self-contained, reviewable standing-jump PPO release with reproducible generated data and truthful evidence.

**Architecture:** Preserve the historical experiment modules. Add a portable native runtime, artifact restoration and hash verification, an explicit dependency setup, and a small CLI. Keep release data immutable and separate from generated experiments.

**Tech Stack:** Python 3.11, NumPy, SciPy, TorchScript/ONNX, native MuJoCo, optional mjlab/MuJoCo-Warp and pinned RSL-RL fork.

1. Snapshot the standing source into an independent repository; write scope and provenance.
2. Replace machine-specific launchers and path assumptions. Lazy-load training dependencies for native evaluation. Add meaningful portable-runtime tests.
3. Select frozen PPO/CEM/reference artifacts; retain numerical arrays and model bytes, use relative metadata paths, and hash every release file. Resolve third-party meshes from upstream separately.
4. Add CLI commands for restore/demo/seed/teacher/search/train/export; make experiment run names and inputs explicit. Preserve failed-search rejection.
5. Write English and Chinese README, generation/training/deployment guides, data schema, GPT contribution statement, results, MIT license and third-party notices.
6. Verify native parity, reference regeneration, artifact integrity, relocation, and a bounded GPU smoke without touching ongoing work. No full retraining is needed for this packaging task.
7. Commit the independent repository, make a release archive and concise handoff, and record workspace changes.
