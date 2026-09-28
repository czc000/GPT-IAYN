# Directory migration in v0.1.1

The root CLI is unchanged. Models, generated references and `acceptance.py` bytes are unchanged. Direct script paths moved as follows. Install the source package with `python -m pip install --no-deps -e .` before direct script use.

| Previous path | Current path |
|---|---|
| `RESULTS.md` | `docs/history/CEM_RESULTS.md` |
| `RL_RESULTS.md` | `docs/history/PPO_RESULTS.md` |
| `acceptance.py` | `src/gpt_iayn_core/acceptance.py` |
| `audit_capsule_plane.py` | `experiments/legacy/audit_capsule_plane.py` |
| `audit_contacts.py` | `tools/validation/audit_contacts.py` |
| `audit_isaac.py` | `experiments/legacy/audit_isaac.py` |
| `audit_physx_geometry.py` | `tools/validation/audit_physx_geometry.py` |
| `calibrate_feedback_gain.py` | `experiments/legacy/calibrate_feedback_gain.py` |
| `cem_refine.py` | `scripts/cem_refine.py` |
| `cem_search.py` | `experiments/legacy/cem_search.py` |
| `check_rl.py` | `tools/validation/check_rl.py` |
| `check_rl_dynamics.py` | `tools/validation/check_rl_dynamics.py` |
| `check_transfer.py` | `experiments/legacy/check_transfer.py` |
| `evaluate.py` | `scripts/evaluate.py` (implementation in `src/gpt_iayn_core/`) |
| `evaluate_isaac.py` | `scripts/isaac/evaluate_isaac.py` |
| `evaluate_rl.py` | `scripts/evaluate_rl.py` (implementation in `src/gpt_iayn_core/`) |
| `export_policy.py` | `scripts/export_policy.py` |
| `feedback_actor.py` | `src/gpt_iayn_core/feedback_actor.py` |
| `isaac_convert.py` | `scripts/isaac/isaac_convert.py` |
| `isaac_runtime.sh` | `scripts/isaac_runtime.sh` |
| `monitor.py` | `experiments/legacy/monitor.py` |
| `motor.py` | `src/gpt_iayn_core/motor.py` |
| `package_final.py` | `scripts/package_final.py` |
| `package_rl.py` | `experiments/legacy/package_rl.py` |
| `policy_export.py` | `src/gpt_iayn_core/policy_export.py` |
| `portable.py` | `src/gpt_iayn_core/portable.py` |
| `prepare.py` | `scripts/prepare.py` |
| `prepare_isaac_model.py` | `scripts/isaac/prepare_isaac_model.py` |
| `prepare_rl.py` | `scripts/prepare_rl.py` |
| `render_final.py` | `scripts/render_final.py` |
| `rl_task.py` | `src/gpt_iayn_core/rl_task.py` |
| `runtime.sh` | `scripts/runtime.sh` |
| `scene.py` | `src/gpt_iayn_core/scene.py` |
| `seed_arm_negative.py` | `experiments/legacy/seed_arm_negative.py` |
| `seed_arm_sweep.py` | `experiments/legacy/seed_arm_sweep.py` |
| `task.py` | `src/gpt_iayn_core/task.py` |
| `tools/artifacts.py` | `src/gpt_iayn_core/artifacts.py` |
| `train.py` | `experiments/legacy/train.py` |
| `train_rl.py` | `scripts/train_rl.py` |
| `training_compat.py` | `src/gpt_iayn_core/training_compat.py` |

Shared `native_obs` moved from `check_rl.py` to `src/gpt_iayn_core/observations.py`; `plane_minimum` moved from `audit_capsule_plane.py` to `src/gpt_iayn_core/geometry.py`. Neither runtime API depends on diagnostic scripts.

`tests/legacy/` moved to `experiments/legacy/tests/`. The three result documents under `docs/history/` describe historical experiments. Current release limitations remain in `docs/RESULTS.md`.
