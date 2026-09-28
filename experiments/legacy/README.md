# Historical exploration — outside the supported quick start

These scripts preserve the investigation that led to the frozen standing PPO release. They can depend on earlier run names and fixtures not distributed with the release; keeping the source does not claim that every historical command is immediately reproducible.

| Files | Historical role |
|---|---|
| `train.py`, `monitor.py` | Earlier general tracking-policy experiment and monitor |
| `cem_search.py` | Earlier CEM parameterization; current search is `scripts/cem_refine.py` |
| `seed_arm_sweep.py`, `seed_arm_negative.py` | Arm initialization sweeps |
| `calibrate_feedback_gain.py`, `package_rl.py` | Original experiment-specific calibration/packaging; current export is `scripts/export_policy.py` |
| `check_transfer.py` | Earlier observation/backend diagnostic |
| `audit_isaac.py`, `audit_capsule_plane.py` | Historical PhysX diagnostics tied to old output directories |
| `tests/` | Tests requiring absent old smoke/CEM fixtures |

Core code does not import this directory. The data-generation chain required for the published result remains supported in `scripts/` and is documented in [DATA_GENERATION.md](../../docs/DATA_GENERATION.md).
