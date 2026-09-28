# Supported execution scripts

Use the repository's editable installation (`python -m pip install --no-deps -e .`) in the appropriate dependency environment. The root `gpt_iayn.py` is the simplest entry point. Launchers change to the repository root and add `src/` to the module path.

| Script | Purpose |
|---|---|
| `prepare.py` | Generate the procedural seed |
| `cem_refine.py` | Search a physical joint-target controller |
| `evaluate.py` | Independently evaluate a CEM candidate |
| `package_final.py` | Promote hash-matched, accepted CEM evidence |
| `prepare_rl.py` | Record the physical teacher for PPO |
| `train_rl.py` | Train the reference-conditioned feedback policy |
| `export_policy.py` | Export a selected run to TorchScript / ONNX |
| `evaluate_rl.py` | Run online native MuJoCo policy inference |
| `render_final.py` | Render the saved historical CEM rollout, not new online inference |
| `isaac/prepare_isaac_model.py` | Flatten MJCF for the optional PhysX path |
| `isaac/isaac_convert.py` | Convert that model with Isaac Sim |
| `isaac/evaluate_isaac.py` | Execute the frozen policy in PhysX |

The online evaluators share implementation in `src/gpt_iayn_core/`; their entries here are deliberately thin. See [data generation](../docs/DATA_GENERATION.md), [training](../docs/TRAINING.md) and [deployment](../docs/DEPLOYMENT.md) for complete commands and acceptance criteria.
