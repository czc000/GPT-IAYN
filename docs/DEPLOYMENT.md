# Export and independent execution

## Native MuJoCo

```bash
python scripts/export_policy.py --run artifacts/ppo --checkpoint model_499 --gain 0.5 --output artifacts/ppo-export
python scripts/evaluate_rl.py --run artifacts/ppo-export --policy artifacts/ppo-export/policy.torchscript --output artifacts/ppo-eval --video
```

The export contains absolute-target TorchScript/ONNX, local reference commands, evaluation initialization, scene and motor/physics contract. Gain 0.5 scales the learned target correction, not torque. The public exporter supports gain interpolation in [0,1]; the old wider calibration sweep remains in the historical script.

The online evaluator initializes qpos once, reads state and calls the policy at 50 Hz, applies PD/DC torque limits and advances eight MuJoCo steps per decision. Video is captured from those live states. `--reference` is an explicitly labeled kinematic diagnostic and always fails task acceptance; it must never be presented as a policy result.

## Optional Isaac Sim / PhysX

Install compatible Isaac Sim 4.5 / Isaac Lab separately and set `ISAAC_PYTHON`. From the native environment run `scripts/isaac/prepare_isaac_model.py` to flatten the frozen CEM MJCF; then use the Isaac environment:

```bash
./scripts/isaac_runtime.sh scripts/isaac/isaac_convert.py --headless
./scripts/isaac_runtime.sh scripts/isaac/evaluate_isaac.py --headless --device cpu \
  --run artifacts/final-rl-sim2sim --policy artifacts/final-rl-sim2sim/policy.torchscript \
  --output artifacts/physx-new --video --enable_cameras
```

Audit contacts and collision geometry before claiming success. The release includes the old audited episode, but packaging changes do not establish a new PhysX result. No simulator install, driver workaround, USD binary or automatic EULA acceptance is bundled.

## Real G1

No hardware deployment entry point is provided. Matching the joint-target interface is not sufficient to claim deployment readiness: ordering, estimator frames, command timing, gains, torque limits, latency and landing behavior require separate work and physical validation.
