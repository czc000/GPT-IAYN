# PPO training

Prepare the environment and data using [INSTALL.md](INSTALL.md) and [DATA_GENERATION.md](DATA_GENERATION.md).

```bash
HIGHJUMP_GPU=0 ./runtime.sh gpt_iayn.py train --name ppo --num-envs 2048 --iterations 500
```

For a new accepted reference:

```bash
HIGHJUMP_GPU=0 ./runtime.sh gpt_iayn.py train --name ppo-new \
  --reference artifacts/new-reference/motion.npz --teacher artifacts/new-teacher
```

The public entry enables differential feedback and perturbation amplitude 0.025. Actor MLP: 152→256→256→128→29 with ELU. The same network is evaluated on actual and reference proprioception; subtract the two outputs so the mean correction is zero at the reference state. The demonstration target remains feedforward. This is a time/reference-conditioned controller, not a free-form high-jump planner.

Training actions produce `clip(reference_target + 0.25 * clip(action, -2, 2), joint_limits)`. The deployment package additionally interpolates the learned correction with gain 0.5. This is not a motor-power percentage. Actor inputs exclude actual world position/linear velocity/contact truth; reward and critic can use privileged simulation states. Gravity/gyro are ideal simulator observations during training, not a validated hardware estimator.

| Setting | Value |
|---|---|
| Environments / rollout length | 2048 / 32 |
| PPO iterations / environment steps | 500 / 32,768,000 |
| Learning rate / schedule | 1e-4 / fixed |
| Epochs / minibatches | 4 / 4 |
| Discount / GAE lambda / clip | 0.99 / 0.95 / 0.2 |
| Initial action std / entropy coefficient | 0.08 / 0 |
| Initial perturbations | joints ±0.025 rad; roll/pitch ±0.025 rad; horizontal velocity ±0.075 m/s |
| Seed | 42 |

Rewards: body position +5, orientation +3, joint position +3, joint velocity +0.5; residual squared magnitude −0.03, early head contact −2, bar contact −5. Tracking rewards use exponentials of errors. Training failures can terminate/reset; deployment evaluation is one continuous episode without mid-episode resets.

Outputs include checkpoints, TorchScript exports, scene, configuration and TensorBoard logs. `--resume` loads a trusted project checkpoint. A two-iteration smoke verifies the interface only and cannot establish learned jump quality. Full retraining and each chosen export need independent native acceptance.

This release keeps the historical compatibility runner because switching PPO implementations could change optimization and checkpoint semantics. It does not claim the environment's declared dependency constraints are satisfied; see the installation exception.
