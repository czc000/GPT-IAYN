# Data and model contracts

All NPZ files use NumPy arrays; quaternions are scalar-first `wxyz`. Units: meters, seconds, radians. Source joint/body orders are explicitly stored and must not be inferred from actuator ordering.

| File family | Main fields | Meaning |
|---|---|---|
| Procedural reference | time, fps, qpos, joint_pos/vel, body_pos_w/quat_w/lin_vel_w/ang_vel_w, joint_names, body_names | 131-frame kinematic seed; no success claim |
| CEM candidate | params, actions, qpos, score, generation | 56 coefficients; sampled residual commands and simulated candidate states. Search score is not strict acceptance. |
| PPO physical reference | time, qpos, qvel, joint_pos/vel, body poses/velocities, target, gravity, gyro, names | Control-boundary states of the accepted controller under native dynamics |
| Deployment commands | target, joint_pos, joint_vel, time | The local reference command sequence, not actual robot-state measurements |
| Native evaluation | qpos, stats, observations, targets | Actual integrated rollout and online policy inputs/outputs |

G1 qpos is N×36 (root position 3 + root quaternion 4 + joints 29), qvel is N×35. The policy has 152 float32 inputs and 29 absolute target-angle outputs. Exported TorchScript/ONNX carries reference gravity/gyro buffers for the differential actor. The runtime also needs `commands.npz`; evaluation additionally uses `evaluation-reference.npz` for initialization and audit context.

Observation slices: 0:29 target, 29:58 reference joints, 58:87 reference velocity×0.05, 87:116 actual-reference joint error, 116:145 actual velocity×0.05, 145:148 projected gravity, 148:151 gyro×0.1, 151 phase `k/130`.

`contract.json` declares physics, motor parameters, order and reference path. Public frozen references are relative to the contract directory; newly generated local training runs can record absolute paths until exported. `export_policy.py` writes a portable relative contract.

`release-data/PROVENANCE.json` binds files to the original standing commit; `SHA256SUMS` verifies release bytes. JSON paths were normalized for publication; numerical reference arrays and original frozen policy weights were not changed. Mesh files are excluded from Git and enumerated separately in `ASSETS.json`.
