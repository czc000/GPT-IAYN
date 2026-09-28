> Historical notes, not the release installation guide. See ../../README.md.

# 最新结果：PPO 状态反馈 / MuJoCo-Warp → PhysX

2026-09-24：已新增参考条件 PPO 策略、29 关节目标模型及 PhysX 在线推理视频。最终交付见 [docs/history/PPO_RESULTS.md](PPO_RESULTS.md) 和 `artifacts/final-rl-sim2sim/`。跨引擎前标定了反馈增益；余量与鲁棒性仍有限，未验证实机。

以下保留旧 CEM 阶段记录，其成功率、方法和回放视频不代表新的 PPO 结果。

---

# Unitree G1: 0.80 m back-first high jump

A self-contained simulation experiment in `<project-root>`. The learned controller performs a **standing, two-foot backward jump**, arches over a collidable 0.80 m bar and lands on a soft-contact pad. This is not a running, single-leg athletics technique. The final frozen controller passed 19/20 independent native small-perturbation trials, including the nominal episode. The nominal clip clears the rod by 3.37 cm, has zero bar contacts and lands supine with arm cushioning. See `docs/history/CEM_RESULTS.md` and `artifacts/final/`.

## Method and deliverables

The successful training path uses the cross-entropy method (CEM) to optimize phase-spline coefficients (40 initially, 56 with elbow and shoulder-spread refinement) around a procedurally generated motion seed. These produce 29 joint targets at 50 Hz. Joint PD feedback and a linear DC torque-speed envelope run at 400 Hz. The controller has phase and joint feedback, but no learned balance network or external root actuator. The initial PPO experiments did not produce the final controller.

Training runs in GPU MuJoCo-Warp through the installed mjlab package. The frozen controller is evaluated independently in native CPU MuJoCo with matching integration/solver settings and a separately implemented NumPy motor controller. **This is Warp-to-native MuJoCo sim2sim, not Isaac-to-MuJoCo.** Evaluation starts grounded with zero velocity, uses one continuous dynamic episode, and never resets/teleports the root or applies external support forces. A multi-view video replays the resulting dynamic trajectory; the procedural reference is never presented as a trained result.

Robot source: installed mjlab Unitree G1, 29 actuators, original masses/armatures and collision shapes. Motor assumptions use the installed effort/speed specifications and mjlab's DC envelope. Joint targets are clipped inside the physical joint ranges; joint-limit contacts use a stiff solver configuration. These are simulation assumptions, not calibrated hardware validation.

Scene: fixed capsule bar at z=0.80 m, radius 0.012 m, two stands, landing mat with top at z=0.18 m in the strict DC scene. The rod is fixed but collidable: **any robot-bar contact fails**. Soft mat contact approximates cushioning, not a deformable foam material model.

## Reproduction

```bash
cd <project-root>
./scripts/runtime.sh scripts/prepare.py
HIGHJUMP_GPU=7 ./scripts/runtime.sh -m unittest discover -s tests -v
HIGHJUMP_GPU=6 ./scripts/runtime.sh scripts/cem_refine.py \
  --init artifacts/cem15-back-support/candidate-b.npz \
  --name reproduce-cem17 --generations 180 --population 1024 \
  --trials 4 --dc --std .1 --min-std .02 \
  --landing-weight 3000 --orientation-weight 5 --wide-arms
```

The accepted CEM17 controller was frozen at generation 11. Training was stopped after native acceptance; rerunning the longer command may produce a different later winner. The full staged optimization history, seeds, frozen candidate arrays, scene XML/assets, exact joint ordering, PD/motor parameters and physics contract are retained in `artifacts/`. The final package includes the reference, learned spline coefficients/actions, native trajectories, per-trial metrics, scene and MP4. Final evaluation and render commands are recorded in `docs/history/CEM_RESULTS.md`.

The original experiment used a host-specific driver library workaround. It is not part of this release; use a working NVIDIA driver and select an available GPU with `HIGHJUMP_GPU`.

## Acceptance and limits

Native evaluation checks every 2.5 ms substep: both feet initially supported, genuine airborne takeoff, conservative collision-geometry bounds above the bar and between uprights, whole-body passage, upward-facing torso at crossing, zero bar contacts, and mat contact only after whole-body clearance. The above-bar geometry test ends after the first simultaneous whole-body far-side crossing; actual bar contacts remain forbidden for the entire episode. Subsequent limb return below the bar is recorded, and the final pelvis/torso must remain on the landing side. First mat contact must exclude the head; head contact must also not precede torso/pelvis mat support (which must be supine, upward-facing component >=0.7); the torso forward-axis upward component must be >=0.7 at that contact and >=0.5 at the finish. Maximum joint range violation must be <=0.02 rad (explicit numerical solver tolerance); speed/rated-speed must be <=1.05. Flight time ends on any environmental support, including the mat.

Seed 0 is nominal. Other evaluation seeds perturb initial joint angles with independent Gaussian noise of standard deviation 0.005 rad. Success rates refer only to this small-perturbation test, not broad robustness across friction, mass, timing or terrain changes. Landing impact safety, real hardware deployment, automatic recovery, run-up and single-leg takeoff are not established by this experiment.

## Rejected diagnostics

- `reference-preview/`: kinematic animation only. `zero-baseline/`: unsuccessful untrained dynamic baseline.
- `run01`, `cem01`: invalid training scene origins (grid-spaced robots versus fixed obstacles); corrected to zero origins in independent Warp worlds.
- `cem03`, `diagnostic-unconstrained-cem03`: geometric clearance under constant-torque PD, rejected because motor speed and joint-stop violations were excessive. Never use their clips as final evidence.
- `cem07-strict-substep` passed geometric/motor criteria in 18/20 native trials but landed head-first and finished prone. It is rejected as a complete back-landing result; later acceptance explicitly rejects this failure.
- Early DC candidates with mat contact before complete clearance or other failed acceptance remain diagnostics. The final evaluator reports such failures rather than suppressing them.

Foundations: [mjlab](https://github.com/mujocolab/mjlab), [BeyondMimic](https://github.com/HybridRobotics/whole_body_tracking), and the installed RSL-RL adapter (PPO diagnostics only). The seed is generated by `scripts/prepare.py`; no external motion-capture file was copied into this project. Upstream repositories remain unchanged.
