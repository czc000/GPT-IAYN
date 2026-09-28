> Historical experiment report. For the public release workflow and current limits, see [README.md](../../README.md) and [docs/RESULTS.md](../RESULTS.md).

# G1 0.80 m back-first high jump — verified result

Frozen CEM controller with 56 learned spline parameters and joint PD feedback. Training: MuJoCo-Warp. Independent evaluation: native CPU MuJoCo. Standing two-foot backward takeoff, no run-up. This is not a neural PPO result or Isaac-to-MuJoCo transfer.

- Native small-initial-pose-perturbation trials: **19/20** passed; seed 0 is nominal, seeds 1–19 use Gaussian joint-angle noise with sigma 0.005 rad.
- Seed 13 fails clearance and touches the bar; it also fails the head-before-back-support check. This failure is included in the denominator and its trajectory is retained.
- Nominal minimum collision-geometry clearance above the rod surface: **0.0337 m**; robot–bar contacts: **0**.
- Nominal maximum pelvis height: 1.1877 m; continuous unsupported flight: 0.5850 s.
- First mat contact at 1.4700 s: robot/right_hand_collision. Head-first: False. First torso/pelvis mat support: 1.5350 s, upward-facing component 0.9968; any head contact before that support: False. Torso upward-facing component at landing/final: 0.9676/0.9438.
- Worst joint-limit solver violation across all trials: 0.009499 rad (acceptance tolerance 0.02); worst speed/rated-speed ratio: 0.941209 (threshold 1.05).
- All dynamics use gravity, original robot mass/collisions and a conservative linear DC torque-speed envelope. Grounded zero-velocity start; no root actuation, support forces, teleports, or mid-episode resets.

Bar center height 0.800 m, radius 0.012 m, x=-0.700 m; mat top 0.180 m. The bar is fixed and collidable; every touch fails acceptance. The soft-contact pad is a simulation approximation. No real-robot impact safety or broad robustness claim is made.

Contact-event timestamps above use step-end labels; contact geometry corresponds to the preceding 2.5 ms substep. Independent reimplementation reproduced the nominal trajectory within 4.04e-8 and confirmed the same contact order. A separate visual-mesh slice check found about 4.15 cm minimum clearance on the saved 50 Hz frames.

## Artifacts and reproduction

Remote project: `<project-root>`. Final package: `<project-root>/artifacts/final`. Per-trial metrics and dynamic trajectories: `artifacts/final/native-evaluation/`. Controller hash: `b8c585ad8c51e833bdca6b949d1728f642977ce48f2bde98e3dab79fcbc24a4c`.

```bash
cd <project-root>
./scripts/runtime.sh scripts/evaluate.py --run artifacts/final --action-sequence artifacts/final/frozen-controller.npz --output artifacts/reproduce-final --seeds 20 --video
./scripts/runtime.sh scripts/render_final.py
```

The multi-view MP4 is a render of the independently simulated native trajectory, including normal speed and labeled slow motion; it is not the kinematic training reference. Earlier rejected results remain separately labeled diagnostics in README.md.
