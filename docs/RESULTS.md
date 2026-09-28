# Results and boundaries

The frozen final release is `release-data/final-rl-sim2sim/`, with feedback gain **0.5**. It retains the original TorchScript/ONNX and checkpoint bytes. Native online evidence and independent holdout metrics are included beside the model; the public demo recomputes a rollout rather than playing those stored states.

| Experiment | Recorded result |
|---|---|
| Native MuJoCo nominal, final frozen policy | Accepted online rollout |
| Calibrated gain 0.5, small perturbations | 19/50; fixed-target comparison 13/50 |
| Calibrated gain 0.5, larger perturbations | 3/50; fixed-target comparison 4/50 |
| PhysX CPU nominal | One calibrated accepted episode; minimum gap 0.0018945 m |
| Cross-engine perturbation robustness | Not evaluated |
| Real robot | Not tested |

The original unscaled gain-1 model had other holdout statistics on different seeds. Do not combine those with the final gain-0.5 results. The older CEM controller's 19/20 also uses a different controller and much smaller Gaussian perturbations, not the PPO holdout protocol.

MuJoCo-Warp training and native MuJoCo evaluation share the same physics engine with different backends. The PhysX episode is the actual cross-engine experiment; it required selecting a constant feedback gain from 0.5/1/2/4. It is not zero-shot sim2sim or validated sim2real.

Strict acceptance includes physical airborne motion, whole-body passage above/between the bar and uprights, no bar contact, allowed mat-contact order, back-oriented support/finish, and joint-limit/speed tolerances. The bar is fixed but collidable; the mat uses soft contact, not a deformable foam model. A small nominal geometric margin is not a robustness guarantee.

The actor depends on a fixed timed reference. Its state inputs are joint sensing and IMU-like quantities; simulator measurements are idealized. Hardware latency, state estimation, calibration, motor behavior, landing impact safety and recovery remain unvalidated.

See `../docs/history/PPO_RESULTS.md` for the original detailed PPO analysis and `../release-data/final-rl-sim2sim/validation/` for actual holdout results. New packaging checks are listed separately in `RELEASE_VERIFICATION.md`; they must not be confused with new policy training.
