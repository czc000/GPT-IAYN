# Data generation, from seed to physical teacher

Run commands from the repository root. Never interpret a generated qpos sequence as a successful physics rollout.

## A. Procedural seed

```bash
python gpt_iayn.py seed
```

`scripts/prepare.py` defines symmetric hip, knee, ankle, shoulder and elbow keyframes plus waist pitch. PCHIP interpolation produces 131 samples at 50 Hz. A manually specified backward root path and an initial ballistic segment provide a kinematic seed. Forward kinematics and finite differences supply the body/velocity fields. Output: `artifacts/reference/motion.npz` and its scene/summary. No mocap or competition video is used.

## B. Search in dynamics

```bash
HIGHJUMP_GPU=0 ./scripts/runtime.sh gpt_iayn.py search --name cem-reproduce
```

This reproduces the final refinement setup starting from the **bundled historical CEM15 candidate**, not from random initialization. It optimizes 56 phase-spline parameters over seven joint groups using a four-trial robust score, PD/DC actuation and native physical constraints. Source seed 719, 1024 parallel environments (256 candidates × four trials), up to 180 generations. `--population` counts environments and must be a multiple of four and at least 48. The historical CEM17 solution was selected early; later winners may differ.

To expose the independent starting point without a historical candidate:

```bash
HIGHJUMP_GPU=0 ./scripts/runtime.sh gpt_iayn.py search --from-zero --name cem-from-zero
```

The same constrained search starts from zero residual coefficients. This is an experimental search recipe, **not a verified guarantee of recovering the published jump from scratch**. Earlier exploratory scripts (`experiments/legacy/cem_search.py`, arm seeding) remain available. Old unaccepted candidates are never called successful demonstrations.

## C. Independently evaluate and promote a new candidate

```bash
python scripts/evaluate.py --run artifacts/cem-reproduce \
  --action-sequence artifacts/cem-reproduce/best.npz \
  --output artifacts/cem-reproduce-native --seeds 20
python scripts/package_final.py --run artifacts/cem-reproduce \
  --controller artifacts/cem-reproduce/best.npz \
  --evaluation artifacts/cem-reproduce-native --output artifacts/new-teacher
```

Promotion requires the expected 20 evaluations, nominal strict success, and hashes matching the exact candidate and acceptance code. It refuses an existing destination. A search failure must be investigated or reported; do not bypass this gate just to continue training. The number passing the remaining trials is reported, not required to be 20/20.

## D. Record a physical demonstration

The release includes a frozen, previously accepted CEM controller. Rebuild its PPO demonstration with:

```bash
python gpt_iayn.py teacher
```

For a newly promoted controller:

```bash
python scripts/prepare_rl.py --source artifacts/new-teacher --output artifacts/new-reference
```

At each 20 ms boundary, record the **actual simulated** qpos/qvel, body poses/velocities, projected gravity, gyro and commanded target. Advance eight 2.5 ms physics steps under joint PD/DC control. Initialize the root once; never copy the reference root into the running robot. Verify the recorded boundaries against the accepted nominal rollout before publishing the new reference.

The frozen-controller path provides an executable regeneration process, even when rerunning stochastic CEM does not rediscover the same coefficients. `release-data/` retains the immutable inputs and original checksums; regenerated outputs live under `artifacts/`.
