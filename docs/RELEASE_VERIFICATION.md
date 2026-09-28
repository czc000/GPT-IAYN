# Release verification — 2026-09-28

This report validates packaging and execution of the preserved standing-jump experiment. It is not a new full-training result.

| Check | Observed result |
|---|---|
| Unit/contract/acceptance suite | 18 tests passed |
| Release manifest | All 40 immutable files verified |
| Frozen PPO online native rollout | Strict 0.8 m acceptance passed, zero bar contacts; minimum clearance 0.0337203 m |
| Frozen rollout compared with original | All qpos, statistics, observations and targets exactly equal (maximum difference 0) |
| Physical teacher regenerated from frozen CEM | Every numeric array exactly equal; names equal |
| Procedural seed regenerated from code | Every numeric array exactly equal; names equal |
| Relocated directory with spaces | Complete native rollout passed with mjlab, rsl_rl and warp imports explicitly blocked; exact trajectory equality |
| Asset retrieval without installed mjlab | Pinned official wheel downloaded using site-packages-disabled Python; all 105 mesh references hash-verified |
| PPO training smoke | 64 environments × 32 steps × 2 iterations = 4096 steps; checkpoints and TorchScript generated |
| CEM search smoke | One generation, 64 environments / four trials; candidate saved, **0/4 accepted**; interface check only |
| New export | Smoke checkpoint exported to TorchScript and ONNX; ONNX checker passed |
| Export execution parity | 131 saved observations; TorchScript gain wrapper error 0; ONNX reference evaluator max error 3.06e-10 rad |
| CEM promotion | Existing historical 20-trial evidence accepted and new package relative reference resolved; no new candidate acceptance claimed |
| Python / shell syntax | Python compile check and launcher shell syntax passed |
| Original standing checkout | Remained at 8ce7be6 with only its pre-existing untracked human-imitation directory |

Machine-readable array comparisons: [native and teacher](verification/native-and-teacher-parity.json), [relocation and seed](verification/relocation-and-seed-parity.json), [new export](verification/export-parity.json).

## Corrections discovered during packaging

- The installed MuJoCo removed the old MULTICCD enable enumeration. The portable loader now preserves the historical false/default behavior; exact native trajectory parity confirms that behavior is unchanged.
- An initial CEM smoke request of 32 environments supplied only eight candidates for a 12-elite search and failed. The entry now rejects such input before environment creation. The 64-environment check passed without changing the historical valid-budget algorithm.
- Promoted CEM packages now store `reference.npz` relative to the package, avoiding duplicated relative directory prefixes.

## What was not rerun or established

The GPU checks used the existing historical training environment, including its documented dependency metadata exceptions. The fresh training-environment installer has **not** been validated from an empty environment. The native import isolation and asset-download checks do not establish a full clean-install result.

No new 500-update PPO training, convergence of a from-zero CEM run, 50-trial holdout, PhysX rollout or real-G1 execution was performed for packaging. Frozen model files and historical results remain the original ones. The optional PhysX path was updated for relocatable contracts and syntax-checked only. New smoke weights are not the published successful policy and are excluded from release data.
