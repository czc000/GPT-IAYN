# Release verification — 2026-09-28

This report validates packaging and execution of the preserved standing-jump experiment. It is not a new full-training result.

| Check | Observed result |
|---|---|
| Unit/contract/acceptance suite | 20 tests passed, including two Warp compatibility regressions |
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

## Fresh-environment publication check

On 2026-09-28, `tools/setup_training.sh` installed into a new Python 3.11.15 virtual environment with `include-system-site-packages = false`. No existing training packages or edited site-packages were inherited. The test used an available RTX 4090 and the host's existing process-local NVIDIA driver-library setup; no system environment was changed.

The first GPU attempt exposed a historical local mjlab patch: official mjlab 1.2.0 calls the removed `warp.context` API. A regression test reproduced the failure. `src/gpt_iayn_core/training_compat.py` now performs the same CUDA-graph driver/memory-pool eligibility check using Warp's public driver API, activated by the task configuration within the process. A source comparison found two other historical mjlab changes (observation history layout and velocity GUI ranges), neither needed by this experiment; they were not copied. The installed RSL-RL Python source matched the pinned fork.

With that explicit compatibility module, the clean environment completed:

- All 20 tests, artifact restoration and online MuJoCo recording (131 video frames; full video decode passed).
- Procedural seed and physical teacher regeneration, with exact array equality to the immutable references.
- Frozen PPO online rollout with exact qpos/statistics/observation/target equality to the original, strict nominal success and 0.0337203 m clearance.
- 64-environment / two-update PPO training (4096 steps), TorchScript and ONNX export, and one CEM generation (64 environments; 0/4 accepted, not a new successful jump).
- Export comparison across 131 observations: TorchScript error 0; ONNX reference evaluator max error 2.62e-10 rad.

`pip check` reports exactly the two documented mjlab metadata exceptions (Torch and RSL-RL), with no additional conflicts. This is a verified runnable historical stack, not a claim of clean dependency metadata. Evidence: [stage results](verification/fresh-environment-stages.json), [array/export parity](verification/fresh-environment-parity.json), [installed versions](verification/fresh-environment-packages.txt), [metadata exceptions](verification/dependency-metadata-exceptions.txt).

The temporary environment, source copy, caches and smoke artifacts are disposable. Only the concise evidence above is part of the release.

## What was not rerun or established

No new 500-update PPO training, convergence of a from-zero CEM run, 50-trial holdout, PhysX rollout or real-G1 execution was performed for packaging. Frozen model files and historical results remain the original ones. The optional PhysX path was updated for relocatable contracts and syntax-checked only. New smoke weights are not the published successful policy and are excluded from release data.


## v0.1.1 directory cleanup verification

The root now contains one Python launcher, with importable implementation in `src/gpt_iayn_core/`, supported scripts in `scripts/`, validators in `tools/validation/` and exploratory scripts in `experiments/legacy/`. Shared native observations and capsule-plane geometry no longer come from diagnostic modules. All 40 immutable release files and the acceptance source bytes are unchanged.

This verification used a disposable virtual environment with **shared existing pinned dependencies** and an isolated editable install of the reorganized project; it was not another fresh dependency download. The previous clean dependency-install evidence above remains applicable to the unchanged requirement pins.

- 21 tests passed, including import isolation from an unrelated working directory with training packages blocked.
- Editable installation and the installed `gpt-iayn` CLI worked; six direct entry points accepted `--help` from outside the checkout.
- Online native inference passed strict nominal acceptance; the 131-frame video decoded fully. qpos, statistics, observations and joint targets exactly matched the original frozen rollout.
- Regenerated procedural seed and physical teacher arrays exactly matched the release data.
- Short PPO training (64 environments, two iterations, 4096 steps), one CEM generation and TorchScript/ONNX export completed. CEM smoke is an interface check, not newly accepted jump evidence.
- Export parity across 131 observations: TorchScript error 0, ONNX reference evaluator max error 2.33e-10 rad.
- The promotion command accepted historical hash-matched evidence and created a valid relative-reference package. An output-path regression discovered during restructuring was corrected and this path rerun successfully.
- Python/shell syntax and local Markdown links checked. No full PPO retraining, PhysX rerun or hardware evaluation was performed.

Evidence: [layout stages](verification/layout-stages.json), [exact data and export comparisons](verification/layout-parity.json). Temporary verification environments and generated smoke outputs are removed after publication.
