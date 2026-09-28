# Installation and dependency provenance

## Native evaluation

Use Python 3.11 and a fresh virtual environment. Install `requirements/eval.txt`, then run `python gpt_iayn.py restore`. The native evaluator does not import mjlab, RSL-RL or Warp. Without video it needs neither a graphics context nor an available CUDA device. With video, choose a working MuJoCo renderer (`MUJOCO_GL=egl` on a headless Linux host, or the appropriate local backend).

The release-data manifest pins the mjlab 1.2.0 wheel hash and each G1 mesh hash. The mesh helper can use a matching existing package, or retrieve the wheel directly from PyPI and extract only the expected asset bytes. It does not install mjlab or change the active environment. A mismatching existing mjlab asset is rejected rather than silently used.

## Historical training environment

Run `bash tools/setup_training.sh .venv-train` from the project root. The script refuses to overwrite an existing directory. It creates a separate environment; never run its package commands manually against a shared environment.

The original successful run used:

| Component | Recorded version/source |
|---|---|
| Python | 3.11 |
| Torch | 2.6.0+cu124 |
| mjlab | 1.2.0 |
| MuJoCo / MuJoCo-Warp | 3.8.1 / 3.8.1 |
| Warp | 1.13.0 |
| NumPy / SciPy | 2.4.4 / 1.17.1 |
| RSL-RL | `ccrpRepo/AMP_mjlab` commit `2d6b0d1b1c00681accfe4bd9f244fad1eaa51940`, `rsl_rl/`, package label 2.3.1 |

**Known dependency metadata exception:** mjlab 1.2.0 declares Torch >=2.7 and RSL-RL 5.0.1, whereas the recorded environment used the table above. The pinned fork provides the actor/critic-config-to-legacy-runner compatibility code. PyPI's generic `rsl-rl-lib==2.3.1` is not a replacement for that source revision. Setup intentionally installs mjlab and this runner with `--no-deps` after support packages; `pip check` will report the two known conflicts. This is a documented historical stack, not a clean dependency-resolution claim.

The release also includes `training_compat.py`: mjlab 1.2.0's CUDA-graph capability check accesses `warp.context`, which Warp 1.13 removed. The project uses Warp's public driver query with the same driver/memory-pool condition, within the training process only. Installed dependency files are never edited. This compatibility behavior was present as a local patch in the historical environment and is now explicit in the repository.

Auxiliary support versions are partly constrained in `requirements/train-support.txt`; a full installed-version record is included in `docs/verification/fresh-environment-packages.txt`. The setup script was validated in a new isolated Python 3.11.15 environment on Linux, followed by 20 tests, native video, data regeneration, GPU short training and export. See [RELEASE_VERIFICATION.md](RELEASE_VERIFICATION.md). Other operating systems and driver combinations still require validation.

Choose an idle GPU with `HIGHJUMP_GPU=0 ./runtime.sh ...`. The launcher uses the active Python (or `GPT_IAYN_PYTHON`) and does not set hard-coded driver paths. The original host needed a process-local driver-library workaround; that is not a portable dependency and is not included. Configure working NVIDIA drivers on the destination host.

## Optional PhysX path

Install Isaac Sim 4.5 and its compatible Isaac Lab separately, following NVIDIA's own setup and terms. Set `ISAAC_PYTHON` to that environment's Python. The source includes MJCF flattening, USD conversion and online evaluation. See [DEPLOYMENT.md](DEPLOYMENT.md). No Isaac installation or license acceptance is bundled.
