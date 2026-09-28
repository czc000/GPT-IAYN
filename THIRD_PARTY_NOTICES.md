# Attribution and release scope

The MIT license covers project-specific code and documentation. Project-generated motion arrays, controller coefficients and trained weights in `release-data/` are included under the same grant to the extent the contributors can grant those rights. It does not replace upstream licenses or imply ownership of robot designs, libraries or simulator software.

| Component | Origin / role | Treatment |
|---|---|---|
| mjlab 1.2.0 | https://github.com/mujocolab/mjlab — environment API, tracking task, G1 configuration | Apache-2.0; license in `third_party/mjlab-LICENSE.txt`. mjlab's inherited third-party portions retain their notices. The project's `src/gpt_iayn_core/training_compat.py` preserves its CUDA-graph eligibility rule with the current Warp API without editing the installed package. |
| MuJoCo / MuJoCo-Warp | https://github.com/google-deepmind/mujoco ; https://github.com/google-deepmind/mujoco_warp | Separately installed dependencies; retain their upstream licenses. |
| RSL-RL compatibility implementation | https://github.com/ccrpRepo/AMP_mjlab at `2d6b0d1b1c00681accfe4bd9f244fad1eaa51940`, `rsl_rl/` | Dependency fetched from upstream, not copied into this repository. Core files identify RSL-RL, Isaac Lab, Legged Lab and TienKung-Lab authors and BSD-3-Clause terms. This experiment uses PPO, not AMP rewards. |
| RSL-RL | https://github.com/leggedrobotics/rsl_rl | PPO foundation, BSD-3-Clause. |
| BeyondMimic | https://github.com/HybridRobotics/whole_body_tracking | Motion-tracking foundation inherited through mjlab; no BeyondMimic motion dataset or checkpoint is included. |
| Unitree G1 model | Robot configuration and meshes supplied in mjlab 1.2.0 | Scene XML derives from the upstream model and retains the mjlab license. Mesh binaries are excluded; the asset helper retrieves matching files from the pinned upstream package. Upstream asset rights still apply. |
| Isaac Sim 4.5 / PhysX | NVIDIA, optional cross-engine experiment | Separately installed; no simulator, driver libraries, installation credentials or EULA acceptance script is redistributed. |
| PyTorch, NumPy, SciPy, Pillow, ImageIO, ONNX | Runtime / export dependencies | Installed separately under their own licenses. |

No competition footage, GVHMR weights, SMPL-X models, GMR assets, human motion dataset, run-up branch or perception branch is part of this standing-jump release. Included videos are recordings of this project's robot simulations and depict the upstream G1 model; they are not evidence of real hardware operation.

The repository name describes an experiment in GPT-assisted development. It does not claim ownership of these foundations, affiliation with OpenAI or Unitree, or that GPT replaces the physics engine or inference policy.
