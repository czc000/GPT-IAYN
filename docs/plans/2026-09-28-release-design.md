# GPT-IAYN release design

Scope approved in the conversation: independently organize the standing-jump PPO project, including data generation, CEM search, physical demonstration capture, PPO training, model export and evaluation. Project name: **GPT-IAYN — GPT Is All You Need**. Prepare a reviewable repository; publishing to a hosting service is not part of this preparation.

Use a clean snapshot of standing-branch commit `8ce7be6876dfcc1d2d31e83210f02f4ae2bac047`. Do not modify the existing project or any running-jump worktree. Keep the original experiment scripts recognizable; add portable paths and a small command-line front door. Include own generated references, frozen controllers and trained weights, with hashes and explicit experimental limits. Third-party mesh files are resolved separately from a pinned upstream package and verified against hashes.

Two entry paths: CPU native MuJoCo evaluation without mjlab; historical training using the exact pinned mjlab/RSL-RL combination. Training dependency metadata conflicts must be documented, not hidden. Preserve the recorded policy, physics and acceptance. No new 500-update training or claimed improvement is required for packaging.

GPT attribution describes project-specific implementation and experiment work under user direction. Upstream physics, robot assets and RL implementations remain credited to their authors. No claim of training GPT itself, GPT runtime control, original PPO or successful real-robot deployment.

Verification: source/unit tests; exact frozen rollout parity against the original saved observations, targets and qpos; physical teacher reconstruction; portable import check with mjlab blocked; bounded training smoke on an idle GPU; artifact hashes; a moved-directory demo; archive and source-path scan. Record failures and limitations.
