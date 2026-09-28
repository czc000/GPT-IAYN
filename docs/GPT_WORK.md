# What “GPT-built” means

GPT-IAYN records an agent-driven development experiment carried out under a user's direction. The user defined the 0.8 m G1 high-jump objective, supplied compute access, reviewed videos, challenged weak or misleading results, and steered the work toward a genuine state-feedback policy.

GPT's project-specific work included scene assembly; synthetic keyframe generation; CEM controller search; actuator and acceptance implementation; identifying invalid candidates; physical demonstration recording; feedback-PPO integration; training/evaluation orchestration; TorchScript/ONNX export; native MuJoCo and PhysX experiment tooling; provenance records; and this release organization.

The robot's runtime controller is a trained MLP, not GPT. It sees a fixed reference and proprioception, outputs joint targets, and relies on the simulator's dynamics. GPT did not design the G1 hardware, author the underlying physics engine, invent PPO, or author mjlab/RSL-RL/BeyondMimic/PyTorch. Their contributions remain attributed.

The name **GPT Is All You Need** is an experiment title. It does not establish fully autonomous research, general robotic intelligence, replacement of human judgment, or a real-world deployment result. The repository includes failures and limitations rather than treating attractive videos as proof.

## Scope of this release

- Standing, two-foot backward jump only, from source commit `8ce7be6876dfcc1d2d31e83210f02f4ae2bac047`.
- No subsequent run-up, perception, human-video reconstruction, teleoperation or unrelated workspace material.
- No real robot operation. No claim that the trained policy handles arbitrary start poses, bar positions or bar heights.
- No complete private conversation dump, credentials, local SSH configuration or external simulator installation.

This attribution describes the workflow documented in the project. It does not identify an exact GPT model revision for every historical turn; those revisions are not reliably recorded in the source artifacts.
