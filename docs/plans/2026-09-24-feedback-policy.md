# G1 state-feedback imitation and PPO implementation plan

Goal: replace the time-only CEM controller as the delivery with a trained neural state-feedback policy crossing a 0.8 m bar.

Architecture: retain the audited scene and DC motors. Generate a dynamically consistent demonstration from the former controller, including measured states and commanded joint targets. Train a PPO residual actor around the demonstration command; actor inputs contain reference commands, joint encoder position/velocity, torso projected gravity and angular velocity, phase. Global position, true linear velocity and contacts are critic/reward only. Reference commands are permitted, as in motion imitation; reference root states are never written during evaluation.

1. Record demonstration state at control boundaries, and matching target angles; verify original success reproducibility.
2. Implement deployable observation contract, reference-target residual action, physical imitation rewards and perturbed starts.
3. Verify GPU/native observation parity and zero-residual dynamics before PPO. Train with saved checkpoints and logs.
4. Export frozen neural actor. Evaluate actual online inference in native dynamics, recording states, targets, contact and limits. Compare perturbed actor runs against fixed-target baseline and disabled-feedback observations.
5. Report exact backend pair. MuJoCo-Warp to native MuJoCo is backend transfer, not a second physics engine. Do not label it Isaac-to-MuJoCo.
6. Export inference-loop video and document measured success/failure, actor dependencies and remaining hardware limitations.

Acceptance: strict existing contact/clearance/back-landing/motor criteria; nominal success plus reproducible perturbation comparison and a measurable policy response to changed proprioceptive state. No claims of robust balance or real-robot readiness beyond evidence.
