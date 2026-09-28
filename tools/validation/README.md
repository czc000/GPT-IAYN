# Independent validation utilities

These utilities depend on the core package, never the other way around. Run from the project root in the documented environment.

- `check_rl.py`: GPU/native observation parity and reset phase checks.
- `check_rl_dynamics.py`: zero-residual GPU/reference dynamics diagnostic.
- `audit_contacts.py`: native substep contact-force audit (`--help` lists required inputs).
- `audit_physx_geometry.py`: audit a PhysX trajectory against native collision geometry (`--run`).

Some diagnostics require generated trajectories; they are not automatically invoked by the demo. The current unit suite is in `tests/`. Shared observation and capsule-plane geometry functions live in `gpt_iayn_core.observations` and `gpt_iayn_core.geometry`.
