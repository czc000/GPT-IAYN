"""Package only independently accepted, hash-matched dynamic results."""
from gpt_iayn_core.paths import ACCEPTANCE_SOURCE
import argparse,hashlib,json,shutil
from pathlib import Path
import numpy as np
from gpt_iayn_core.scene import ROOT

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--controller',required=True);p.add_argument('--evaluation',required=True);p.add_argument('--output',type=Path,default=ROOT/'artifacts/final');a=p.parse_args()
    run=Path(a.run);controller=Path(a.controller);ev=Path(a.evaluation)
    metrics=json.loads((ev/'metrics.json').read_text());assert len(metrics)==20
    assert metrics[0]['success'] and not metrics[0]['reference_only']
    assert all(r['controller_sha256']==sha(controller) and r['acceptance_sha256']==sha(ACCEPTANCE_SOURCE) for r in metrics)
    out=a.output;assert not out.exists(), 'Preserve existing package; choose/archive explicitly first.'
    out.mkdir();shutil.copytree(run/'scene',out/'scene');shutil.copytree(ev,out/'native-evaluation')
    contract=json.loads((run/'contract.json').read_text());shutil.copy2(__import__('gpt_iayn_core.portable',fromlist=['motion_path']).motion_path(run,contract),out/'reference.npz');contract['motion_file']='reference.npz'
    (out/'contract.json').write_text(json.dumps(contract,indent=2));shutil.copy2(controller,out/'frozen-controller.npz')
    shutil.copy2(ev/'seed0.npz',out/'nominal-trajectory.npz');shutil.copy2(ev/'metrics.json',out/'metrics-20-seeds.json')
    if (ROOT/'artifacts/environment-pip-freeze.txt').exists():shutil.copy2(ROOT/'artifacts/environment-pip-freeze.txt',out/'environment-pip-freeze.txt')
    n=sum(r['success'] for r in metrics);nom=metrics[0];good=[r for r in metrics if r['success']]
    info=dict(training_run=str(run.resolve()),controller_source=str(controller.resolve()),controller_sha256=sha(controller),acceptance_sha256=sha(ACCEPTANCE_SOURCE),passed=n,trials=20,parameter_count=int(np.load(controller)['params'].size),nominal=nom)
    (out/'provenance.json').write_text(json.dumps(info,indent=2))
    report=f'''# G1 0.80 m back-first high jump — verified result

Frozen CEM controller with {info['parameter_count']} learned spline parameters and joint PD feedback. Training: MuJoCo-Warp. Independent evaluation: native CPU MuJoCo. Standing two-foot backward takeoff, no run-up. This is not a neural PPO result or Isaac-to-MuJoCo transfer.

- Native small-initial-pose-perturbation trials: **{n}/20** passed; seed 0 is nominal, seeds 1–19 use Gaussian joint-angle noise with sigma 0.005 rad.
- Nominal minimum collision-geometry clearance above the rod surface: **{nom['minimum_clearance_m']:.4f} m**; robot–bar contacts: **{nom['bar_contacts']}**.
- Nominal maximum pelvis height: {nom['max_pelvis_height_m']:.4f} m; continuous unsupported flight: {nom['max_air_time_s']:.4f} s.
- First mat contact at {nom['first_mat_time_s']:.4f} s: {', '.join(nom['first_mat_contact_geoms'])}. Head-first: {nom['first_mat_contact_head']}. First torso/pelvis mat support: {nom['first_back_support_time_s']:.4f} s, upward-facing component {nom['back_support_torso_up']:.4f}; any head contact before that support: {nom['head_before_back_support']}. Torso upward-facing component at landing/final: {nom['first_mat_torso_up']:.4f}/{nom['final_torso_up']:.4f}.
- Worst joint-limit solver violation across all trials: {max(r['max_joint_limit_violation_rad'] for r in metrics):.6f} rad (acceptance tolerance 0.02); worst speed/rated-speed ratio: {max(r['max_joint_speed_ratio'] for r in metrics):.6f} (threshold 1.05).
- All dynamics use gravity, original robot mass/collisions and a conservative linear DC torque-speed envelope. Grounded zero-velocity start; no root actuation, support forces, teleports, or mid-episode resets.

Bar center height 0.800 m, radius 0.012 m, x={contract.get('bar_x_m',-.7):.3f} m; mat top {contract.get('mat_top_m',.18):.3f} m. The bar is fixed and collidable; every touch fails acceptance. The soft-contact pad is a simulation approximation. No real-robot impact safety or broad robustness claim is made.

## Artifacts and reproduction

Remote project: `{ROOT}`. Final package: `{out}`. Per-trial metrics and dynamic trajectories: `artifacts/final/native-evaluation/`. Controller hash: `{info['controller_sha256']}`.

```bash
cd {ROOT}
./scripts/runtime.sh scripts/evaluate.py --run artifacts/final --action-sequence artifacts/final/frozen-controller.npz --output artifacts/reproduce-final --seeds 20 --video
./scripts/runtime.sh scripts/render_final.py
```

The multi-view MP4 is a render of the independently simulated native trajectory, including normal speed and labeled slow motion; it is not the kinematic training reference. Earlier rejected results remain separately labeled diagnostics in README.md.
'''
    (out/'RESULTS.md').write_text(report)
    print(json.dumps(info,indent=2))
if __name__=='__main__':main()
