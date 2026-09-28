"""Reject failed PhysX transfer using actual substep trajectory and contact reports."""
import json,numpy as np,mujoco
from scene import ROOT,BAR_RADIUS
from evaluate import load_native_model,collision_bounds
out=ROOT/'artifacts/rl-isaac/frozen-feedback499'
z=np.load(out/'trajectory.npz');c=json.loads((ROOT/'artifacts/final/contract.json').read_text());m=load_native_model(ROOT/'artifacts/final',c);d=mujoco.MjData(m)
ids=np.array([i for i in range(m.ngeom) if (m.geom(i).name or '').startswith('robot/') and (m.geom_contype[i] or m.geom_conaffinity[i])])
minimum=float('inf');cleared=False;limit=0.
for q in z['substep_qpos']:
 d.qpos[:]=q;mujoco.mj_forward(m,d);lo,hi=collision_bounds(m,d,ids)
 spans=(lo[:,0]<=-.7+BAR_RADIUS)&(hi[:,0]>=-.7-BAR_RADIUS)
 if spans.any() and not cleared:minimum=min(minimum,float((lo[spans,2]-.8-BAR_RADIUS).min()))
 if (hi[:,0]<-.7-BAR_RADIUS).all():cleared=True
 limit=max(limit,float(np.maximum(m.jnt_range[1:,0]-q[7:],q[7:]-m.jnt_range[1:,1]).max()))
contacts=json.loads((out/'contacts.json').read_text())
bar=[x for x in contacts if ('/World/Bar' in x['a'] or '/World/Bar' in x['b']) and (x['separation']<=0 or x['impulse']>1e-5)]
result=dict(success=False,backend='Isaac Sim 4.5 / PhysX CPU',minimum_clearance_m=minimum,bar_contact_reports=len(bar) if contacts else None,physics_contact_reporting_valid=bool(contacts),all_crossed=cleared,max_joint_limit_violation_rad=limit,
 max_joint_speed_ratio=float((abs(z['substep_joint_vel'])/np.array(c['motor_parameters']['velocity'])).max()),max_root_height_m=float(z['substep_qpos'][:,2].max()),
 note='Post-hoc geometric audit of PhysX-produced states, not another dynamic rollout. Any failed necessary condition rejects success; complete landing audit is not asserted.')
result['failed_necessary_conditions']=(['bar_clearance'] if minimum<=0 else [])+(['bar_contacts'] if bar else [])+(['joint_limits'] if limit>.02 else [])
assert result['failed_necessary_conditions'],'Need a full contact/landing audit before any success claim'
(out/'audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
