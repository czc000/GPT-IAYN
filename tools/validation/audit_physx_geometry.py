"""Exact geometry audit of saved PhysX dynamics, without resimulating or changing poses."""
import json,numpy as np,mujoco
from gpt_iayn_core.scene import ROOT
from gpt_iayn_core.evaluate import load_native_model,collision_bounds
from gpt_iayn_core.geometry import plane_minimum
from gpt_iayn_core.acceptance import assess
import argparse
p=argparse.ArgumentParser();p.add_argument('--run',default=str(ROOT/'artifacts/rl-isaac/frozen-feedback499-contact-enabled'));args=p.parse_args()
src=__import__('pathlib').Path(args.run);z=np.load(src/'trajectory.npz');contract=json.load(open(ROOT/'artifacts/final/contract.json'))
m=load_native_model(ROOT/'artifacts/final',contract);d=mujoco.MjData(m)
ids=np.array([i for i in range(m.ngeom) if m.geom(i).name.startswith('robot/') and (m.geom_contype[i] or m.geom_conaffinity[i])]);names=[m.geom(i).name for i in ids]
bar=m.geom('crossbar').id;mat=m.geom('landing_mat').id;floor=next(i for i in range(m.ngeom) if m.geom_type[i]==mujoco.mjtGeom.mjGEOM_PLANE);torso=m.body('robot/torso_link').id
bar_dist=[];mat_dist=[];floor_dist=[];ups=[];bounds=[];plane=[];torso_x=[]
for pose in z['substep_qpos']:
 d.qpos[:]=pose;mujoco.mj_forward(m,d)
 bar_dist.append([mujoco.mj_geomDistance(m,d,int(i),bar,10.,None) for i in ids])
 mat_dist.append([mujoco.mj_geomDistance(m,d,int(i),mat,10.,None) for i in ids])
 floor_dist.append([mujoco.mj_geomDistance(m,d,int(i),floor,10.,None) for i in ids])
 ups.append(float(d.xmat[torso].reshape(3,3)[2,0]));torso_x.append(float(d.xpos[torso,0]));bounds.append(collision_bounds(m,d,ids));plane.append(plane_minimum(m,d,ids)[0])
bar_dist=np.array(bar_dist);mat_dist=np.array(mat_dist);floor_dist=np.array(floor_dist);bounds=np.array(bounds);plane=np.array(plane)
clear=np.all(bounds[:,1,:,0]<-.712,axis=1);clear_i=int(np.flatnonzero(clear)[0]) if clear.any() else None
crossing=np.flatnonzero(np.array(torso_x)<-.7);cross_i=int(crossing[0]) if len(crossing) else None
limit=float(np.maximum(m.jnt_range[1:,0]-z['substep_qpos'][:,7:],z['substep_qpos'][:,7:]-m.jnt_range[1:,1]).max());speed=float((abs(z['substep_joint_vel'])/np.array(contract['motor_parameters']['velocity'])).max())
reports=[]
for tolerance in [0.,.0001,.0005,.001]:
 touch=mat_dist<=tolerance;floor_touch=floor_dist<=tolerance
 first=np.flatnonzero(touch.any(-1));first_i=int(first[0]) if len(first) else None
 back_ids=[i for i,n in enumerate(names) if n in ['robot/pelvis_collision','robot/torso_collision']];head_ids=[i for i,n in enumerate(names) if 'head' in n]
 backs=np.flatnonzero(touch[:,back_ids].any(-1));back_i=int(backs[0]) if len(backs) else None
 heads=np.flatnonzero(touch[:,head_ids].any(-1));head_i=int(heads[0]) if len(heads) else None
 foots=[i for i,n in enumerate(names) if 'foot' in n];side_ground=[any(floor_touch[:120,i].any() for i in foots if side in names[i]) for side in ['left','right']]
 spans=(bounds[:,0,:,0]<=-.688)&(bounds[:,1,:,0]>=-.712)
 through=(bounds[:,0,:,1]>-1)&(bounds[:,1,:,1]<1)
 airborne=~((mat_dist<=tolerance).any(-1)|(floor_dist<=tolerance).any(-1)|(bar_dist<=0).any(-1));flight=0;maxflight=0
 for a in airborne:flight=flight+1 if a else 0;maxflight=max(maxflight,flight)
 trace=dict(finished_on_landing_side=bool(z['substep_qpos'][-1,0]<-.712 and torso_x[-1]<-.712),grounded_start=all(side_ground),took_off=bool(maxflight*.0025>=.05 and z['substep_qpos'][:,2].max()>.8),
 all_above_bar=bool(np.min(plane[:clear_i+1 if clear_i is not None else len(plane)])>0),all_crossed=bool(clear.any()),between_posts=bool(through[spans].all()),landed_on_mat=bool(first_i is not None and clear_i is not None and touch[clear_i:].any()),
 max_joint_limit_violation_rad=limit,max_joint_speed_ratio=speed,premature_mat_contact=bool(first_i is not None and (clear_i is None or first_i<clear_i)),
 head_before_back_support=bool(head_i is not None and (back_i is None or head_i<back_i)),back_support_torso_up=ups[back_i] if back_i is not None else -1,
 first_mat_contact_head=bool(first_i is not None and touch[first_i,head_ids].any()),first_mat_torso_up=ups[first_i] if first_i is not None else -1,
 final_torso_up=ups[-1],bar_contacts=int((bar_dist< -1e-5).sum()),torso_up_at_crossing=ups[cross_i] if cross_i is not None else -1,
 first_mat_contact_geoms=[names[i] for i in np.flatnonzero(touch[first_i])] if first_i is not None else [],
 first_mat_time_s=(first_i+1)*.0025 if first_i is not None else None,first_back_support_time_s=(back_i+1)*.0025 if back_i is not None else None,first_head_contact_time_s=(head_i+1)*.0025 if head_i is not None else None)
 reports.append(dict(contact_distance_tolerance_m=tolerance,**trace,**assess(trace)))
contacts=json.loads((src/'contacts.json').read_text())
bar_reports=[r for r in contacts if '/World/Bar' in (r['a']+r['b']) and (r['impulse']>1e-6 or r['separation']<=0)]
for report in reports:
    report['geometry_only_success']=report.pop('success')
result=dict(source='PhysX dynamic substep states; offline exact capsule/sphere geometry audit plus independent PhysX contact reports',physics_contact_reporting_valid=bool(contacts),
 success=bool(contacts) and not bar_reports and all(r['geometry_only_success'] for r in reports),physics_bar_contact_reports=len(bar_reports),
 physics_bar_peak_report_impulse_Ns=max([r['impulse'] for r in bar_reports],default=0.),
 minimum_bar_surface_distance_m=float(bar_dist.min()),minimum_centre_plane_clearance_m=float(np.min(plane[:clear_i+1 if clear_i is not None else len(plane)])),
 all_clear_time_s=(clear_i+1)*.0025 if clear_i is not None else None,contact_tolerance_sensitivity=reports)
(src/'exact-geometry-audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
