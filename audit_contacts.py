"""Independent full-substep contact load audit of a frozen native controller."""
import argparse,json
from pathlib import Path
import mujoco,numpy as np
from evaluate import rollout
p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--controller',required=True);p.add_argument('--output',required=True);a=p.parse_args()
original=mujoco.mj_step;rows=[];back_seen=False

def step(m,d):
 global back_seen
 original(m,d)
 names=[m.geom(i).name for i in range(m.ngeom)];mat=names.index('landing_mat');forces={};dist={}
 for i,c in enumerate(d.contact):
  if mat not in (c.geom1,c.geom2):continue
  name=names[c.geom2 if c.geom1==mat else c.geom1];f=np.zeros(6);mujoco.mj_contactForce(m,d,i,f)
  forces[name]=forces.get(name,0)+float(f[0]);dist[name]=min(dist.get(name,0),float(c.dist))
 torso=m.body('robot/torso_link').id
 back_now=any(n in forces for n in ['robot/torso_collision','robot/pelvis_collision']);back_seen|=back_now
 if forces:rows.append(dict(time=float(d.time),forces=forces,distance=dist,back_seen=back_seen,up=float(d.xmat[torso].reshape(3,3)[2,0])))
mujoco.mj_step=step
result,q,stats=rollout(Path(a.run),None,seed=0,action_sequence=a.controller)
heads=[r for r in rows if 'robot/head_collision' in r['forces']];backs=[r for r in rows if any(n in r['forces'] for n in ['robot/torso_collision','robot/pelvis_collision'])];pre=[r for r in heads if not r['back_seen']]
summary=dict(result=result,first_head=heads[0] if heads else None,first_back=backs[0] if backs else None,head_pre_back_peak_normal_force_N=max([r['forces']['robot/head_collision'] for r in pre],default=0),head_pre_back_impulse_Ns=sum(r['forces']['robot/head_collision']*.0025 for r in pre),contacts=rows)
Path(a.output).write_text(json.dumps(summary,indent=2));print(json.dumps({k:v for k,v in summary.items() if k not in ['contacts','result']},indent=2))
