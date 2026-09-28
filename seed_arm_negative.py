"""Physical arm-placement sweep to seed subsequent GPU CEM refinement."""
import json
import numpy as np
from pathlib import Path
from evaluate import rollout
from scene import ROOT
run=ROOT/'artifacts/cem13-arms';out=ROOT/'artifacts/arm-negative-sweep';out.mkdir(exist_ok=True)
names=json.loads((run/'contract.json').read_text())['joint_names'];groups=[[names.index(side+'_'+stem+'_joint') for side in ['left','right']] for stem in ['hip_pitch','knee','ankle_pitch','shoulder_pitch']]+[[names.index('waist_pitch_joint')]]+[[names.index(side+'_'+stem+'_joint') for side in ['left','right']] for stem in ['elbow','shoulder_roll']]
kt=np.array([0,.25,.48,.64,.78,.9,1.15,1.5,1.9,2.6]);base=np.load(ROOT/'artifacts/cem11-back-landing/best.npz')['params'].reshape(8,5);base=np.pad(base,((0,0),(0,2)))
results=[]
for start in [4,5,6]:
 for shoulder in [-.5,-1.,-1.5,-2.2]:
  for elbow in [0.,-.6]:
   p=base.copy();p[start:,3]=shoulder;p[start:,5]=elbow
   act=np.zeros((125,29),dtype='f')
   for g,ids in enumerate(groups):
    values=np.r_[0,p[:,g],0];v=np.interp(np.arange(125)*.02,kt,values)/.5
    act[:,ids]=v[:,None]
   act[:,names.index('right_shoulder_roll_joint')]*=-1
   tag=f's{start}-a{shoulder}-e{elbow}';f=out/(tag+'.npz');np.savez(f,params=p.reshape(-1),actions=act)
   r,q,stats=rollout(run,None,seed=0,action_sequence=str(f));r['tag']=tag;results.append(r)
   np.savez(out/(tag+'-trajectory.npz'),qpos=q,stats=stats)
   print(json.dumps({k:r.get(k) for k in ['tag','success','failed_requirements','first_mat_torso_up','first_mat_contact_geoms','final_torso_up','minimum_clearance_m']}),flush=True)
(out/'metrics.json').write_text(json.dumps(results,indent=2))
