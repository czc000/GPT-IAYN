"""Exact capsule/sphere intersection with bar centre plane, supplementary geometric diagnostic."""
import numpy as np,mujoco,json
from gpt_iayn_core.scene import ROOT
from gpt_iayn_core.evaluate import load_native_model

from gpt_iayn_core.geometry import plane_minimum

if __name__=='__main__':
 c=json.load(open(ROOT/'artifacts/final/contract.json'));m=load_native_model(ROOT/'artifacts/final',c);d=mujoco.MjData(m)
 ids=[i for i in range(m.ngeom) if m.geom(i).name.startswith('robot/') and (m.geom_contype[i] or m.geom_conaffinity[i])]
 z=np.load(ROOT/'artifacts/rl-isaac/frozen-feedback499/trajectory.npz');results=[]
 # Only outbound crossing until full clearance; mat-side limbs returning under bar are separate.
 for i,q in enumerate(z['substep_qpos']):
  d.qpos[:]=q;mujoco.mj_forward(m,d);r=plane_minimum(m,d,ids);results.append((*r,(i+1)*.0025))
  from gpt_iayn_core.evaluate import collision_bounds
  _,hi=collision_bounds(m,d,np.array(ids))
  if (hi[:,0]<-.712).all():break
 result=min(results);print('CAPSULE_CENTRE_PLANE_CLEARANCE',result)
 (ROOT/'artifacts/rl-isaac/frozen-feedback499/plane-audit.json').write_text(json.dumps(dict(minimum_clearance_m=result[0],geom=result[1],time_s=result[2],contact_reporting_valid=False),indent=2))
