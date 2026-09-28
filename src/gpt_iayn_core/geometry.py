"""Exact collision-geometry helpers shared by validators."""
import numpy as np,mujoco

def plane_minimum(m,d,ids,x=-.7):
 result=[]
 for g in ids:
  radius=m.geom_size[g,0];half=m.geom_size[g,1] if m.geom_type[g]==mujoco.mjtGeom.mjGEOM_CAPSULE else 0.
  axis=d.geom_xmat[g].reshape(3,3)[:,2];a=d.geom_xpos[g]-axis*half;b=d.geom_xpos[g]+axis*half;delta=b-a
  if abs(delta[0])<1e-12:
   u=x-a[0]
   if abs(u)>radius:continue
   low=min(a[2],b[2])-np.sqrt(max(0,radius*radius-u*u))
  else:
   ends=sorted([(x-radius-a[0])/delta[0],(x+radius-a[0])/delta[0]])
   lo=max(0,ends[0]);hi=min(1,ends[1])
   if lo>hi:continue
   candidates=[lo,hi]
   ustar=radius*delta[2]*np.sign(delta[0])/np.hypot(delta[0],delta[2]);t=(x-a[0]-ustar)/delta[0]
   if lo<t<hi:candidates.append(t)
   low=min(a[2]+t*delta[2]-np.sqrt(max(0,radius*radius-(x-a[0]-t*delta[0])**2)) for t in candidates)
  result.append((float(low-.812),m.geom(g).name))
 return min(result) if result else (float('inf'),'none')
