import numpy as np,torch
from mjlab.envs import ManagerBasedRlEnv
from rl_task import config
from scene import ROOT
c=config(1,play=True);c.terminations={};e=ManagerBasedRlEnv(c,device='cuda:0');e.reset()
z=np.load(ROOT/'artifacts/rl-reference/motion.npz');worst=0
for k in range(130):
 q=e.sim.data.qpos[0].cpu().numpy();err=np.max(abs(q-z['qpos'][k]));worst=max(worst,float(err))
 if k%10==0:print(k,'maxerr',err,'root',q[:3],flush=True)
 e.step(torch.zeros(1,29,device='cuda:0'))
print('ZERO_RESIDUAL_GPU_NATIVE_MAX_ERROR',worst,flush=True);e.close()
