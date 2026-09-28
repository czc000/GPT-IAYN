import json
import numpy as np
import mujoco
from task import config
from scene import ROOT
from evaluate import observation
from mjlab.envs import ManagerBasedRlEnv

c=config(1,play=True); e=ManagerBasedRlEnv(c,device='cuda:0'); e.reset()
actual=e.observation_manager.compute()['actor'][0].cpu().numpy()
z=np.load(ROOT/'artifacts/reference/motion.npz')
contract=json.loads((ROOT/'artifacts/smoke/contract.json').read_text())
m=mujoco.MjModel.from_xml_path(str(ROOT/'artifacts/smoke/scene/scene.xml')); d=mujoco.MjData(m)
d.qpos[:]=e.sim.data.qpos[0].cpu().numpy();d.qvel[:]=e.sim.data.qvel[0].cpu().numpy();mujoco.mj_forward(m,d)
k=int(e.command_manager.get_term('motion').time_steps[0]); native=observation(m,d,z,k,contract,np.zeros(29))
error=np.abs(actual-native)
print('TRANSFER_OBSERVATION',json.dumps(dict(dim=len(actual),reference_frame=k,max_abs=float(error.max()),indices=np.where(error>1e-4)[0].tolist())))
np.testing.assert_allclose(actual,native,atol=2e-4,rtol=1e-4)
e.close()
