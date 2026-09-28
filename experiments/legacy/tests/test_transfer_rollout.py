"""Integration check: exact policy observations after moving through the GPU simulator."""
import json, unittest
import numpy as np
import mujoco, torch
from gpt_iayn_core.task import config
from gpt_iayn_core.scene import ROOT
from gpt_iayn_core.evaluate import observation, collision_bounds
from cem_refine import Bounds
from mjlab.envs import ManagerBasedRlEnv

class TransferRolloutTest(unittest.TestCase):
    def test_native_observations_match_gpu_states_over_multiple_phases(self):
        c=config(4,play=True); c.terminations={}
        e=ManagerBasedRlEnv(c,device='cuda:0');e.reset()
        np.testing.assert_allclose(e.scene.env_origins.cpu().numpy(),0.,atol=0.,err_msg='All independent Warp worlds must share the obstacle coordinate origin')
        z=np.load(ROOT/'artifacts/reference/motion.npz')
        contract=json.loads((ROOT/'artifacts/smoke/contract.json').read_text())
        m=mujoco.MjModel.from_xml_path(str(ROOT/'artifacts/smoke/scene/scene.xml'));d=mujoco.MjData(m)
        bounds=Bounds(e.sim.mj_model,'cuda:0')
        worst=0.
        for k in range(70):
            action=torch.zeros((4,29),device='cuda:0')
            action[0,0]=.2*np.sin(k*.1)
            e.step(action)
            actual=e.observation_manager.compute()['actor'][0].cpu().numpy()
            d.qpos[:]=e.sim.data.qpos[0].cpu().numpy();d.qvel[:]=e.sim.data.qvel[0].cpu().numpy();mujoco.mj_forward(m,d)
            frame=int(e.command_manager.get_term('motion').time_steps[0])
            native=observation(m,d,z,frame,contract,action[0].cpu().numpy())
            worst=max(worst,float(np.max(np.abs(actual-native))))
            np.testing.assert_allclose(actual,native,atol=3e-3,rtol=1e-3)
            glo,ghi=bounds.get(e);clo,chi=collision_bounds(m,d,bounds.ids.cpu().numpy())
            np.testing.assert_allclose(glo[0].cpu().numpy(),clo,atol=2e-5)
            np.testing.assert_allclose(ghi[0].cpu().numpy(),chi,atol=2e-5)
        print('MULTIPHASE_OBSERVATION_MAX_ERROR',worst)
        e.close()
