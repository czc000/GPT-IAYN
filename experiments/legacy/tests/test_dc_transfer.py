import unittest,json
import numpy as np,mujoco,torch
from pathlib import Path
from gpt_iayn_core.scene import ROOT
from gpt_iayn_core.task import config
from gpt_iayn_core.motor import torque_numpy,parameters
from mjlab.envs import ManagerBasedRlEnv
from gpt_iayn_core.evaluate import load_native_model
class DcTransferTests(unittest.TestCase):
    def test_native_motor_loop_matches_gpu_motor_loop(self):
        c=config(1,play=True,dc=True);c.terminations={};e=ManagerBasedRlEnv(c,device='cuda:0');e.reset()
        run=ROOT/'artifacts/cem04-dc';contract=json.loads((run/'contract.json').read_text());m=load_native_model(run,contract);d=mujoco.MjData(m)
        d.qpos[:]=e.sim.data.qpos[0].cpu().numpy();d.qvel[:]=e.sim.data.qvel[0].cpu().numpy();mujoco.mj_forward(m,d)
        names=contract['joint_names'];mp=parameters(names);z=np.load(contract['motion_file']);limits=m.jnt_range[1:]
        j2c={int(j):i for i,j in enumerate(m.actuator_trnid[:,0])};ctrl=[j2c[m.joint('robot/'+n).id] for n in names]
        worst=0.
        for k in range(30):
            act=np.zeros(29,dtype='f');act[0]=.05*np.sin(k*.2)
            target=np.clip(z['joint_pos'][k]+.5*act,limits[:,0]+.01,limits[:,1]-.01)
            for _ in range(8):
                d.ctrl[ctrl]=torque_numpy(target,d.qpos[7:],d.qvel[6:],mp);mujoco.mj_step(m,d)
            e.step(torch.tensor(act[None],device='cuda:0'))
            actual=e.sim.data.qpos[0].cpu().numpy();worst=max(worst,float(np.max(abs(actual-d.qpos))))
        e.close();print('DC_ROLLOUT_MAX_QPOS_ERROR',worst)
        self.assertLess(worst,.005)
