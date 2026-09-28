"""Actor dependencies and deployment output contract, independent of reward code."""
import unittest,json
import numpy as np,mujoco,torch
from gpt_iayn_core.scene import ROOT
from gpt_iayn_core.evaluate import load_native_model
from gpt_iayn_core.observations import native_obs
from gpt_iayn_core.policy_export import TargetActor

class ActorContractTests(unittest.TestCase):
    def setUp(self):
        src=ROOT/'artifacts/final';self.c=json.loads((src/'contract.json').read_text())
        self.m=load_native_model(src,self.c);self.d=mujoco.MjData(self.m)
        self.z=np.load(ROOT/'artifacts/rl-reference/motion.npz')
        self.d.qpos[:]=self.z['qpos'][35];self.d.qvel[:]=self.z['qvel'][35];mujoco.mj_forward(self.m,self.d)
    def test_actor_does_not_observe_world_translation_or_linear_velocity(self):
        before=native_obs(self.m,self.d,self.z,35)
        self.d.qpos[:3]+=[10,20,30];self.d.qvel[:3]+=[2,3,4];mujoco.mj_forward(self.m,self.d)
        after=native_obs(self.m,self.d,self.z,35)
        np.testing.assert_allclose(before,after,atol=1e-6)
    def test_actor_observes_joint_and_attitude_changes(self):
        before=native_obs(self.m,self.d,self.z,35)
        self.d.qpos[7]+=.1;self.d.qvel[6]+=.2;mujoco.mj_forward(self.m,self.d)
        after=native_obs(self.m,self.d,self.z,35)
        self.assertAlmostEqual(float(after[87]-before[87]),.1,places=5)
        self.assertAlmostEqual(float(after[116]-before[116]),.01,places=5)
        self.assertEqual(before.shape,(152,))
    def test_export_returns_absolute_targets_and_respects_joint_limits(self):
        actor=torch.nn.Linear(152,29);torch.nn.init.zeros_(actor.weight);torch.nn.init.constant_(actor.bias,1.)
        limits=torch.tensor(self.m.jnt_range[1:],dtype=torch.float32)
        module=TargetActor(actor,torch.nn.Identity(),limits).eval()
        obs=torch.tensor(native_obs(self.m,self.d,self.z,35)[None])
        expected=np.clip(self.z['target'][35]+.25,limits[:,0].numpy()+.01,limits[:,1].numpy()-.01)
        np.testing.assert_allclose(module(obs).detach().numpy()[0],expected,atol=2e-7)
        traced=torch.jit.trace(module,obs)
        np.testing.assert_array_equal(traced(obs).detach().numpy(),module(obs).detach().numpy())
if __name__=='__main__':unittest.main()
