import unittest,numpy as np,torch
from gpt_iayn_core.scene import ROOT
from gpt_iayn_core.feedback_actor import FeedbackActor
class FeedbackTests(unittest.TestCase):
    def test_no_time_only_correction_and_state_gradients_survive(self):
        z=np.load(ROOT/'artifacts/rl-reference/motion.npz');net=torch.nn.Sequential(torch.nn.Linear(152,64),torch.nn.ELU(),torch.nn.Linear(64,29))
        actor=FeedbackActor(net,z['gravity'],z['gyro'])
        obs=torch.zeros(131,152);obs[:,:29]=torch.tensor(z['target']);obs[:,29:58]=torch.tensor(z['joint_pos']);obs[:,58:87]=torch.tensor(z['joint_vel']*.05)
        obs[:,116:145]=obs[:,58:87];obs[:,145:148]=torch.tensor(z['gravity']);obs[:,148:151]=actor.reference_gyro*.1;obs[:,-1]=torch.arange(131)/130.
        torch.testing.assert_close(actor(obs),torch.zeros(131,29),atol=0.,rtol=0.)
        changed=obs.clone();changed[:,87]+=.025
        result=actor(changed);self.assertGreater(float(result.abs().max()),1e-5)
        result.square().mean().backward();self.assertGreater(float(net[-1].weight.grad.abs().max()),1e-7)
        traced=torch.jit.trace(actor,obs[:1]);torch.testing.assert_close(traced(changed),result)
if __name__=='__main__':unittest.main()
