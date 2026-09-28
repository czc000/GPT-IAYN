import numpy as np,mujoco,torch,json
from gpt_iayn_core.evaluate import load_native_model
from gpt_iayn_core.scene import ROOT

from gpt_iayn_core.observations import native_obs

def main():
    from gpt_iayn_core.rl_task import config,actor_observation
    from mjlab.envs import ManagerBasedRlEnv
    e=ManagerBasedRlEnv(config(1,play=True),device='cuda:0');e.reset()
    z=np.load(ROOT/'artifacts/rl-reference/motion.npz');m=e.sim.mj_model;d=mujoco.MjData(m)
    for step in range(100):
        d.qpos[:]=e.sim.data.qpos[0].cpu().numpy();d.qvel[:]=e.sim.data.qvel[0].cpu().numpy();mujoco.mj_forward(m,d)
        k=int(e.command_manager.get_term('motion').time_steps[0])
        actual=actor_observation(e)[0].cpu().numpy();expected=native_obs(m,d,z,k)
        np.testing.assert_allclose(actual,expected,atol=3e-4,rtol=3e-4)
        assert actual.shape==(152,)
        e.step(torch.zeros(1,29,device='cuda:0'))
    # Force an automatic reset and ensure its first control observation is phase 0.
    e.episode_length_buf[:]=e.max_episode_length
    e.step(torch.zeros(1,29,device='cuda:0'))
    assert int(e.command_manager.get_term('motion').time_steps[0])==0
    np.testing.assert_allclose(e.sim.data.qpos[0].cpu().numpy(),z['qpos'][0],atol=2e-6)
    e.step(torch.zeros(1,29,device='cuda:0'))
    assert int(e.command_manager.get_term('motion').time_steps[0])==1
    print('100 GPU/native observation comparisons and automatic reset phase checks passed');e.close()
if __name__=='__main__':main()
