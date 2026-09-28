import numpy as np,mujoco,torch,json
from evaluate import load_native_model
from scene import ROOT

def native_obs(m,d,z,k,ablate=False):
    torso=m.body('robot/pelvis').id;rot=d.xmat[torso].reshape(3,3)
    velocity=np.zeros(6);mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_BODY,torso,velocity,1)
    if ablate:state=[np.zeros(29),z['joint_vel'][k]*.05,z['gravity'][k],z['gyro'][k]*.1]
    else:state=[d.qpos[7:]-z['joint_pos'][k],d.qvel[6:]*.05,-rot[2,:],d.sensor('robot/imu_ang_vel').data*.1]
    return np.concatenate([z['target'][k],z['joint_pos'][k],z['joint_vel'][k]*.05,*state,[k/130.]]).astype('f')

def main():
    from rl_task import config,actor_observation
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
