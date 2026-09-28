"""Deployable 152-dimensional native observation contract."""
import numpy as np,mujoco

def native_obs(m,d,z,k,ablate=False):
    torso=m.body('robot/pelvis').id;rot=d.xmat[torso].reshape(3,3)
    velocity=np.zeros(6);mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_BODY,torso,velocity,1)
    if ablate:state=[np.zeros(29),z['joint_vel'][k]*.05,z['gravity'][k],z['gyro'][k]*.1]
    else:state=[d.qpos[7:]-z['joint_pos'][k],d.qvel[6:]*.05,-rot[2,:],d.sensor('robot/imu_ang_vel').data*.1]
    return np.concatenate([z['target'][k],z['joint_pos'][k],z['joint_vel'][k]*.05,*state,[k/130.]]).astype('f')
