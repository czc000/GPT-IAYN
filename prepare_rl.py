"""Record a dynamic teacher, not kinematic reference root playback."""
import argparse,json
import numpy as np
import mujoco
from scene import ROOT
from evaluate import load_native_model
from motor import torque_numpy

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=__import__('pathlib').Path,default=ROOT/'artifacts/final');parser.add_argument('--output',type=__import__('pathlib').Path,default=ROOT/'artifacts/rl-reference');args=parser.parse_args()
    src=args.source;c=json.loads((src/'contract.json').read_text())
    ref=np.load(__import__('portable').motion_path(src,c)); seq=np.load(src/'frozen-controller.npz')['actions']
    m=load_native_model(src,c);d=mujoco.MjData(m);d.qpos[:]=ref['qpos'][0];mujoco.mj_forward(m,d)
    bids=[m.body('robot/'+n).id for n in c['body_names']]
    ctrls=[next(i for i,j in enumerate(m.actuator_trnid[:,0]) if j==m.joint('robot/'+n).id) for n in c['joint_names']]
    records={k:[] for k in ['qpos','qvel','body_pos_w','body_quat_w','body_lin_vel_w','body_ang_vel_w','target','gravity','gyro']}
    for k in range(len(ref['time'])):
        mujoco.mj_forward(m,d)
        target=np.clip(ref['joint_pos'][k]+.5*seq[min(k,len(seq)-1)],m.jnt_range[1:,0]+.01,m.jnt_range[1:,1]-.01)
        vel=[]
        for b in bids:
            v=np.zeros(6);mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_BODY,b,v,0);vel.append(v)
        vel=np.array(vel);torso=m.body('robot/pelvis').id
        values=[d.qpos.copy(),d.qvel.copy(),d.xpos[bids].copy(),d.xquat[bids].copy(),vel[:,3:],vel[:,:3],target,
                d.xmat[torso].reshape(3,3).T@np.array([0,0,-1]),d.sensor('robot/imu_ang_vel').data.copy()]
        for name,value in zip(records,values):records[name].append(value)
        for _ in range(8):
            d.ctrl[ctrls]=torque_numpy(target,d.qpos[7:],d.qvel[6:],c['motor_parameters']);mujoco.mj_step(m,d)
    records={k:np.asarray(v) for k,v in records.items()}
    old=np.load(src/'nominal-trajectory.npz')['qpos']
    error=float(np.max(np.abs(records['qpos'][1:]-old[:-1])))
    assert error<1e-6,error
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    np.savez(out/'motion.npz',**records,joint_pos=records['qpos'][:,7:],joint_vel=records['qvel'][:,6:],fps=50,
             body_names=np.array(c['body_names']),joint_names=np.array(c['joint_names']),time=ref['time'])
    old=np.load(src/'nominal-trajectory.npz')['qpos']
    error=float(np.max(np.abs(records['qpos'][1:]-old[:-1])))
    assert error<1e-6,error
    print('teacher boundary alignment max error',error)
if __name__=='__main__':main()
