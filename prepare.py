"""Procedural reference only: these kinematic frames are never a success result."""
import json
import numpy as np
import mujoco
from scipy.interpolate import PchipInterpolator
from scipy.spatial.transform import Rotation
from scene import ROOT, make_scene, BAR_HEIGHT


def main():
    scene=make_scene(); model=scene.compile(); data=mujoco.MjData(model)
    names=list(scene['robot'].joint_names); bodies=list(scene['robot'].body_names)
    bids=[model.body('robot/'+n).id for n in bodies]
    dt=.02; t=np.arange(0,2.62,dt)
    knots=np.array([0,.30,.62,.84,1.05,1.23,1.43,1.66,1.95,2.60])
    jp=np.tile(model.key_qpos[0,7:],(len(knots),1))
    profiles={
        'hip_pitch':[-.25,-.25,-.85,.05,-.1,-.65,-1.35,-.9,-.25,-.25],
        'knee':[.5,.5,1.65,.05,.28,.55,.6,1.0,1.1,1.1],
        'ankle_pitch':[-.25,-.25,-.8,-.15,-.1,.05,.10,.0,.0,.0],
        'shoulder_pitch':[.15,.15,.8,-1.9,-2.3,-1.8,-.7,.2,.3,.3],
        'elbow':[.5,.5,.7,.2,.3,.6,.7,.7,.6,.6],
    }
    for stem,values in profiles.items():
        for side in ['left','right']: jp[:,names.index(side+'_'+stem+'_joint')]=values
    jp[:,names.index('waist_pitch_joint')]=[0,0,.1,-.08,-.2,-.25,-.1,0,0,0]
    q=np.tile(model.key_qpos[0],(len(t),1)); q[:,7:]=PchipInterpolator(knots,jp,axis=0)(t)
    pitch=PchipInterpolator(knots,[0,0,.03,-.14,-.65,-1.15,-1.55,-1.65,-1.57,-1.57])(t)
    r=Rotation.from_euler('y',pitch[:,None]).as_quat(); q[:,3:7]=r[:,[3,0,1,2]]
    # Feet grounded through the crouch; root path later follows an explicit ballistic seed.
    q[:,0]=PchipInterpolator(knots,[0,0,.04,-.02,-.43,-.80,-1.20,-1.64,-1.85,-1.85])(t)
    foot_ids=[i for i in range(model.ngeom) if 'foot' in (model.geom(i).name or '') and model.geom_contype[i]+model.geom_conaffinity[i]>0]
    for i,tm in enumerate(t):
        data.qpos[:]=q[i]; data.qpos[2]=0; mujoco.mj_forward(model,data)
        low=min(data.geom_xpos[g,2]-model.geom_size[g,0] for g in foot_ids)
        grounded=-low+.002
        if tm<=.68: q[i,2]=grounded
        elif tm<.84:
            q[i,2]=float(PchipInterpolator([.68,.76,.84],[.58,.68,.81])(tm))
        elif tm<=1.62:
            tau=tm-.84; q[i,2]=.81+3.35*tau-4.905*tau*tau
        else:
            q[i,2]=float(PchipInterpolator([1.62,1.85,2.60],[.81+3.35*.78-4.905*.78**2,.39,.39])(tm))
    ranges=model.jnt_range[1:]; q[:,7:]=np.clip(q[:,7:],ranges[:,0]+.01,ranges[:,1]-.01)
    bp=[]; bq=[]; contacts=[]
    for pose in q:
        data.qpos[:]=pose; mujoco.mj_forward(model,data)
        bp.append(data.xpos[bids].copy()); bq.append(data.xquat[bids].copy())
        contacts.append(sum('crossbar' in [model.geom(c.geom1).name,model.geom(c.geom2).name] for c in data.contact))
    bp=np.asarray(bp); bq=np.asarray(bq)
    av=np.empty((len(t),len(bids),3))
    for j in range(len(bids)):
        rot=Rotation.from_quat(bq[:,j,[1,2,3,0]])
        av[:-1,j]=(rot[1:]*rot[:-1].inv()).as_rotvec()/dt
        av[-1,j]=av[-2,j]
    out=ROOT/'artifacts/reference'; out.mkdir(parents=True,exist_ok=True)
    np.savez(out/'motion.npz',fps=50,joint_pos=q[:,7:].astype('f'),joint_vel=np.gradient(q[:,7:],dt,axis=0).astype('f'),
             body_pos_w=bp.astype('f'),body_quat_w=bq.astype('f'),body_lin_vel_w=np.gradient(bp,dt,axis=0).astype('f'),
             body_ang_vel_w=av.astype('f'),qpos=q,body_names=np.array(bodies),joint_names=np.array(names),time=t)
    scene.write(out/'scene')
    summary=dict(kind='kinematic_reference_not_trained_policy',bar_height_m=BAR_HEIGHT,frames=len(t),
                 bar_contact_frames=int(np.count_nonzero(contacts)),pelvis_height_max=float(q[:,2].max()),
                 root_start=q[0,:7].tolist(),reference_path=str(out/'motion.npz'))
    (out/'summary.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
