"""Frozen-policy transfer to native MuJoCo. No root forces or per-frame resets."""
import argparse, json
from pathlib import Path
import numpy as np
import mujoco
import torch
from scipy.spatial.transform import Rotation
from scene import ROOT, BAR_X, BAR_RADIUS, BAR_HALF_WIDTH, BAR_HEIGHT
from acceptance import assess
from motor import parameters,torque_numpy


def collision_bounds(m,d,ids):
    centers=d.geom_xpos[ids].copy(); rots=d.geom_xmat[ids].reshape(-1,3,3)
    ext=np.zeros_like(centers)
    for k,g in enumerate(ids):
        typ=m.geom_type[g]; size=m.geom_size[g]; rot=rots[k]
        if typ==mujoco.mjtGeom.mjGEOM_SPHERE: ext[k]=size[0]
        elif typ==mujoco.mjtGeom.mjGEOM_CAPSULE: ext[k]=np.abs(rot[:,2])*size[1]+size[0]
        elif typ==mujoco.mjtGeom.mjGEOM_CYLINDER: ext[k]=np.sqrt(np.maximum(0,1-rot[:,2]**2))*size[0]+np.abs(rot[:,2])*size[1]
        elif typ==mujoco.mjtGeom.mjGEOM_BOX: ext[k]=np.abs(rot)@size
        else:
            centers[k]+=rot@m.geom_aabb[g,:3]; ext[k]=np.abs(rot)@m.geom_aabb[g,3:]
    return centers-ext,centers+ext


def observation(m,d,z,k,contract,last):
    torso=m.body('robot/torso_link').id
    bi=list(z['body_names']).index('torso_link')
    rot=d.xmat[torso].reshape(3,3)
    rq=z['body_quat_w'][k,bi]; rr=Rotation.from_quat(rq[[1,2,3,0]]).as_matrix()
    qaddr=[m.jnt_qposadr[m.joint('robot/'+n).id] for n in contract['joint_names']]
    vaddr=[m.jnt_dofadr[m.joint('robot/'+n).id] for n in contract['joint_names']]
    terms=[z['joint_pos'][k],z['joint_vel'][k],rot.T@(z['body_pos_w'][k,bi]-d.xpos[torso]),
           (rot.T@rr)[:,:2].reshape(-1),d.sensor('robot/imu_lin_vel').data.copy(),d.sensor('robot/imu_ang_vel').data.copy(),
           d.qpos[qaddr]-np.array(contract['default_joint_pos']),d.qvel[vaddr],last]
    return np.concatenate(terms).astype('f')


def load_native_model(run,contract):
    from portable import apply_physics
    m=mujoco.MjModel.from_xml_path(str(run/'scene/scene.xml'))
    # Earlier exports predate this field; their exact fixed config is recorded here.
    physics=contract.get('physics',dict(timestep=.0025,integrator='implicitfast',iterations=20,ls_iterations=20))
    apply_physics(m,physics)
    return m


def rollout(run,policy,seed=0,video=None,reference=False,action_sequence=None):
    contract=json.loads((run/'contract.json').read_text());bar_x=contract.get('bar_x_m',BAR_X)
    from portable import motion_path
    z=np.load(motion_path(run,contract));m=load_native_model(run,contract)
    assert abs(m.geom_pos[m.geom('crossbar').id,0]-bar_x)<1e-8
    assert abs(m.geom_pos[m.geom('crossbar').id,2]-contract['bar_height_m'])<1e-8
    d=mujoco.MjData(m);d.qpos[:]=z['qpos'][0];d.qvel[:]=0
    if seed: d.qpos[7:]+=np.random.default_rng(seed).normal(0,.005,29)
    mujoco.mj_forward(m,d)
    net=torch.jit.load(str(policy),map_location='cpu').eval() if policy else None
    sequence=np.load(action_sequence)['actions'] if action_sequence else None
    joint_names=contract['joint_names'];aids=np.array(contract['action_joint_ids'])
    motor=contract['motor_parameters'] if 'motor_parameters' in contract else parameters(joint_names);dc=contract.get('motor_model')=='dc'
    vmax=np.array(motor['velocity']);limits=m.jnt_range[1:]
    # Builtin actuators are grouped by motor type; map through transmission joint ids.
    joint_to_ctrl={int(j):i for i,j in enumerate(m.actuator_trnid[:,0])}
    ctrl_for_action=[joint_to_ctrl[m.joint('robot/'+joint_names[j]).id] for j in aids]
    coll=np.array([i for i in range(m.ngeom) if (m.geom(i).name or '').startswith('robot/') and (m.geom_contype[i] or m.geom_conaffinity[i])])
    foot=set(i for i in coll if 'foot' in (m.geom(i).name or ''))
    bar=m.geom('crossbar').id;mat=m.geom('landing_mat').id
    ground=set(i for i in range(m.ngeom) if 'terrain' in (m.geom(i).name or '') or m.geom_type[i]==mujoco.mjtGeom.mjGEOM_PLANE)
    torso=m.body('robot/torso_link').id
    trace=dict(grounded_start=False,took_off=False,all_above_bar=True,all_crossed=False,between_posts=True,
               bar_contacts=0,torso_up_at_crossing=-1.,landed_on_mat=False,premature_mat_contact=False)
    max_joint_limit_violation=0.;max_speed_ratio=0.;minimum_clearance=float('inf')
    first_mat_geoms=None
    air_time=0.;max_air_time=0.;ground_sides=set();max_joint_speed=0.;torque_saturated=0;torque_samples=0
    seen_cross=False;rows=[];poses=[];crossed=set();last=np.zeros(29,'f');previous_x=d.xpos[torso,0]
    renderer=None;writer=None;frames=0
    if video:
        import imageio.v2 as imageio
        m.vis.global_.offwidth=1280;m.vis.global_.offheight=720
        renderer=mujoco.Renderer(m,height=720,width=1280)
        writer=imageio.get_writer(str(video),fps=50,codec='libx264',quality=8,macro_block_size=1)
        cam=mujoco.MjvCamera();cam.lookat[:]=[-.9,0,.75];cam.distance=4.3;cam.azimuth=100;cam.elevation=-14
    for k in range(len(z['time'])):
        if reference:
            d.qpos[:]=z['qpos'][k];d.qvel[:]=0;mujoco.mj_forward(m,d);act=np.zeros(29)
        else:
            obs=observation(m,d,z,k,contract,last)
            assert len(obs)==contract['actor_obs_dim']
            with torch.inference_mode():act=net(torch.from_numpy(obs).unsqueeze(0)).numpy()[0] if net else np.zeros(29)
            if sequence is not None:act=sequence[min(k,len(sequence)-1)]
            act=np.clip(act,-5,5);target=z['joint_pos'][k,aids]+contract['action_scale']*act
            if contract.get('clip_joint_targets'):target=np.clip(target,limits[aids,0]+.01,limits[aids,1]-.01)
            full_target=z['joint_pos'][k].copy();full_target[aids]=target
            if not dc:d.ctrl[ctrl_for_action]=target
            for sub in range(8):
                if dc:d.ctrl[ctrl_for_action]=torque_numpy(full_target,d.qpos[7:],d.qvel[6:],motor)[aids]
                mujoco.mj_step(m,d)
                if not np.isfinite(d.qpos).all():raise RuntimeError('non-finite dynamic state')
                pairs=[(int(c.geom1),int(c.geom2)) for c in d.contact]
                foot_ground=any((a in foot and b in ground) or (b in foot and a in ground) for a,b in pairs)
                if k<15:
                    for a,b in pairs:
                        g=a if a in foot and b in ground else (b if b in foot and a in ground else None)
                        if g is not None:ground_sides.add('left' if 'left' in m.geom(g).name else 'right')
                    trace['grounded_start']=len(ground_sides)==2
                any_ground=any((a in coll and b not in coll) or (b in coll and a not in coll) for a,b in pairs)
                air_time=air_time+m.opt.timestep if not any_ground else 0.
                max_air_time=max(max_air_time,air_time)
                if k>20 and air_time>=.05 and d.qpos[2]>.8:trace['took_off']=True
                max_joint_speed=max(max_joint_speed,float(np.max(np.abs(d.qvel[6:]))))
                max_joint_limit_violation=max(max_joint_limit_violation,float(np.max(np.maximum(limits[:,0]-d.qpos[7:],d.qpos[7:]-limits[:,1]))))
                max_speed_ratio=max(max_speed_ratio,float(np.max(np.abs(d.qvel[6:])/vmax)))
                torque_saturated+=int(np.sum(np.abs(d.actuator_force)>=.99*np.max(np.abs(m.actuator_forcerange),axis=1)))
                torque_samples+=m.nu
                trace['bar_contacts']+=sum(bar in (a,b) and (a in coll or b in coll) for a,b in pairs)
                lo,hi=collision_bounds(m,d,coll)
                spans=(lo[:,0]<=bar_x+BAR_RADIUS)&(hi[:,0]>=bar_x-BAR_RADIUS)
                if trace['all_crossed'] and spans.any():trace['post_clearance_return_under_bar']=bool(trace.get('post_clearance_return_under_bar',False) or (lo[spans,2]<=contract['bar_height_m']+BAR_RADIUS).any())
                if spans.any() and not trace['all_crossed']:
                    seen_cross=True
                    minimum_clearance=min(minimum_clearance,float(np.min(lo[spans,2]-contract['bar_height_m']-BAR_RADIUS)))
                    trace['all_above_bar'] &= bool((lo[spans,2]>contract['bar_height_m']+BAR_RADIUS).all())
                    trace['between_posts'] &= bool((lo[spans,1]>-BAR_HALF_WIDTH).all() and (hi[spans,1]<BAR_HALF_WIDTH).all())
                crossed.update(coll[hi[:,0]<bar_x-BAR_RADIUS].tolist())
                if (hi[:,0]<bar_x-BAR_RADIUS).all() and not trace['all_crossed']:
                    trace['first_full_clearance_time_s']=float(d.time)
                    trace['all_crossed']=True
                if previous_x>=bar_x and d.xpos[torso,0]<bar_x and 'torso_crossing_time_s' not in trace:
                    trace['torso_crossing_time_s']=float(d.time)
                    trace['torso_up_at_crossing']=float(d.xmat[torso].reshape(3,3)[2,0])
                previous_x=d.xpos[torso,0]
                mat_touch=any(mat in (a,b) and (a in coll or b in coll) for a,b in pairs)
                if mat_touch:
                    mat_robot_geoms=[m.geom(b if a==mat else a).name for a,b in pairs if mat in (a,b) and (a in coll or b in coll)]
                    back_touch=any(name in ['robot/pelvis_collision','robot/torso_collision'] for name in mat_robot_geoms)
                    if back_touch and 'first_back_support_time_s' not in trace:
                        trace['first_back_support_time_s']=float(d.time)
                        trace['back_support_torso_up']=float(d.xmat[torso].reshape(3,3)[2,0])
                    if any('head' in name for name in mat_robot_geoms) and 'first_back_support_time_s' not in trace:
                        trace['head_before_back_support']=True
                    else:trace.setdefault('head_before_back_support',False)
                    if first_mat_geoms is None:
                        first_mat_geoms=[m.geom(b if a==mat else a).name for a,b in pairs if mat in (a,b) and (a in coll or b in coll)]
                        trace['first_mat_contact_head']=any('head' in name for name in first_mat_geoms)
                        trace['first_mat_torso_up']=float(d.xmat[torso].reshape(3,3)[2,0])
                        trace['first_mat_contact_geoms']=first_mat_geoms
                        trace['first_mat_time_s']=float(d.time)
                    if trace['all_crossed']:trace['landed_on_mat']=True
                    else:trace['premature_mat_contact']=True
            last=act.astype('f')
        poses.append(d.qpos.copy());rows.append([d.time,d.qpos[0],d.qpos[2],np.max(np.abs(d.actuator_force))])
        if writer:
            renderer.update_scene(d,camera=cam);frame=renderer.render()
            from PIL import Image, ImageDraw
            im=Image.fromarray(frame);draw=ImageDraw.Draw(im);draw.rectangle((0,0,1280,55),fill=(15,18,25))
            label='KINEMATIC REFERENCE - NOT TRAINED POLICY' if reference else ('NATIVE MUJOCO / PPO POLICY' if net else ('NATIVE MUJOCO / CEM PHASE-PD POLICY' if sequence is not None else 'NATIVE MUJOCO / ZERO RESIDUAL BASELINE'))
            draw.text((18,12),f'{label} | BAR {contract["bar_height_m"]:.2f} m | t={k*.02:.2f}s | contacts={trace["bar_contacts"]}',fill='white')
            writer.append_data(np.asarray(im));frames+=1
    if writer:writer.close();renderer.close()
    trace['finished_on_landing_side']=bool(d.qpos[0]<bar_x-BAR_RADIUS and d.xpos[torso,0]<bar_x-BAR_RADIUS)
    trace['final_torso_up']=float(d.xmat[torso].reshape(3,3)[2,0])
    trace['all_above_bar'] &= seen_cross
    trace['max_joint_limit_violation_rad']=max_joint_limit_violation
    trace['max_joint_speed_ratio']=max_speed_ratio
    result={**assess(trace),**trace,'seed':seed,'bar_height_m':contract['bar_height_m'],'backend':'native MuJoCo CPU',
            'controller_sha256':__import__('hashlib').sha256(Path(action_sequence or policy).read_bytes()).hexdigest() if (action_sequence or policy) else None,
            'acceptance_sha256':__import__('hashlib').sha256((ROOT/'acceptance.py').read_bytes()).hexdigest(),
            'reference_only':reference,'policy':str(policy) if policy else action_sequence,'max_pelvis_height_m':float(np.max(np.asarray(poses)[:,2])),
            'final_root_xyz':poses[-1][:3].tolist(),'video_frames':frames,
            'minimum_clearance_m':minimum_clearance if seen_cross else None,'motor_model':contract.get('motor_model','builtin_position'),
            'max_air_time_s':max_air_time,'max_joint_speed_rad_s':max_joint_speed,'torque_saturation_fraction':torque_saturated/max(1,torque_samples)}
    if reference:result['success']=False;result['failed_requirements']=['reference_animation_not_dynamic_evaluation']
    return result,np.asarray(poses),np.asarray(rows)


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--policy');p.add_argument('--output',required=True)
    p.add_argument('--action-sequence');p.add_argument('--reference',action='store_true');p.add_argument('--video',action='store_true');p.add_argument('--seeds',type=int,default=1)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
    results=[]
    for seed in range(a.seeds):
        result,q,rows=rollout(Path(a.run),a.policy,seed,out/f'seed{seed}.mp4' if a.video and seed==0 else None,a.reference,a.action_sequence)
        np.savez(out/f'seed{seed}.npz',qpos=q,stats=rows);results.append(result)
    (out/'metrics.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))
if __name__=='__main__':main()
