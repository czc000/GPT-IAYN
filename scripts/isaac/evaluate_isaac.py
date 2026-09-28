"""Independent PhysX execution of a frozen MuJoCo-trained actor (diagnostic until audited)."""
import argparse
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--run',default='artifacts/final-rl-sim2sim');p.add_argument('--policy',required=True);p.add_argument('--output',required=True);p.add_argument('--video',action='store_true');p.add_argument('--physics-device');AppLauncher.add_app_launcher_args(p);args=p.parse_args()
launcher=AppLauncher(args,multi_gpu=False);app=launcher.app
import json,numpy as np,torch
from pathlib import Path
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation,ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils.math import quat_rotate_inverse
from gpt_iayn_core.paths import ROOT as root
from gpt_iayn_core.portable import motion_path, video_font
run=Path(args.run)
c=json.loads((run/'contract.json').read_text());z=np.load(motion_path(run,c));device=args.physics_device or args.device
sim=sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=.0025,device=device,gravity=(0,0,-9.81)))
import carb
carb.settings.get_settings().set('/physics/disableContactProcessing',False)
sim.set_camera_view([2.5,-3.5,1.5],[-.8,0,.7])
mat=sim_utils.RigidBodyMaterialCfg(static_friction=.8,dynamic_friction=.8,restitution=0.)
ground=sim_utils.GroundPlaneCfg(physics_material=mat);ground.func('/World/Ground',ground)
light=sim_utils.DomeLightCfg(intensity=2000);light.func('/World/Light',light)
bar=sim_utils.CapsuleCfg(radius=.012,height=2.,axis='Y',collision_props=sim_utils.CollisionPropertiesCfg(),physics_material=mat,visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1,.25,.05)))
bar.func('/World/Bar',bar,translation=(-.7,0,.8))
pad=sim_utils.CuboidCfg(size=(2.,2.1,.18),collision_props=sim_utils.CollisionPropertiesCfg(),physics_material=mat,visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.1,.3,.8)))
pad.func('/World/Mat',pad,translation=(-1.95,0,.09))
metadata=json.loads((root/'artifacts/rl-isaac/native-model.json').read_text())
robot=Articulation(ArticulationCfg(prim_path='/World/Robot',spawn=sim_utils.UsdFileCfg(usd_path=str(root/'artifacts/rl-isaac/usd/g1.usd'),
    rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False,linear_damping=0.,angular_damping=0.,max_depenetration_velocity=10.),
    articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=True,solver_position_iteration_count=16,solver_velocity_iteration_count=4)),
    actuators={'effort':ImplicitActuatorCfg(joint_names_expr=['.*'],stiffness=0.,damping=0.,effort_limit_sim=200.,velocity_limit_sim=100.,armature=metadata['joint_armature'])},soft_joint_pos_limit_factor=1.))
from pxr import UsdPhysics,PhysxSchema,PhysicsSchemaTools
from omni.physx import get_physx_simulation_interface
import omni.usd
stage=omni.usd.get_context().get_stage()
rigid={prim.GetName():prim for prim in stage.Traverse() if str(prim.GetPath()).startswith('/World/Robot') and prim.HasAPI(UsdPhysics.RigidBodyAPI)}
for first,second in [('pelvis','left_hip_roll_link'),('pelvis','right_hip_roll_link'),('left_elbow_link','left_wrist_pitch_link'),('right_elbow_link','right_wrist_pitch_link')]:
    UsdPhysics.FilteredPairsAPI.Apply(rigid[first]).CreateFilteredPairsRel().AddTarget(rigid[second].GetPath())
contact_log=[];physics_step=0
for prim in rigid.values():PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr().Set(0.)
def on_contact(headers,data):
    for h in headers:
        if h.num_contact_data==0:continue
        a=str(PhysicsSchemaTools.intToSdfPath(h.collider0));b=str(PhysicsSchemaTools.intToSdfPath(h.collider1))
        if a.startswith('/World/Robot')==b.startswith('/World/Robot'):continue
        pts=data[h.contact_data_offset:h.contact_data_offset+h.num_contact_data]
        contact_log.append(dict(step=physics_step,a=a,b=b,separation=min(float(pt.separation) for pt in pts),impulse=max(float(np.linalg.norm(pt.impulse)) for pt in pts)))
contact_subscription=get_physx_simulation_interface().subscribe_contact_report_events(on_contact)
colliders=[]
for prim in stage.Traverse():
    if str(prim.GetPath()).startswith('/World/Robot') and prim.HasAPI(UsdPhysics.CollisionAPI):
        api=UsdPhysics.CollisionAPI(prim)
        colliders.append(dict(path=str(prim.GetPath()),type=prim.GetTypeName(),enabled=api.GetCollisionEnabledAttr().Get()))
(root/'artifacts/rl-isaac/colliders.json').write_text(json.dumps(colliders,indent=2))
camera=None;writer=None
if args.video:
    from isaaclab.sensors import Camera,CameraCfg
    camera=Camera(CameraCfg(prim_path='/World/Camera',height=720,width=1280,data_types=['rgb'],spawn=sim_utils.PinholeCameraCfg(focal_length=24.,horizontal_aperture=24.,clipping_range=(.05,100.))))
for index,y in enumerate([-1.10,1.10]):
    post=sim_utils.CuboidCfg(size=(.07,.07,1.2),collision_props=sim_utils.CollisionPropertiesCfg(),visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.75,.8,.85)))
    post.func(f'/World/Post{index}',post,translation=(-.7,y,.6))
    base=sim_utils.CuboidCfg(size=(.6,.3,.05),collision_props=sim_utils.CollisionPropertiesCfg(),visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.25,.25,.3)))
    base.func(f'/World/Base{index}',base,translation=(-.7,y,.025))
sim.reset();print('MASSES',robot.root_physx_view.get_masses(),flush=True);print('JOINTS',robot.joint_names,flush=True);print('BODIES',robot.body_names,flush=True)
ids,names=robot.find_joints(c['joint_names'],preserve_order=True);assert names==c['joint_names']
initial=torch.tensor(z['qpos'][0],dtype=torch.float32,device=device)
root_name=robot.body_names[0];root_index=list(z['body_names']).index(root_name)
root_pose=np.concatenate([z['body_pos_w'][0,root_index],z['body_quat_w'][0,root_index]])
robot.write_root_pose_to_sim(torch.tensor(root_pose,dtype=torch.float32,device=device)[None]);robot.write_root_velocity_to_sim(torch.zeros(1,6,device=device))
q=robot.data.default_joint_pos.clone();q[:,ids]=initial[7:];robot.write_joint_state_to_sim(q,torch.zeros_like(q));robot.reset();sim.forward();robot.update(.0025)
reference_body_ids=[list(z['body_names']).index(n) for n in robot.body_names]
initial_body_error=float(np.max(np.abs(robot.data.body_pos_w[0].cpu().numpy()-z['body_pos_w'][0,reference_body_ids])))
print('INITIAL_BODY_POSITION_ERROR',initial_body_error,flush=True)
if camera is not None:
    import subprocess
    from PIL import Image,ImageDraw,ImageFont
    camera.set_world_poses_from_view(torch.tensor([[1.5,-4.0,1.5]],device=device),torch.tensor([[-.9,0,.75]],device=device))
    for _ in range(5):sim.render()
    Path(args.output).mkdir(parents=True,exist_ok=True)
    writer=subprocess.Popen(['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','1280x720','-r','50','-i','-','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p',str(Path(args.output)/'online-physx.mp4')],stdin=subprocess.PIPE)
mp={k:torch.tensor(v,device=device) for k,v in c['motor_parameters'].items()}
net=torch.jit.load(args.policy,map_location=device).eval();reference={k:torch.tensor(z[k],dtype=torch.float32,device=device) for k in ['target','joint_pos','joint_vel']}
poses=[];targets=[];observations=[];substep_poses=[];substep_velocities=[];pelvis_id=robot.body_names.index('pelvis')
for k in range(131):
    pelvis_quat=robot.data.body_quat_w[:,pelvis_id]
    gravity=quat_rotate_inverse(pelvis_quat,torch.tensor([[0.,0.,-1.]],device=device))[0]
    gyro=quat_rotate_inverse(pelvis_quat,robot.data.body_ang_vel_w[:,pelvis_id])[0]
    obs=torch.cat([reference['target'][k],reference['joint_pos'][k],reference['joint_vel'][k]*.05,
        robot.data.joint_pos[0,ids]-reference['joint_pos'][k],robot.data.joint_vel[0,ids]*.05,gravity,gyro*.1,torch.tensor([k/130.],device=device)])
    observations.append(obs.cpu().numpy().copy())
    with torch.inference_mode():target=net(obs[None])[0]
    for sub in range(8):
        physics_step=k*8+sub+1
        vel=robot.data.joint_vel[0,ids];vmax=mp['velocity'];effort=mp['effort'];bounded=vel.clamp(-2*vmax,2*vmax)
        upper=torch.minimum(effort*(1-bounded/vmax),effort);lower=torch.maximum(effort*(-1-bounded/vmax),-effort)
        tau=torch.clamp(mp['kp']*(target-robot.data.joint_pos[0,ids])-mp['kd']*vel,lower,upper)
        robot.set_joint_effort_target(tau[None],joint_ids=ids);robot.write_data_to_sim();sim.step(render=False);robot.update(.0025)
        substep_poses.append(torch.cat([robot.data.body_pos_w[0,pelvis_id],robot.data.body_quat_w[0,pelvis_id],robot.data.joint_pos[0,ids]]).cpu().numpy())
        substep_velocities.append(robot.data.joint_vel[0,ids].cpu().numpy().copy())
    poses.append(torch.cat([robot.data.body_pos_w[0,pelvis_id],robot.data.body_quat_w[0,pelvis_id],robot.data.joint_pos[0,ids]]).cpu().numpy());targets.append(target.cpu().numpy())
    if camera is not None:
        sim.render();camera.update(.02,force_recompute=True)
        frame=camera.data.output['rgb'][0].cpu().numpy()[:,:,:3]
        im=Image.fromarray(frame);draw=ImageDraw.Draw(im);draw.rectangle((0,0,1280,52),fill=(15,18,25))
        bar_hits=sum('/World/Bar' in (r['a']+r['b']) and r['impulse']>1e-6 for r in contact_log)
        draw.text((15,14),f'FROZEN PPO / ONLINE PHYSX DYNAMICS | BAR 0.80 m | t={(k+1)*.02:.2f}s | bar contacts {bar_hits}',fill='white',font=video_font(19))
        writer.stdin.write(np.asarray(im).tobytes())
    if k%10==0:print('STEP',k,'ROOT',poses[-1][:3],flush=True)
if writer is not None:
    writer.stdin.close();assert writer.wait()==0
out=Path(args.output);out.mkdir(parents=True,exist_ok=True);np.savez(out/'trajectory.npz',qpos=np.array(poses),targets=np.array(targets),observations=np.array(observations),substep_qpos=np.array(substep_poses),substep_joint_vel=np.array(substep_velocities))
(out/'contacts.json').write_text(json.dumps(contact_log,indent=2))
(out/'manifest.json').write_text(json.dumps(dict(backend='Isaac Sim 4.5 / PhysX',policy=args.policy,steps=131,dt=.0025,control_dt=.02,success_not_yet_audited=True,mat_model='rigid PhysX box, different from MuJoCo soft contact'),indent=2))
print('COMPLETE',out,flush=True);app.close()
