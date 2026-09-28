"""Reference-conditioned state-feedback PPO; no actor world position/linear velocity."""
from dataclasses import dataclass,fields
import numpy as np
import torch
from mjlab.managers import ObservationTermCfg, ObservationGroupCfg, RewardTermCfg
from mjlab.envs.mdp.actions.actions import JointPositionAction,JointPositionActionCfg
from mjlab.utils.lab_api.math import matrix_from_quat,quat_error_magnitude
from scene import ROOT
from task import config as base_config

from mjlab.tasks.tracking.mdp.commands import MotionCommand,MotionCommandCfg
class AlignedMotionCommand(MotionCommand):
    def _update_command(self):
        # The manager computes commands after automatic resets but not after explicit
        # env.reset(). Preserve frame zero in those newly reset environments.
        self.time_steps[self._env.episode_length_buf==0]-=1
        super()._update_command()

@dataclass(kw_only=True)
class AlignedMotionCommandCfg(MotionCommandCfg):
    def build(self,env):return AlignedMotionCommand(self,env)

SCALE=.25
OBS_DIM=152

def reference(env):
    if not hasattr(env,'_jump_ref'):
        z=np.load(env.command_manager.get_term('motion').cfg.motion_file)
        env._jump_ref={k:torch.tensor(z[k],device=env.device,dtype=torch.float32) for k in ['target','joint_pos','joint_vel','body_pos_w','body_quat_w','gravity','gyro']}
    return env._jump_ref

def actor_observation(env):
    cmd=env.command_manager.get_term('motion');k=cmd.time_steps;ref=reference(env);r=env.scene['robot']
    torso=r.body_names.index('pelvis')
    rot=matrix_from_quat(r.data.body_link_quat_w[:,torso])
    gravity=-rot[:,2,:]
    gyro=env.scene['robot/imu_ang_vel'].data
    return torch.cat([ref['target'][k],ref['joint_pos'][k],ref['joint_vel'][k]*.05,
        r.data.joint_pos-ref['joint_pos'][k],r.data.joint_vel*.05,gravity,gyro*.1,
        k[:,None].float()/130.],dim=-1)
    # 29*5 + 3 + 3 + 1 = 152. No global location, velocity, contacts or future simulator states.

class FeedbackAction(JointPositionAction):
    def process_actions(self,actions):
        self._raw_actions[:]=actions
        k=self._env.command_manager.get_term('motion').time_steps
        target=reference(self._env)['target'][k]+actions*SCALE
        limits=self._entity.data.joint_pos_limits
        self._processed_actions=torch.clamp(target,limits[:,:,0]+.01,limits[:,:,1]-.01)

@dataclass(kw_only=True)
class FeedbackActionCfg(JointPositionActionCfg):
    def build(self,env):return FeedbackAction(self,env)

def tracking(env,kind):
    cmd=env.command_manager.get_term('motion');k=(cmd.time_steps+1).clamp(max=130);z=reference(env);r=env.scene['robot']
    if kind=='joint':err=((r.data.joint_pos-z['joint_pos'][k])**2).mean(-1);return torch.exp(-err/.04)
    if kind=='velocity':err=((r.data.joint_vel-z['joint_vel'][k])**2).mean(-1);return torch.exp(-err/16.)
    if kind=='position':err=((r.data.body_link_pos_w-z['body_pos_w'][k])**2).sum(-1).mean(-1);return torch.exp(-err/.025)
    if kind=='orientation':err=quat_error_magnitude(r.data.body_link_quat_w,z['body_quat_w'][k])**2;return torch.exp(-err.mean(-1)/.10)
    if kind=='residual':return env.action_manager.action.square().mean(-1)
    if kind=='head':return (env.scene['head_mat_contact'].data.found>0).float().sum(-1)*(k<82)

def config(n=2048,play=False,perturb=.02,motion_file=None):
    c=base_config(n,play=play,start_only=True,dc=True)
    original=c.commands['motion']
    c.commands['motion']=AlignedMotionCommandCfg(**{f.name:getattr(original,f.name) for f in fields(original)})
    c.commands['motion'].motion_file=str(motion_file or ROOT/'artifacts/rl-reference/motion.npz')
    c.commands['motion'].joint_position_range=(-perturb,perturb) if not play else (0.,0.)
    c.commands['motion'].pose_range={'roll':(-perturb,perturb),'pitch':(-perturb,perturb)} if not play else {}
    c.commands['motion'].velocity_range={'x':(-perturb*3,perturb*3),'y':(-perturb*3,perturb*3)} if not play else {}
    c.actions['joint_pos']=FeedbackActionCfg(entity_name='robot',actuator_names=('.*',),scale=SCALE,use_default_offset=False)
    c.observations['actor']=ObservationGroupCfg(terms={'proprioception':ObservationTermCfg(func=actor_observation)},concatenate_terms=True,enable_corruption=False)
    c.rewards={kind:RewardTermCfg(func=tracking,weight=w,params={'kind':kind}) for kind,w in [('joint',3.),('velocity',.5),('position',5.),('orientation',3.),('residual',-.03),('head',-2.)]}
    from task import bar_penalty
    c.rewards['bar']=RewardTermCfg(func=bar_penalty,weight=-5.)
    # Training failures reset; native evaluation never resets within an episode.
    c.terminations.pop('ee_body_pos',None)
    c.terminations['anchor_pos'].params['threshold']=.6
    c.terminations['anchor_ori'].params['threshold']=1.5
    return c
