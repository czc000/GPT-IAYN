from dataclasses import dataclass
import torch
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
from mjlab.envs.mdp.actions.actions import JointPositionAction, JointPositionActionCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from scene import ROOT, add_obstacles

class ResidualAction(JointPositionAction):
    def process_actions(self,actions):
        self._raw_actions[:]=actions
        cmd=self._env.command_manager.get_term('motion')
        self._processed_actions=cmd.joint_pos[:,self._target_ids]+actions*self._scale
        if self.cfg.clip_joint_targets:
            limits=self._entity.data.joint_pos_limits[:,self._target_ids]
            self._processed_actions=torch.clamp(self._processed_actions,limits[:,:,0]+.01,limits[:,:,1]-.01)

@dataclass(kw_only=True)
class ResidualActionCfg(JointPositionActionCfg):
    clip_joint_targets: bool = False
    def build(self,env):return ResidualAction(self,env)

def bar_penalty(env):
    return (env.scene['bar_contact'].data.found>0).float().sum(-1)

def clip_done(env):
    c=env.command_manager.get_term('motion')
    return c.time_steps>=c.motion.time_step_total-2

def config(n=2048,height=.8,play=False,start_only=False,dc=False,bar_x=-.7,mat_top=.18,mat_front=-.95):
    from training_compat import configure_training_runtime
    configure_training_runtime()
    c=unitree_g1_flat_tracking_env_cfg(play=play)
    c.scene.num_envs=n
    c.scene.env_spacing=0.0  # independent Warp worlds, same local obstacle coordinates
    c.scene.spec_fn=lambda s:add_obstacles(s,height,mat_top=mat_top,bar_x=bar_x,mat_front=mat_front)
    if dc:
        from motor import dc_robot_cfg,strict_limits
        c.scene.entities['robot']=dc_robot_cfg()
        def scene_setup(s):
            add_obstacles(s,height,mat_top=mat_top,bar_x=bar_x,mat_front=mat_front);strict_limits(s)
        c.scene.spec_fn=scene_setup
    c.scene.sensors=c.scene.sensors+(ContactSensorCfg(name='bar_contact',
        primary=ContactMatch(mode='subtree',pattern='pelvis',entity='robot'),
        secondary=ContactMatch(mode='geom',pattern='crossbar'),fields=('found',),reduce='none'),
        ContactSensorCfg(name='back_mat_contact',primary=ContactMatch(mode='geom',pattern='(pelvis|torso)_collision',entity='robot'),secondary=ContactMatch(mode='geom',pattern='landing_mat'),fields=('found',),reduce='none'),
        ContactSensorCfg(name='head_mat_contact',primary=ContactMatch(mode='geom',pattern='head_collision',entity='robot'),secondary=ContactMatch(mode='geom',pattern='landing_mat'),fields=('found',),reduce='none'),
        ContactSensorCfg(name='mat_contact',primary=ContactMatch(mode='subtree',pattern='pelvis',entity='robot'),secondary=ContactMatch(mode='geom',pattern='landing_mat'),fields=('found',),reduce='none'))
    c.commands['motion'].motion_file=str(ROOT/'artifacts/reference/motion.npz')
    c.commands['motion'].pose_range={}
    c.commands['motion'].velocity_range={}
    c.commands['motion'].joint_position_range=(-.02,.02) if not play else (0.,0.)
    c.commands['motion'].sampling_mode='start' if (start_only or play) else 'uniform'
    c.actions['joint_pos']=ResidualActionCfg(entity_name='robot',actuator_names=('.*',),scale=.5,use_default_offset=False,clip_joint_targets=dc)
    c.events={}
    c.observations['actor'].enable_corruption=False
    c.rewards['motion_global_root_pos'].weight=3.
    c.rewards['motion_global_root_pos'].params['std']=.4
    c.rewards['motion_global_root_ori'].weight=2.
    c.rewards['motion_global_root_ori'].params['std']=.5
    c.rewards['motion_body_pos'].weight=2.
    c.rewards['motion_body_lin_vel'].weight=.5
    c.rewards['motion_body_ang_vel'].weight=.5
    c.rewards['action_rate_l2'].weight=-.01
    c.rewards['self_collisions'].weight=-.2
    c.rewards['bar_touch']=RewardTermCfg(func=bar_penalty,weight=-3.)
    c.terminations['anchor_pos'].params['threshold']=.45
    c.terminations['ee_body_pos'].params['threshold']=.5
    c.terminations['anchor_ori'].params['threshold']=1.3
    c.terminations['clip_done']=TerminationTermCfg(func=clip_done,time_out=True)
    c.episode_length_s=2.62
    c.sim.nconmax=96; c.sim.njmax=500
    c.sim.mujoco.iterations=20
    c.sim.mujoco.timestep=.0025; c.decimation=8
    return c
