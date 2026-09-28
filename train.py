import argparse, copy, json, os
from pathlib import Path
from dataclasses import asdict
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.tracking.config.g1.rl_cfg import unitree_g1_tracking_ppo_runner_cfg
from rsl_rl.runners import OnPolicyRunner
from task import config
from scene import ROOT

class ExportRunner(OnPolicyRunner):
    def save(self,path,infos=None):
        super().save(path,infos)
        module=torch.nn.Sequential(copy.deepcopy(self.obs_normalizer),copy.deepcopy(self.alg.policy.actor)).cpu().eval()
        obs=self.env.get_observations()['actor'][:1].cpu()
        export=Path(path).with_suffix('.torchscript')
        torch.jit.trace(module,obs).save(str(export))
        print('EXPORTED',str(export),flush=True)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--num-envs',type=int,default=2048);p.add_argument('--iterations',type=int,default=5000)
    p.add_argument('--height',type=float,default=.8);p.add_argument('--name',default='run01');p.add_argument('--resume');p.add_argument('--start-only',action='store_true')
    a=p.parse_args(); torch.set_num_threads(4);torch.manual_seed(42)
    c=config(a.num_envs,a.height,start_only=a.start_only)
    out=ROOT/'artifacts'/a.name;out.mkdir(parents=True,exist_ok=True)
    (out/'launch.json').write_text(json.dumps(vars(a),indent=2))
    env=ManagerBasedRlEnv(c,device='cuda:0');env.scene.write(out/'scene')
    v=RslRlVecEnvWrapper(env,clip_actions=5.)
    action=env.action_manager.get_term('joint_pos')
    contract=dict(joint_names=list(env.scene['robot'].joint_names),body_names=list(env.scene['robot'].body_names),
        action_joint_ids=action.target_ids.cpu().tolist(),action_joint_names=action.target_names,
        action_scale=.5,action_type='reference_joint_position_plus_scaled_residual',control_dt=env.step_dt,
        default_joint_pos=env.scene['robot'].data.default_joint_pos[0].cpu().tolist(),
        actor_obs_dim=v.get_observations()['actor'].shape[-1],clip_actions=5.,bar_height_m=a.height,
        motion_file=str(ROOT/'artifacts/reference/motion.npz'),backend='MuJoCo-Warp',physics=asdict(c.sim.mujoco))
    (out/'contract.json').write_text(json.dumps(contract,indent=2))
    r=unitree_g1_tracking_ppo_runner_cfg();r.logger='tensorboard';r.save_interval=100;r.max_iterations=a.iterations
    r.actor.hidden_dims=(256,256,128);r.critic.hidden_dims=(256,256,128);r.actor.distribution_cfg['init_std']=.5
    r.algorithm.num_learning_epochs=4;r.algorithm.learning_rate=3e-4
    rc=asdict(r);(out/'agent.json').write_text(json.dumps(rc,indent=2))
    runner=ExportRunner(v,rc,str(out),device='cuda:0')
    if a.resume:runner.load(a.resume)
    runner.learn(a.iterations,init_at_random_ep_len=False)
    env.close()
if __name__=='__main__':main()
