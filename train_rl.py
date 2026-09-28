import argparse,copy,json
from dataclasses import asdict
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.tracking.config.g1.rl_cfg import unitree_g1_tracking_ppo_runner_cfg
from rsl_rl.runners import OnPolicyRunner
from rl_task import config,SCALE
from scene import ROOT

from policy_export import TargetActor

class Runner(OnPolicyRunner):
    def save(self,path,infos=None):
        super().save(path,infos)
        limits=self.env.unwrapped.scene['robot'].data.joint_pos_limits[0].cpu()
        module=TargetActor(copy.deepcopy(self.alg.policy.actor).cpu(),copy.deepcopy(self.obs_normalizer).cpu(),limits).eval()
        from pathlib import Path
        torch.jit.trace(module,torch.zeros(1,152)).save(str(Path(path).with_suffix('.torchscript')))

def main():
    p=argparse.ArgumentParser();p.add_argument('--name',default='rl01');p.add_argument('--num-envs',type=int,default=2048);p.add_argument('--iterations',type=int,default=1000);p.add_argument('--perturb',type=float,default=.02);p.add_argument('--resume');p.add_argument('--normalize',action='store_true');p.add_argument('--feedback-only',action='store_true');p.add_argument('--reference',default=str(ROOT/'artifacts/rl-reference/motion.npz'));p.add_argument('--teacher',default=str(ROOT/'artifacts/final'));a=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(42)
    out=ROOT/'artifacts'/a.name;out.mkdir(parents=True,exist_ok=True)
    c=config(a.num_envs,perturb=a.perturb,motion_file=a.reference);env=ManagerBasedRlEnv(c,device='cuda:0');env.scene.write(out/'scene')
    v=RslRlVecEnvWrapper(env,clip_actions=2.)
    contract=json.loads((__import__('pathlib').Path(a.teacher)/'contract.json').read_text())
    contract.update(motion_file=str(__import__('pathlib').Path(a.reference).resolve()),actor_obs_dim=152,action_scale=SCALE,
        actor_output='absolute_joint_position_target_rad',actor_inputs=['reference_target:29','reference_q:29','reference_dq_scaled:29','joint_error:29','joint_velocity_scaled:29','pelvis_projected_gravity:3','pelvis_gyro_scaled:3','phase:1'],
        backend='MuJoCo-Warp',physics=asdict(c.sim.mujoco),training=vars(a))
    (out/'contract.json').write_text(json.dumps(contract,indent=2))
    r=unitree_g1_tracking_ppo_runner_cfg();r.logger='tensorboard';r.save_interval=50;r.num_steps_per_env=32
    r.actor.hidden_dims=(256,256,128);r.critic.hidden_dims=(256,256,128)
    r.actor.obs_normalization=a.normalize;r.critic.obs_normalization=a.normalize
    r.actor.distribution_cfg['init_std']=.08
    r.algorithm.num_learning_epochs=4;r.algorithm.learning_rate=1e-4;r.algorithm.entropy_coef=0.;r.algorithm.schedule='fixed'
    rc=asdict(r);(out/'agent.json').write_text(json.dumps(rc,indent=2))
    runner=Runner(v,rc,str(out),device='cuda:0')
    torch.nn.init.zeros_(runner.alg.policy.actor[-1].weight);torch.nn.init.zeros_(runner.alg.policy.actor[-1].bias)
    if a.feedback_only:
        assert not a.normalize, 'FeedbackActor uses an explicit raw observation contract and internal scaling'
        from feedback_actor import FeedbackActor
        z=np.load(a.reference)
        runner.alg.policy.actor=FeedbackActor(runner.alg.policy.actor,z['gravity'],z['gyro']).to('cuda:0')
    if a.resume:runner.load(a.resume)
    runner.logger_type='tensorboard'
    runner.save(str(out/'initial.pt'))
    runner.learn(a.iterations,init_at_random_ep_len=False);env.close()
if __name__=='__main__':main()
