"""Freeze deployable artifacts separately from the former time-curve baseline."""
import json,shutil,hashlib,argparse
from pathlib import Path
import numpy as np,torch
from gpt_iayn_core.scene import ROOT
p=argparse.ArgumentParser();p.add_argument('--gain',type=float,default=1.);p.add_argument('--output',default='artifacts/final-rl-feedback');args=p.parse_args()
source=ROOT/'artifacts/rl03-feedback-aligned';out=ROOT/args.output;out.mkdir(exist_ok=True)
policy_source=source/'model_499.torchscript' if args.gain==1 else ROOT/f'artifacts/rl-gain-calibration/gain{args.gain:g}.torchscript'
shutil.copy2(policy_source,out/'policy.torchscript');shutil.copy2(source/'model_499.pt',out/'training-checkpoint.pt')
shutil.copytree(source/'scene',out/'scene',dirs_exist_ok=True)
shutil.copy2(ROOT/'artifacts/rl-reference/motion.npz',out/'evaluation-reference.npz')
z=np.load(out/'evaluation-reference.npz');np.savez(out/'commands.npz',**{k:z[k] for k in ['target','joint_pos','joint_vel','time']})
c=json.loads((source/'contract.json').read_text());c['motion_file']=str(out/'evaluation-reference.npz');c['deployment_feedback_gain']=args.gain;c['action_scale']=.25*args.gain;c['algorithm']='PPO with dynamically feasible imitation rewards and reference-conditioned differential state-feedback actor'
c['actor_state_feedback']='MLP(actual proprioception) - MLP(reference proprioception); reference-state correction is zero to floating-point precision'
c['actor_output']='29 absolute joint target angles, radians';c['deployment_commands_file']='commands.npz';c['policy_file']='policy.torchscript'
(out/'contract.json').write_text(json.dumps(c,indent=2));shutil.copy2(ROOT/'docs/RL_METHOD.md',out/'METHOD.md')
net=torch.jit.load(str(out/'policy.torchscript')).cpu().eval();sample=torch.tensor(np.load(ROOT/'artifacts/rl03-final-validation/seed0.npz')['observations'][:1])
torch.onnx.export(net,sample,str(out/'policy.onnx'),input_names=['observation'],output_names=['joint_target'],opset_version=17,dynamo=False)
import onnx
onnx.checker.check_model(str(out/'policy.onnx'))
manifest=dict(deployment_feedback_gain=args.gain,zero_shot_transfer=False if args.gain!=1 else None,training_run=str(source),checkpoint='model_499.pt',iterations=500,environment_steps=32768000,
 policy_sha256=hashlib.sha256((out/'policy.torchscript').read_bytes()).hexdigest(),onnx_sha256=hashlib.sha256((out/'policy.onnx').read_bytes()).hexdigest(),
 training_backend='MuJoCo-Warp',primary_evaluation_backend='native MuJoCo CPU',real_robot_tested=False)
(out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2))
