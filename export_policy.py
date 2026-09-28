"""Export any trained run with an explicit, constant deployment feedback gain."""
import argparse,copy,hashlib,json,shutil
from pathlib import Path
import numpy as np
import torch
from portable import motion_path

class Gain(torch.nn.Module):
    def __init__(self,network,gain):
        super().__init__();self.network=network;self.gain=gain
    def forward(self,observation):
        # Interpolation between two already limited targets preserves the limits.
        return observation[:,:29]+self.gain*(self.network(observation)-observation[:,:29])

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);p.add_argument('--checkpoint',default='model_499');p.add_argument('--gain',type=float,default=.5);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    if not 0<=a.gain<=1:raise ValueError('This exporter supports feedback interpolation gains in [0,1]')
    if a.output.exists():raise FileExistsError('Choose a new export directory')
    contract=json.loads((a.run/'contract.json').read_text())
    net=torch.jit.load(str(a.run/(a.checkpoint+'.torchscript')),map_location='cpu').eval()
    reference=motion_path(a.run,contract)
    z=np.load(reference)
    obs=np.concatenate([z['target'][0],z['joint_pos'][0],z['joint_vel'][0]*.05,np.zeros(29),z['joint_vel'][0]*.05,z['gravity'][0],z['gyro'][0]*.1,[0]]).astype('float32')[None]
    sample=torch.from_numpy(obs)
    module=Gain(net,a.gain).eval()
    traced=torch.jit.trace(module,sample)
    a.output.mkdir(parents=True)
    traced.save(str(a.output/'policy.torchscript'))
    torch.onnx.export(traced,sample,str(a.output/'policy.onnx'),input_names=['observation'],output_names=['joint_target'],opset_version=17,dynamo=False)
    import onnx
    onnx.checker.check_model(str(a.output/'policy.onnx'))
    shutil.copytree(a.run/'scene',a.output/'scene')
    shutil.copy2(reference,a.output/'evaluation-reference.npz')
    np.savez(a.output/'commands.npz',**{k:z[k] for k in ['target','joint_pos','joint_vel','time']})
    checkpoint=a.run/(a.checkpoint+'.pt')
    if checkpoint.exists():shutil.copy2(checkpoint,a.output/'training-checkpoint.pt')
    contract.update(motion_file='evaluation-reference.npz',policy_file='policy.torchscript',deployment_commands_file='commands.npz',deployment_feedback_gain=a.gain,action_scale=.25*a.gain)
    (a.output/'contract.json').write_text(json.dumps(contract,indent=2))
    report={'checkpoint':a.checkpoint,'gain':a.gain,'onnx_checked':True,'reference_frames':len(z['time']),'real_robot_tested':False,'cross_engine_validated':False,'policy_sha256':hashlib.sha256((a.output/'policy.torchscript').read_bytes()).hexdigest()}
    (a.output/'manifest.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
