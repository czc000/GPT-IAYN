"""Constant deployment gain candidates for the learned PPO correction; no phase-curve optimization."""
import torch,numpy as np,json
from pathlib import Path
from gpt_iayn_core.scene import ROOT
class Gain(torch.nn.Module):
 def __init__(self,net,gain,limits):
  super().__init__();self.net=net;self.gain=gain;self.register_buffer('lo',limits[:,0]+.01);self.register_buffer('hi',limits[:,1]-.01)
 def forward(self,o):return torch.clamp(o[:,:29]+self.gain*(self.net(o)-o[:,:29]),self.lo,self.hi)
if __name__=='__main__':
 from gpt_iayn_core.evaluate import load_native_model
 out=ROOT/'artifacts/rl-gain-calibration';out.mkdir(exist_ok=True);src=ROOT/'artifacts/final-rl-feedback'
 c=json.loads((src/'contract.json').read_text());m=load_native_model(src,c);limits=torch.tensor(m.jnt_range[1:],dtype=torch.float32)
 for gain in [.5,2.,4.]:
  module=Gain(torch.jit.load(str(src/'policy.torchscript')),gain,limits).eval()
  torch.jit.trace(module,torch.zeros(1,152)).save(str(out/f'gain{gain:g}.torchscript'))
