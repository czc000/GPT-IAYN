"""Portable absolute-joint-target export contract."""
import torch
SCALE=.25

class TargetActor(torch.nn.Module):
    def __init__(self,actor,normalizer,limits):
        super().__init__();self.actor=actor;self.normalizer=normalizer
        self.register_buffer('lower',limits[:,0]+.01);self.register_buffer('upper',limits[:,1]-.01)
    def forward(self,obs):
        residual=torch.clamp(self.actor(self.normalizer(obs)),-2.,2.)
        return torch.clamp(obs[:,:29]+SCALE*residual,self.lower,self.upper)
