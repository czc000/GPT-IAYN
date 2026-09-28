"""PPO actor constrained to learn proprioceptive corrections, not another phase curve."""
import torch
class FeedbackActor(torch.nn.Module):
    def __init__(self,network,gravity,gyro):
        super().__init__();self.network=network
        self.register_buffer('reference_gravity',torch.as_tensor(gravity,dtype=torch.float32))
        self.register_buffer('reference_gyro',torch.as_tensor(gyro,dtype=torch.float32))
        scale=torch.ones(152);scale[87:116]=20.;scale[116:145]=2.;scale[145:148]=5.;scale[148:151]=5.
        self.register_buffer('feature_scale',scale)
    def forward(self,obs):
        phase=torch.clamp(torch.round(obs[:,-1]*130).long(),0,130)
        nominal=obs.clone()
        nominal[:,87:116]=0.
        nominal[:,116:145]=obs[:,58:87]
        nominal[:,145:148]=self.reference_gravity[phase]
        nominal[:,148:151]=self.reference_gyro[phase]*.1
        # Exact zero correction at the reference state. PPO cannot improve solely by
        # producing another time-only target curve; its mean action must use errors.
        return self.network(obs*self.feature_scale)-self.network(nominal*self.feature_scale)
