"""Robust CEM policy training with physical bar contacts and geometric clearance."""
import argparse,json,time
import mujoco,numpy as np,torch
from mjlab.envs import ManagerBasedRlEnv
from scene import ROOT,BAR_X,BAR_HEIGHT,BAR_RADIUS
from task import config

class Bounds:
    def __init__(self,m,device):
        ids=[i for i in range(m.ngeom) if (m.geom(i).name or '').startswith('robot/') and (m.geom_contype[i] or m.geom_conaffinity[i])]
        self.ids=torch.tensor(ids,device=device)
        self.size=torch.tensor(m.geom_size[ids],device=device,dtype=torch.float32)
        self.typ=torch.tensor(m.geom_type[ids],device=device)
        self.aabb=torch.tensor(m.geom_aabb[ids],device=device,dtype=torch.float32)
    def get(self,env):
        p=env.sim.data.geom_xpos[:,self.ids];r=env.sim.data.geom_xmat[:,self.ids].reshape(p.shape[0],-1,3,3)
        ext=(r.abs()@self.size[None,:,:,None]).squeeze(-1)
        capsule=self.typ==int(mujoco.mjtGeom.mjGEOM_CAPSULE)
        sphere=self.typ==int(mujoco.mjtGeom.mjGEOM_SPHERE)
        mesh=self.typ==int(mujoco.mjtGeom.mjGEOM_MESH)
        cap=r[:,:,:,2].abs()*self.size[None,:,1,None]+self.size[None,:,0,None]
        ext=torch.where(capsule[None,:,None],cap,ext)
        ext=torch.where(sphere[None,:,None],self.size[None,:,0,None],ext)
        mex=(r.abs()@self.aabb[None,:,3:,None]).squeeze(-1)
        off=(r@self.aabb[None,:,:3,None]).squeeze(-1)
        ext=torch.where(mesh[None,:,None],mex,ext)
        p=p+torch.where(mesh[None,:,None],off,0.)
        return p-ext,p+ext

def main():
    p=argparse.ArgumentParser();p.add_argument('--init',required=True);p.add_argument('--name',default='cem03');p.add_argument('--generations',type=int,default=120);p.add_argument('--population',type=int,default=768);p.add_argument('--trials',type=int,default=3);p.add_argument('--dc',action='store_true');p.add_argument('--std',type=float,default=.16);p.add_argument('--min-std',type=float,default=.025);p.add_argument('--landing-weight',type=float,default=0.);p.add_argument('--orientation-weight',type=float,default=.6);p.add_argument('--wide-arms',action='store_true');p.add_argument('--bar-x',type=float,default=-.7);p.add_argument('--mat-top',type=float,default=.18);p.add_argument('--mat-front',type=float,default=-.95)
    a=p.parse_args();torch.set_num_threads(4);torch.manual_seed(719);device='cuda:0'
    n=a.population;trials=a.trials
    if trials<1 or n<12*trials or n%trials:
        p.error('--population must be divisible by --trials and provide at least 12 candidates (population >= 12 * trials)')
    nc=n//trials
    out=ROOT/'artifacts'/a.name;out.mkdir(parents=True,exist_ok=True)
    c=config(n,play=True,dc=a.dc,bar_x=a.bar_x,mat_top=a.mat_top,mat_front=a.mat_front);c.terminations={};c.rewards={};c.observations={}
    env=ManagerBasedRlEnv(c,device=device);r=env.scene['robot'];bounds=Bounds(env.sim.mj_model,device)
    env.scene.write(out/'scene')
    from motor import parameters
    from dataclasses import asdict
    motor=parameters(list(r.joint_names))
    contract=dict(joint_names=list(r.joint_names),body_names=list(r.body_names),action_joint_ids=list(range(29)),action_scale=.5,actor_obs_dim=160,
        default_joint_pos=r.data.default_joint_pos[0].cpu().tolist(),motion_file=str(ROOT/'artifacts/reference/motion.npz'),bar_height_m=.8,bar_x_m=a.bar_x,mat_top_m=a.mat_top,mat_front_m=a.mat_front,physics=asdict(c.sim.mujoco),
        motor_model='dc' if a.dc else 'builtin_position',motor_parameters=motor,clip_joint_targets=a.dc,control_dt=.02)
    (out/'contract.json').write_text(json.dumps(contract,indent=2))
    z=np.load(ROOT/'artifacts/reference/motion.npz');steps=125
    names=list(r.joint_names);groups=[[names.index(side+'_'+stem+'_joint') for side in ['left','right']] for stem in ['hip_pitch','knee','ankle_pitch','shoulder_pitch']]+[[names.index('waist_pitch_joint')]]
    if a.wide_arms:groups += [[names.index(side+'_'+stem+'_joint') for side in ['left','right']] for stem in ['elbow','shoulder_roll']]
    kt=np.array([0,.25,.48,.64,.78,.9,1.15,1.5,1.9,2.6]);basis=[]
    for j in range(1,len(kt)-1):
        values=np.zeros(len(kt));values[j]=1;basis.append(np.interp(np.arange(steps)*.02,kt,values))
    basis=torch.tensor(np.array(basis).T,device=device,dtype=torch.float32)
    initial=np.load(a.init)['params'].reshape(8,-1)
    if initial.shape[1]<len(groups):initial=np.pad(initial,((0,0),(0,len(groups)-initial.shape[1])))
    mean=torch.tensor(initial.reshape(-1),device=device,dtype=torch.float32);std=torch.full_like(mean,a.std)
    ref=torch.tensor(z['qpos'][:steps],device=device,dtype=torch.float32)
    sub_violation=torch.zeros(n,device=device);sub_speedratio=sub_violation.clone()
    sub_limits=r.data.joint_pos_limits;sub_vmax=torch.tensor(motor['velocity'],device=device)
    sub_back=torch.zeros(n,device=device,dtype=torch.bool);sub_head_before=sub_back.clone();sub_backup=torch.zeros(n,device=device)
    sub_headtime=torch.full((n,),3.,device=device);sub_backtime=sub_headtime.clone();sub_clock=[0]
    torso_id=env.sim.mj_model.body('robot/torso_link').id
    original_step=env.sim.step
    def monitored_step():
        original_step();sub_clock[0]+=1
        h=(env.scene['head_mat_contact']._extract_sensor_data().found>0).any(-1)
        b=(env.scene['back_mat_contact']._extract_sensor_data().found>0).any(-1)
        up=env.sim.data.xmat[:,torso_id].reshape(n,3,3)[:,2,0]
        sub_backup.copy_(torch.where(b&(~sub_back),up,sub_backup))
        sub_back.logical_or_(b);sub_head_before.logical_or_(h&(~sub_back))
        sub_headtime.copy_(torch.where(h,torch.minimum(sub_headtime,torch.full_like(sub_headtime,sub_clock[0]*.0025)),sub_headtime))
        sub_backtime.copy_(torch.where(b,torch.minimum(sub_backtime,torch.full_like(sub_backtime,sub_clock[0]*.0025)),sub_backtime))
        q=env.sim.data.qpos[:,7:];v=env.sim.data.qvel[:,6:]
        violation=torch.maximum(sub_limits[:,:,0]-q,q-sub_limits[:,:,1]).clamp(min=0).max(-1).values
        torch.maximum(sub_violation,violation,out=sub_violation)
        torch.maximum(sub_speedratio,(v.abs()/sub_vmax).max(-1).values,out=sub_speedratio)
    if a.dc:env.sim.step=monitored_step
    best=-1e9;bestparam=mean.clone();start=time.time()
    for gen in range(a.generations):
        params=(mean+torch.randn(nc,len(mean),device=device)*std).clamp(-2.2,2.2);params[0]=bestparam;params[1]=mean
        repeated=params.repeat_interleave(trials,dim=0)
        groupact=torch.einsum('tk,nkg->ntg',basis,repeated.view(n,basis.shape[1],len(groups)))
        env.reset();sub_violation.zero_();sub_speedratio.zero_();sub_back.zero_();sub_head_before.zero_();sub_backup.zero_();sub_headtime.fill_(3.);sub_backtime.fill_(3.);sub_clock[0]=0
        # Reproducible, small pose disturbances for each parameter candidate.
        noise=torch.randn(n,29,device=device)*.005;noise[::trials]=0
        r.write_joint_state_to_sim(r.data.joint_pos+noise,r.data.joint_vel);env.sim.forward();env.scene.update(.0)
        maxviolation=torch.zeros(n,device=device);maxspeedratio=maxviolation.clone()
        score=torch.zeros(n,device=device);touch=score.clone();peak=score.clone();full=torch.zeros(n,device=device,dtype=torch.bool)
        premature=torch.zeros(n,device=device,dtype=torch.bool);landed=premature.clone();landing_up=torch.zeros(n,device=device);head_first=premature.clone();back_supported=premature.clone();head_before_back=premature.clone();back_up=torch.zeros(n,device=device)
        good=torch.ones(n,device=device,dtype=torch.bool);minclear=torch.ones(n,device=device);seen=torch.zeros_like(full)
        qs=[];acts=[]
        for k in range(steps):
            act=torch.zeros(n,29,device=device)
            for j,ids in enumerate(groups):act[:,ids]=groupact[:,k,j,None]/.5
            if a.wide_arms:act[:,names.index('right_shoulder_roll_joint')]*=-1
            env.step(act);pos=r.data.root_link_pos_w;quat=r.data.root_link_quat_w
            err=((pos-ref[k,:3])**2*torch.tensor([3.,4.,2.],device=device)).sum(-1)
            rot=1-(quat*ref[k,3:7]).sum(-1).square();score-=(1. if k<30 else 2.)*(err+a.orientation_weight*rot)
            touched=(env.scene['bar_contact'].data.found>0).float().sum(-1);touch+=touched;score-=5.*touched
            lo,hi=bounds.get(env)
            spans=(lo[:,:,0]<=a.bar_x+BAR_RADIUS)&(hi[:,:,0]>=a.bar_x-BAR_RADIUS)&(~full[:,None])
            clearance=torch.where(spans,lo[:,:,2]-BAR_HEIGHT-BAR_RADIUS,torch.ones_like(lo[:,:,2])*5.).min(-1).values
            seen|=spans.any(-1);minclear=torch.minimum(minclear,clearance)
            score-=25*torch.relu(.06-clearance)
            limits=r.data.joint_pos_limits
            violation=torch.maximum(limits[:,:,0]-r.data.joint_pos,r.data.joint_pos-limits[:,:,1]).clamp(min=0).max(-1).values
            speedratio=(r.data.joint_vel.abs()/torch.tensor(motor['velocity'],device=device)).max(-1).values
            maxviolation=torch.maximum(maxviolation,sub_violation if a.dc else violation);maxspeedratio=torch.maximum(maxspeedratio,sub_speedratio if a.dc else speedratio)
            if a.dc:score-=1000*sub_violation+5*torch.relu(sub_speedratio-.95)
            good&=clearance>0
            full|=(hi[:,:,0]<a.bar_x-BAR_RADIUS).all(-1)
            mat_touch=(env.scene['mat_contact'].data.found>0).any(-1)
            premature|=mat_touch&(~full)
            first_land=mat_touch&(~landed)
            torso_id=env.sim.mj_model.body('robot/torso_link').id
            up=env.sim.data.xmat[:,torso_id].reshape(n,3,3)[:,2,0]
            landing_up=torch.where(first_land,up,landing_up)
            head_touch=(env.scene['head_mat_contact'].data.found>0).any(-1)
            head_first |= first_land&head_touch
            back_touch=(env.scene['back_mat_contact'].data.found>0).any(-1)
            back_up=torch.where(back_touch&(~back_supported),up,back_up)
            back_supported |= back_touch
            head_before_back |= head_touch&(~back_supported)
            score-=a.landing_weight*first_land.float()*(1-up).square()
            landed|=mat_touch
            if a.dc:score-=10*(mat_touch&(~full)).float()
            peak=torch.maximum(peak,pos[:,2]);score-=.001*act.square().sum(-1)
            qs.append(env.sim.data.qpos.clone());acts.append(act.clone())
        success=full&good&(touch==0)&seen&(~premature)
        if a.dc:success &= (maxviolation<.02)&(maxspeedratio<1.05)
        if a.landing_weight>0:
            if a.dc:head_before_back=sub_head_before;back_supported=sub_back;back_up=sub_backup
            final_up=env.sim.data.xmat[:,torso_id].reshape(n,3,3)[:,2,0]
            score-=10000*((~good)|(~full)|(touch>0)|premature|(~landed)).float()
            score-=a.landing_weight*(head_first|head_before_back).float()
            score-=10*a.landing_weight*torch.relu(sub_backtime-sub_headtime)
            score-=.5*a.landing_weight*(1-back_up).square()
            score-=.5*a.landing_weight*(1-final_up).square()
            success &= landed&(landing_up>=.7)&(final_up>=.5)&(~head_first)&(~head_before_back)&back_supported&(back_up>=.7)
        score+=30*full.float()+30*success.float()+30*minclear.clamp(-.2,.15)
        if a.dc:score-=1500*torch.relu(-minclear)+100*(~full).float()+100*(~seen).float()
        scores=score.view(nc,trials);robust=.4*scores.mean(-1)+.6*scores.min(-1).values
        v,b=robust.max(0);b=int(b);bi=b*trials
        elites=params[torch.topk(robust,k=max(12,nc//16)).indices]
        mean=.2*mean+.8*elites.mean(0);std=(.2*std+.8*elites.std(0)).clamp(a.min_std,max(.35,a.std))
        # Track the current robust winner, so every saved controller was re-tested on noise.
        if float(v)>best:
            best=float(v);bestparam=params[b].clone()
            result=out/'best.npz';temp=out/'best.tmp.npz'
            np.savez(temp,params=bestparam.cpu().numpy(),actions=torch.stack(acts)[:,bi].cpu().numpy(),qpos=torch.stack(qs)[:,bi].cpu().numpy(),score=best,generation=gen,bar_height_m=.8)
            temp.replace(result)
        rec=dict(generation=gen,best=best,current=float(v),successes=int(success[bi:bi+trials].sum()),trials=trials,
                 max_joint_violation=float(maxviolation[bi:bi+trials].max()),max_speed_ratio=float(maxspeedratio[bi:bi+trials].max()),head_before_back=bool(head_before_back[bi]),back_up=float(back_up[bi]),head_first=bool(head_first[bi]),landing_up=float(landing_up[bi]),final_up=float(env.sim.data.xmat[bi,torso_id].reshape(3,3)[2,0]),nominal_peak=float(peak[bi]),min_clearance=float(minclear[bi:bi+trials].min()),contact_steps=float(touch[bi:bi+trials].sum()),std=float(std.mean()),seconds=time.time()-start)
        print(json.dumps(rec),flush=True)
        with (out/'progress.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
    env.close()
if __name__=='__main__':main()
