"""GPU policy search for a dynamics-feasible takeoff/clearance seed.

Learns phase spline parameters. Only joint PD targets are controlled; root is free.
This auxiliary controller is distinct from the PPO network.
"""
import argparse,json,time
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from gpt_iayn_core.scene import ROOT
from gpt_iayn_core.task import config


def main():
    p=argparse.ArgumentParser();p.add_argument('--population',type=int,default=1024);p.add_argument('--generations',type=int,default=100);p.add_argument('--name',default='cem01');p.add_argument('--init')
    a=p.parse_args();torch.set_num_threads(4);torch.manual_seed(123)
    out=ROOT/'artifacts'/a.name;out.mkdir(exist_ok=True,parents=True)
    c=config(a.population,play=True);c.terminations={};c.rewards={};c.observations={}
    env=ManagerBasedRlEnv(c,device='cuda:0');r=env.scene['robot'];motion=env.command_manager.get_term('motion').motion
    z=np.load(ROOT/'artifacts/reference/motion.npz');steps=125;dt=.02
    joints=list(r.joint_names);groups=[[joints.index(side+'_'+stem+'_joint') for side in ['left','right']] for stem in ['hip_pitch','knee','ankle_pitch','shoulder_pitch']]+[[joints.index('waist_pitch_joint')]]
    # Compact smooth phase parameters, with a fixed zero-residual stand before 0.20 seconds.
    kt=np.array([0,.25,.48,.64,.78,.9,1.15,1.5,1.9,2.6]);ts=np.arange(steps)*dt
    basis=[]
    for j in range(1,len(kt)-1):
        values=np.zeros(len(kt));values[j]=1;basis.append(np.interp(ts,kt,values))
    basis=torch.tensor(np.array(basis).T,dtype=torch.float32,device='cuda:0')
    dims=basis.shape[1]*len(groups);mean=torch.zeros(dims,device='cuda:0');std=torch.full_like(mean,.65)
    root_ref=torch.tensor(z['qpos'][:steps,:3],device='cuda:0',dtype=torch.float32)
    ori_ref=torch.tensor(z['qpos'][:steps,3:7],device='cuda:0',dtype=torch.float32)
    if a.init:
        mean=torch.tensor(np.load(a.init)['params'],device='cuda:0',dtype=torch.float32);std[:]=.35
    best=-1e9;bestparam=mean.clone();start=time.time()
    for gen in range(a.generations):
        params=mean+torch.randn(a.population,dims,device='cuda:0')*std
        params[0]=bestparam;params[1]=mean;params=params.clamp(-2.2,2.2)
        co=params.view(a.population,basis.shape[1],len(groups))
        groupact=torch.einsum('tk,nkg->ntg',basis,co)
        env.reset();score=torch.zeros(a.population,device='cuda:0');peak=score.clone();touch=score.clone();passed=score.clone()
        qs=[];acts=[]
        for k in range(steps):
            action=torch.zeros(a.population,29,device='cuda:0')
            for j,ids in enumerate(groups):action[:,ids]=groupact[:,k,j,None]/.5
            env.step(action)
            pos=r.data.root_link_pos_w-env.scene.env_origins
            quat=r.data.root_link_quat_w
            err=((pos-root_ref[k])**2*torch.tensor([3.,4.,2.],device='cuda:0')).sum(-1)
            rot=1-(quat*ori_ref[k]).sum(-1).square()
            weight=1.0 if k<30 else 2.0
            score-=weight*(err+0.6*rot)
            peak=torch.maximum(peak,pos[:,2])
            touched=(env.scene['bar_contact'].data.found>0).float().sum(-1)
            touch+=touched;score-=2.*touched
            score-=.001*action.square().sum(-1)
            if k>45:
                # Encourage crossing while high; crawling below cannot earn this term.
                crossed=(pos[:,0]<-.7)&(pos[:,2]>.85)
                passed=torch.maximum(passed,crossed.float())
            qs.append(env.sim.data.qpos.clone());acts.append(action.clone())
        score+=20*passed
        idx=torch.topk(score,k=max(16,a.population//32)).indices
        elite=params[idx];mean=.2*mean+.8*elite.mean(0);std=(.2*std+.8*elite.std(0)).clamp(.08,.8)
        v,bi=score.max(0);bi=int(bi)
        if float(v)>best:
            best=float(v);bestparam=params[bi].clone()
            np.savez(out/'best.npz',params=bestparam.cpu().numpy(),actions=torch.stack(acts)[:,bi].cpu().numpy(),
                qpos=torch.stack(qs)[:,bi].cpu().numpy(),score=best,generation=gen,bar_height_m=.8)
        rec=dict(generation=gen,best=best,current=float(v),peak=float(peak[bi]),bar_contact_steps=float(touch[bi]),high_crossing=float(passed[bi]),std=float(std.mean()),seconds=time.time()-start)
        print(json.dumps(rec),flush=True)
        with (out/'progress.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
    env.close()
if __name__=='__main__':main()
