"""GPT-IAYN experiment front door. Run with --help for the public workflow."""
import argparse,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def run(script,*args):
    subprocess.run([sys.executable,str(ROOT/script),*map(str,args)],cwd=ROOT,check=True)

def main():
    p=argparse.ArgumentParser(description='GPT-IAYN — GPT Is All You Need: standing G1 high jump')
    sp=p.add_subparsers(dest='command',required=True)
    for name in ['verify','restore','seed','teacher']:sp.add_parser(name)
    d=sp.add_parser('demo');d.add_argument('--video',action='store_true');d.add_argument('--output',default='artifacts/demo');d.add_argument('--seeds',type=int,default=1);d.add_argument('--perturb',type=float,default=.01)
    t=sp.add_parser('train');t.add_argument('--name',default='ppo');t.add_argument('--num-envs',type=int,default=2048);t.add_argument('--iterations',type=int,default=500);t.add_argument('--resume');t.add_argument('--reference',default='artifacts/rl-reference/motion.npz');t.add_argument('--teacher',default='artifacts/final')
    s=sp.add_parser('search');s.add_argument('--name',default='cem-reproduce');s.add_argument('--generations',type=int,default=180);s.add_argument('--population',type=int,default=1024);s.add_argument('--from-zero',action='store_true')
    a=p.parse_args();os.chdir(ROOT)
    if a.command=='verify':run('tools/artifacts.py','verify')
    elif a.command=='restore':
        run('tools/artifacts.py','restore');run('tools/artifacts.py','assets')
    elif a.command=='demo':
        run('evaluate_rl.py','--run','artifacts/final-rl-sim2sim','--policy','artifacts/final-rl-sim2sim/policy.torchscript','--output',a.output,'--seeds',a.seeds,'--perturb',a.perturb,*(['--video'] if a.video else []))
    elif a.command=='seed':run('prepare.py')
    elif a.command=='teacher':run('prepare_rl.py')
    elif a.command=='train':
        run('train_rl.py','--name',a.name,'--num-envs',a.num_envs,'--iterations',a.iterations,'--perturb','.025','--feedback-only','--reference',a.reference,'--teacher',a.teacher,*(['--resume',a.resume] if a.resume else []))
    elif a.command=='search':
        initial=ROOT/'release-data/cem15-back-support/candidate-b.npz'
        if a.from_zero:
            import numpy as np
            initial=ROOT/'artifacts/cem-zero-init.npz';initial.parent.mkdir(exist_ok=True)
            np.savez(initial,params=np.zeros(56))
        run('cem_refine.py','--init',initial,'--name',a.name,'--generations',a.generations,'--population',a.population,'--trials',4,'--dc','--std','.1','--min-std','.02','--landing-weight',3000,'--orientation-weight',5,'--wide-arms')

if __name__=='__main__':main()
