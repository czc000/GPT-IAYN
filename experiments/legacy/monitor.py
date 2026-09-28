"""Evaluate completed frozen exports; never count reference animation as success."""
import argparse, json, subprocess, sys, time
from pathlib import Path
from gpt_iayn_core.scene import ROOT

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',default='run01');p.add_argument('--interval',type=int,default=200);p.add_argument('--max-iteration',type=int,default=4999)
    a=p.parse_args();run=ROOT/'artifacts'/a.run;summary=ROOT/'artifacts'/f'{a.run}-evaluations.jsonl'
    seen=set()
    if summary.exists():seen={row['iteration'] for row in map(json.loads,summary.read_text().splitlines())}
    while True:
        files=sorted(run.glob('model_*.torchscript'),key=lambda f:int(f.stem.split('_')[-1]))
        for f in files:
            it=int(f.stem.split('_')[-1])
            if it in seen or (it%a.interval!=0 and it!=a.max_iteration) or time.time()-f.stat().st_mtime<5:continue
            out=ROOT/'artifacts'/f'eval-{a.run}-{it}'
            cmd=[sys.executable,str(ROOT/'scripts/evaluate.py'),'--run',str(run),'--policy',str(f),'--output',str(out),'--seeds','5','--video']
            with open(str(out)+'.log','w') as log:ret=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT).returncode
            if ret:print('EVAL_ERROR',it,ret,flush=True);continue
            results=json.loads((out/'metrics.json').read_text())
            record={'iteration':it,'successes':sum(r['success'] for r in results),'trials':len(results),
                    'max_pelvis_height_m':max(r['max_pelvis_height_m'] for r in results),
                    'seed0_failures':results[0]['failed_requirements'],'output':str(out)}
            with summary.open('a') as fd:fd.write(json.dumps(record)+'\n')
            print(json.dumps(record),flush=True);seen.add(it)
        if a.max_iteration in seen:break
        exitfile=ROOT/'artifacts'/f'{a.run}.exit'
        if exitfile.exists() and exitfile.read_text().strip()!='0':print('TRAINING_FAILED',exitfile.read_text(),flush=True);break
        time.sleep(30)
if __name__=='__main__':main()
