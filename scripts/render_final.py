"""Render multiple views of the stored native dynamic rollout; playback labels included."""
import json
import numpy as np
import mujoco
import imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
from gpt_iayn_core.scene import ROOT
from gpt_iayn_core.evaluate import load_native_model

def main():
    out=ROOT/'artifacts/final';contract=json.loads((out/'contract.json').read_text())
    m=load_native_model(out,contract);d=mujoco.MjData(m)
    trajectory=np.load(out/'nominal-trajectory.npz');q=trajectory['qpos'];times=trajectory['stats'][:,0]
    metrics=json.loads((out/'metrics-20-seeds.json').read_text());assert metrics[0]['success'] and metrics[0]['bar_contacts']==0
    passed=sum(r['success'] for r in metrics)
    m.vis.global_.offwidth=1920;m.vis.global_.offheight=1080
    renderer=mujoco.Renderer(m,height=1080,width=1920)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',28)
    small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',23)
    writer=imageio.get_writer(str(out/'g1-highjump-0.8m-sim2sim.mp4'),fps=50,codec='libx264',quality=8,macro_block_size=1)
    configs=[('Oblique view | 1x speed',135,-18,4.0,1),('Side view | 1x speed',110,-12,4.1,1),('Side view | 0.5x slow motion',110,-12,4.1,2)]
    n=0
    for title,az,el,dist,repeats in configs:
        cam=mujoco.MjvCamera();cam.lookat[:]=[-.85,0,.7];cam.distance=dist;cam.azimuth=az;cam.elevation=el
        for k,pose in enumerate(q):
            d.qpos[:]=pose;mujoco.mj_forward(m,d);renderer.update_scene(d,camera=cam);im=Image.fromarray(renderer.render());dr=ImageDraw.Draw(im)
            dr.rectangle((0,0,1920,112),fill=(13,19,28))
            dr.text((28,17),'UNITREE G1  |  0.80 m BACK-FIRST HIGH JUMP',font=font,fill=(245,249,253))
            dr.text((28,66),f'MuJoCo-Warp trained CEM + PD controller -> native MuJoCo   |   {title}',font=small,fill=(170,202,225))
            dr.rectangle((0,1000,1920,1080),fill=(13,19,28))
            dr.text((28,1017),f'Dynamic rollout replay  |  t = {times[k]:4.2f} s  |  bar contacts: 0  |  {passed} / {len(metrics)} small-perturbation trials passed',font=small,fill=(225,240,231))
            frame=np.asarray(im)
            count=repeats
            if k==0:count+=35
            if k==len(q)-1:count+=35
            for _ in range(count):writer.append_data(frame);n+=1
    writer.close();renderer.close();print(json.dumps({'frames':n,'fps':50,'duration_s':n/50,'source':'native simulation trajectory, not reference','video':str(out/'g1-highjump-0.8m-sim2sim.mp4')}))
if __name__=='__main__':main()
