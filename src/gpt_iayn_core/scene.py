from pathlib import Path
import mujoco

from gpt_iayn_core.paths import ROOT
BAR_X=-0.70
BAR_HEIGHT=0.80
BAR_HALF_WIDTH=1.0
BAR_RADIUS=0.012
MAT_TOP=0.25

def add_obstacles(spec, height=BAR_HEIGHT, mat_top=MAT_TOP, bar_x=BAR_X, mat_front=-.95):
    w=spec.worldbody
    w.add_geom(name='crossbar',type=mujoco.mjtGeom.mjGEOM_CAPSULE,
               fromto=[bar_x,-BAR_HALF_WIDTH,height,bar_x,BAR_HALF_WIDTH,height],
               size=[BAR_RADIUS,0,0],rgba=[1,.25,.08,1],contype=1,conaffinity=1,friction=[.4,.01,.001])
    for y in [-1.10,1.10]:
        w.add_geom(name='post_'+str(y),type=mujoco.mjtGeom.mjGEOM_BOX,
                   pos=[bar_x,y,.6],size=[.035,.035,.6],rgba=[.75,.8,.85,1])
        w.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,pos=[bar_x,y,.025],size=[.3,.15,.025],rgba=[.25,.25,.3,1])
    w.add_geom(name='landing_mat',type=mujoco.mjtGeom.mjGEOM_BOX,
               pos=[mat_front-1.,0,mat_top/2],size=[1.0,1.05,mat_top/2],rgba=[.06,.42,.8,1],
               solref=[.04,1.0],solimp=[.9,.95,.001,.5,2],friction=[.8,.02,.001])

def make_scene(height=BAR_HEIGHT):
    from mjlab.asset_zoo.robots import get_g1_robot_cfg
    from mjlab.scene import Scene, SceneCfg
    from mjlab.terrains import TerrainEntityCfg
    return Scene(SceneCfg(terrain=TerrainEntityCfg(terrain_type='plane'),
        entities={'robot':get_g1_robot_cfg()},spec_fn=lambda s:add_obstacles(s,height)),device='cpu')
