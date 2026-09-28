"""Convert explicitly flattened MJCF into PhysX USD (separate diagnostic asset)."""
import argparse
from pathlib import Path
from gpt_iayn_core.paths import ROOT as root
out=root/'artifacts/rl-isaac';p=out/'scene/robot.xml'
assert p.exists(),'Run prepare_isaac_model.py first'
from isaaclab.app import AppLauncher
parser=argparse.ArgumentParser();AppLauncher.add_app_launcher_args(parser);args=parser.parse_args()
launcher=AppLauncher(args);app=launcher.app
import omni.kit.app
omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate('isaacsim.asset.importer.mjcf',True)
from isaaclab.sim.converters import MjcfConverter,MjcfConverterCfg
conv=MjcfConverter(MjcfConverterCfg(asset_path=str(p),usd_dir=str(out/'usd'),usd_file_name='g1.usd',fix_base=False,self_collision=True,make_instanceable=False,force_usd_conversion=True))
from pxr import Usd,UsdPhysics
stage=Usd.Stage.Open(conv.usd_path)
for prim in stage.Traverse():
    if prim.GetName()=='worldBody' and prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
stage.GetRootLayer().Save()
print('CONVERTED',conv.usd_path,flush=True)
app.close()
