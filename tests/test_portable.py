"""Release portability and physics equivalence, independent of mjlab."""
import importlib.abc,json,subprocess,sys,unittest,os
from pathlib import Path
import mujoco,numpy as np
from gpt_iayn_core.portable import apply_physics,motion_path
ROOT=Path(__file__).resolve().parents[1]

class PortableTests(unittest.TestCase):
    def test_relative_reference_follows_package_after_move(self):
        self.assertEqual(motion_path(Path('/new/location/run'),{'motion_file':'evaluation-reference.npz'}),Path('/new/location/run/evaluation-reference.npz'))

    def test_native_evaluator_does_not_import_training_stack(self):
        code='''import sys,importlib.abc
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('mjlab','rsl_rl','warp'):raise RuntimeError('Unexpected training dependency: '+fullname)
sys.meta_path.insert(0,Block())
from gpt_iayn_core import evaluate_rl,policy_export
'''
        subprocess.run([sys.executable,'-c',code],cwd=ROOT,check=True,env=dict(os.environ,PYTHONPATH=str(ROOT/'src')))

    def test_serialized_options_are_applied_and_unknown_fields_rejected(self):
        m=mujoco.MjModel.from_xml_string('<mujoco/>')
        cfg=json.loads((ROOT/'release-data/final/contract.json').read_text())['physics']
        apply_physics(m,cfg)
        self.assertEqual(m.opt.integrator,mujoco.mjtIntegrator.mjINT_IMPLICITFAST)
        self.assertEqual(m.opt.solver,mujoco.mjtSolver.mjSOL_NEWTON)
        self.assertEqual(m.opt.iterations,20);self.assertEqual(m.opt.timestep,.0025)
        np.testing.assert_array_equal(m.opt.gravity,[0,0,-9.81])
        with self.assertRaises(ValueError):apply_physics(m,{'unreviewed_option':1})

    def test_release_models_and_demonstrations_match_manifest(self):
        from gpt_iayn_core.artifacts import verify
        self.assertGreater(verify(),20)

if __name__=='__main__':unittest.main()
