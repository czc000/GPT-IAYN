import unittest,json
import mujoco
from gpt_iayn_core import evaluate
from gpt_iayn_core.scene import ROOT
class PhysicsContractTests(unittest.TestCase):
    def test_native_model_uses_training_integrator_and_timestep(self):
        self.assertTrue(hasattr(evaluate,'load_native_model'),'native model must apply saved training physics settings')
        run=ROOT/'artifacts/smoke';contract=json.loads((run/'contract.json').read_text())
        m=evaluate.load_native_model(run,contract)
        self.assertEqual(m.opt.integrator,mujoco.mjtIntegrator.mjINT_IMPLICITFAST)
        self.assertAlmostEqual(m.opt.timestep,.0025)
        self.assertEqual(m.opt.iterations,20)
        self.assertAlmostEqual(m.opt.tolerance,1e-8)
