import unittest
import numpy as np
from scene import ROOT, make_scene
class ReferenceTests(unittest.TestCase):
    def test_reference_contract(self):
        z=np.load(ROOT/'artifacts/reference/motion.npz'); m=make_scene().compile()
        self.assertEqual(z['joint_pos'].shape[1],29)
        for k in ['joint_pos','joint_vel','body_pos_w','body_quat_w','body_lin_vel_w','body_ang_vel_w']:
            self.assertTrue(np.isfinite(z[k]).all(),k)
        np.testing.assert_allclose(np.linalg.norm(z['body_quat_w'],axis=-1),1,atol=1e-5)
        self.assertTrue((z['joint_pos']>=m.jnt_range[1:,0]-1e-6).all())
        self.assertTrue((z['joint_pos']<=m.jnt_range[1:,1]+1e-6).all())
    def test_requested_height(self):
        m=make_scene().compile()
        self.assertAlmostEqual(m.geom_pos[m.geom('crossbar').id,2],.8)
