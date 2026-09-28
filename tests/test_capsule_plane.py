import unittest,numpy as np,mujoco
from scipy.spatial.transform import Rotation
from gpt_iayn_core.geometry import plane_minimum
class PlaneTests(unittest.TestCase):
 def test_sphere_slice_and_no_intersection(self):
  m=mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom name="sphere" type="sphere" size=".05" pos="-.7 0 .9"/></worldbody></mujoco>');d=mujoco.MjData(m);mujoco.mj_forward(m,d)
  self.assertAlmostEqual(plane_minimum(m,d,[0])[0],.038,places=10)
  self.assertTrue(np.isinf(plane_minimum(m,d,[0],x=0)[0]))
 def test_rotated_capsule_agrees_with_dense_segment_minimization(self):
  m=mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom name="capsule" type="capsule" size=".03 .2" pos="-.7 0 1"/></worldbody></mujoco>');d=mujoco.MjData(m);rng=np.random.default_rng(8)
  for _ in range(100):
   m.geom_quat[0]=Rotation.random(random_state=rng).as_quat()[[3,0,1,2]];m.geom_pos[0,0]=rng.uniform(-.8,-.6);mujoco.mj_forward(m,d)
   axis=d.geom_xmat[0].reshape(3,3)[:,2];a=d.geom_xpos[0]-.2*axis;delta=.4*axis
   if abs(delta[0])<1e-10:continue
   endpoints=sorted([(-.7-.03-a[0])/delta[0],(-.7+.03-a[0])/delta[0]]);lo=max(0,endpoints[0]);hi=min(1,endpoints[1])
   actual=plane_minimum(m,d,[0])[0]
   if lo>hi:self.assertTrue(np.isinf(actual));continue
   t=np.linspace(lo,hi,10001);u=-.7-a[0]-t*delta[0];dense=np.min(a[2]+t*delta[2]-np.sqrt(np.maximum(0,.03**2-u**2)))-.812
   self.assertLessEqual(actual,dense+1e-9);self.assertLess(dense-actual,2e-5)
if __name__=='__main__':unittest.main()
