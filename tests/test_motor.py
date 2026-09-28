import unittest,numpy as np
from gpt_iayn_core.motor import torque_numpy,dc_robot_cfg,parameters
from mjlab.asset_zoo.robots import get_g1_robot_cfg
class MotorTests(unittest.TestCase):
    def test_torque_falls_to_zero_at_no_load_speed(self):
        p=dict(kp=[100],kd=[0],effort=[139],velocity=[20])
        self.assertEqual(float(torque_numpy([10],[0],[0],p)[0]),139.)
        self.assertEqual(float(torque_numpy([10],[0],[10],p)[0]),69.5)
        self.assertEqual(float(torque_numpy([10],[0],[20],p)[0]),0.)
        self.assertLess(float(torque_numpy([10],[0],[25],p)[0]),0.)
    def test_dc_configuration_does_not_change_existing_robot_configs(self):
        a=get_g1_robot_cfg();before=type(a.articulation.actuators[0]);dc_robot_cfg()
        self.assertIs(type(a.articulation.actuators[0]),before)
    def test_knee_and_hip_specs(self):
        p=parameters(['left_knee_joint','right_hip_pitch_joint'])
        self.assertEqual(p['velocity'],[20.,32.]);self.assertEqual(p['effort'],[139.,88.])
