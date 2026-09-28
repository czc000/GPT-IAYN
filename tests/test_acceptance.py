import unittest
from acceptance import assess

class AcceptanceTests(unittest.TestCase):
    def trace(self):
        return dict(grounded_start=True, finished_on_landing_side=True, head_before_back_support=False, back_support_torso_up=1., took_off=True, all_above_bar=True,
                    all_crossed=True, between_posts=True, bar_contacts=0,
                    torso_up_at_crossing=0.9, landed_on_mat=True, max_joint_limit_violation_rad=0.,max_joint_speed_ratio=.8,premature_mat_contact=False,first_mat_contact_head=False,first_mat_torso_up=1.,final_torso_up=1.)
    def test_accepts_complete_jump(self):
        self.assertTrue(assess(self.trace())['success'])
    def test_rejects_each_cheat(self):
        for key in ['finished_on_landing_side','grounded_start','took_off','all_above_bar','all_crossed','between_posts','landed_on_mat']:
            t=self.trace(); t[key]=False
            with self.subTest(key=key): self.assertFalse(assess(t)['success'])
    def test_rejects_touch_and_face_down(self):
        for key,value in [('bar_contacts',1),('torso_up_at_crossing',-0.9),('max_joint_limit_violation_rad',.04),('max_joint_speed_ratio',1.2),('premature_mat_contact',True),('head_before_back_support',True),('back_support_torso_up',.1),('first_mat_contact_head',True),('first_mat_torso_up',.2),('final_torso_up',-.9)]:
            t=self.trace(); t[key]=value
            self.assertFalse(assess(t)['success'])

if __name__ == '__main__': unittest.main()
