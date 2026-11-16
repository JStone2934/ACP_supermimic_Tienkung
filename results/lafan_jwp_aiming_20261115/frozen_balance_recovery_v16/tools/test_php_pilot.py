"""Regression tests for pilot selection and independent rejection gates."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from supermimic.acp.robots import get_robot
from tools.retarget_php_pilot import selections,support_height
from tools.check_php_pilot import quality,longest


class PilotTests(unittest.TestCase):
    def test_source_selection_matches_declared_commands(self):
        expected={'standing':0,'walking':1}
        for category,name,path,start,end in selections():
            d=np.load(path);end=len(d['joint_pos']) if end is None else end
            commands=d['vel_cmd'][start:end].argmax(-1)
            if category in expected:self.assertTrue(np.all(commands==expected[category]),(category,name,np.unique(commands)))
            if category=='turning':self.assertTrue(np.all(commands==({'left45':2,'right45':4}[name])))
            self.assertGreater(end-start,30)

    def test_yawed_box_support_and_contiguous_gate(self):
        box=np.array([[0,0,.4,0,1,1,.2,.2]])
        np.testing.assert_allclose(support_height(np.array([[.1,.8],[.8,.1],[2,2]]),box),[.4,0,0])
        self.assertEqual(longest(np.array([1,1,0,1,1,1,0],bool)),3)

    def fixture(self,folder,dz=0,sliding=False):
        robot=get_robot('tienkung3_29dof')
        original=np.load('data/tienkung3_29dof/motions/php_pilot_v2/standing/stand_motion.npz')
        fields=dict(original);jp=np.repeat(fields['joint_pos'][:1],6,axis=0);jp[:,2]+=dz
        if sliding:jp[:,0]+=np.arange(6)*.01
        fields['joint_pos']=jp;path=Path(folder)/'check_motion.npz';np.savez(path,**fields)
        np.save(Path(folder)/'check_terrain.npy',np.zeros((0,1,10)))
        np.savez(Path(folder)/'check_intent.npz',contact=np.ones((6,2),bool))
        return quality(robot,path)

    def test_raw_self_collision_is_not_hidden(self):
        with tempfile.TemporaryDirectory() as t:
            report=self.fixture(t)
        self.assertFalse(report['strict_geometry_pass'])
        self.assertGreater(report['self_penetration_raw_max_m'],.03)
        self.assertTrue(report['pilot_screen_pass'])

    def test_penetrating_motion_is_rejected(self):
        with tempfile.TemporaryDirectory() as t:report=self.fixture(t,dz=-.025)
        self.assertIn('terrain_penetration',report['rejection_reasons'])
        self.assertFalse(report['pilot_screen_pass'])

    def test_sliding_support_is_rejected(self):
        with tempfile.TemporaryDirectory() as t:report=self.fixture(t,sliding=True)
        self.assertIn('support_slip',report['rejection_reasons'])
        self.assertGreater(report['support_slip_p95_m_s'],.4)


if __name__=='__main__':unittest.main()
