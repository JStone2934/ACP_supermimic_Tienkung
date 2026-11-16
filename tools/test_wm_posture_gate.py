import math
import unittest
import torch
from supermimic.acp.aiming_fullbody_validation import world_up_error_deg,add_posture_thresholds

class PostureGateTests(unittest.TestCase):
    def rotation(self,tilt,yaw=0.):
        t,y=math.radians(tilt),math.radians(yaw)
        ry=torch.tensor([[math.cos(t),0.,math.sin(t)],[0.,1.,0.],[-math.sin(t),0.,math.cos(t)]],dtype=torch.float64)
        rz=torch.tensor([[math.cos(y),-math.sin(y),0.],[math.sin(y),math.cos(y),0.],[0.,0.,1.]],dtype=torch.float64)
        return rz@ry

    def test_world_axis_and_equal_tilt_different_lean(self):
        a=self.rotation(10.)
        self.assertAlmostEqual(world_up_error_deg(a,self.rotation(0.)).item(),10.)
        self.assertAlmostEqual(world_up_error_deg(a,self.rotation(10.,180.)).item(),20.)
        self.assertAlmostEqual(world_up_error_deg(self.rotation(0.),self.rotation(0.,90.)).item(),0.)
        self.assertAlmostEqual(world_up_error_deg(a,a).item(),0.)
        torch.testing.assert_close(world_up_error_deg(torch.stack([a,a]),torch.stack([self.rotation(0.),self.rotation(10.,180.)])),torch.tensor([10.,20.],dtype=torch.float64))

    def test_optional_gate_preserves_legacy_thresholds(self):
        base={'root_m_p95':.05,'feet_m_p95':.025}
        self.assertEqual(add_posture_thresholds(base,{}),base)
        self.assertEqual(add_posture_thresholds(base,{'wm_gate_chest_deg':3.,'wm_gate_pelvis_deg':2.}),dict(base,chest_up_deg_p95=3.,pelvis_up_deg_p95=2.))
        self.assertEqual(base,{'root_m_p95':.05,'feet_m_p95':.025})
        for bad in (0.,-1.,float('nan'),float('inf')):
            with self.assertRaises(ValueError):add_posture_thresholds(base,{'wm_gate_chest_deg':bad})

if __name__=='__main__':unittest.main()
