import unittest
from types import SimpleNamespace
import torch
from tools.evaluate_tracker_standing import quiet_frames
from supermimic.acp.env import make_sim


class StandingGateTests(unittest.TestCase):
    def test_quiet_selection_requires_entire_window_and_both_feet(self):
        motion=SimpleNamespace(q=torch.zeros(200,29),qd=torch.zeros(200,29),
            root_v=torch.zeros(200,3),root_w=torch.zeros(200,3),
            clip_start=torch.tensor([0,100]),clip_end=torch.tensor([99,199]),
            first_variants=lambda clips:torch.zeros_like(clips))
        labels=torch.zeros(200,1,9,dtype=torch.bool);labels[:,:,:2]=True
        labels[100:,0,1]=False
        frames=quiet_frames(motion,SimpleNamespace(labels=labels),100)
        self.assertEqual(set(frames.tolist()),{25,75})
        motion.root_v[49,0]=.2
        frames=quiet_frames(motion,SimpleNamespace(labels=labels),100)
        self.assertEqual(frames.tolist(),[75])
        labels[:100,0,1]=False
        with self.assertRaises(ValueError):quiet_frames(motion,SimpleNamespace(labels=labels),100)

    def test_explicit_substeps_preserve_control_frequency(self):
        cfg=make_sim(robot='tienkung3_29dof',physics_substeps=16)
        self.assertAlmostEqual(cfg.mujoco.timestep,.00125)
        self.assertAlmostEqual(cfg.mujoco.timestep*16,.02)
        for bad in (0,-1,1.5,True):
            with self.assertRaises(ValueError):make_sim(physics_substeps=bad)


if __name__=='__main__':unittest.main()
