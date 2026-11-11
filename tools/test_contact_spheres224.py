"""Accepted proxy identity, legacy compatibility, differentiable distances and resume guard."""
import hashlib
from types import SimpleNamespace
import unittest
import numpy as np
import torch

from supermimic.acp.task import BodySpheres, ContactCfg, ContactTask, check_contact_geometry_resume


class SphereProxyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.spheres = BodySpheres('tienkung3_29dof', 'spheres224')

    def test_matches_user_accepted_visualization(self):
        sp = self.spheres
        packed = torch.cat((sp.pts, sp.rad[:, None]), -1).numpy().astype('<f4')
        # Frozen from the reviewed CSV before promotion, rather than recomputing the sampling recipe.
        self.assertEqual(hashlib.sha256(packed.tobytes()).hexdigest(),
                         'd97738195ac0160804dacee9306f68ba29e14ca05cbc8eed465feab4bc021a2f')
        self.assertEqual([e-s for s,e in sp.bounds], [28,28]+[24]*7)
        self.assertFalse(sp.hulls)
        for b,(s,e) in enumerate(sp.bounds):
            self.assertTrue(bool((sp.owner[s:e] == b).all()))

    def test_g1_geometry_is_unchanged(self):
        for mode in ('legacy', 'spheres224'):
            sp = BodySpheres('g1', mode)
            packed = torch.cat((sp.pts, sp.rad[:, None]), -1).numpy().astype('<f4')
            self.assertEqual(hashlib.sha256(packed.tobytes()).hexdigest(),
                             'f613330862b241f8a6ec5096ec33b89a0391c08dbb70b27e1ba1464df1b632ca')

    def test_checkpoint_geometry_requires_explicit_migration(self):
        self.assertFalse(check_contact_geometry_resume({}, 'legacy'))
        with self.assertRaisesRegex(ValueError, 'geometry mismatch'):
            check_contact_geometry_resume({}, 'spheres224')
        self.assertTrue(check_contact_geometry_resume({}, 'spheres224', allow_change=True))
        self.assertFalse(check_contact_geometry_resume({'contact_geometry':'spheres224'}, 'spheres224'))
        with self.assertRaisesRegex(ValueError, 'geometry mismatch'):
            check_contact_geometry_resume({'contact_geometry':'spheres224'}, 'legacy')

    def test_sphere_to_ground_and_box_pose_gradients(self):
        sp = self.spheres
        task = ContactTask.__new__(ContactTask)
        task.cfg = ContactCfg(contact_geometry='spheres224')
        task.sph_pts, task.sph_rad, task.sph_owner = sp.pts.double(), sp.rad.double(), sp.owner
        task.sph_bounds, task.hulls = sp.bounds, {}
        task.link_idx = torch.arange(9)
        task.has_box_terrain = True
        task.motion = SimpleNamespace(goal_boxes=None)
        gen = torch.Generator().manual_seed(224)
        p = (torch.rand(1,9,3,generator=gen,dtype=torch.float64)*.7+.03).requires_grad_()
        rotation = (torch.eye(3,dtype=torch.float64).expand(1,9,3,3).clone()
                    + torch.rand(1,9,3,3,generator=gen,dtype=torch.float64)*.02).requires_grad_()
        boxes = torch.tensor([[[.2,.1,.55,1.,0.,.31,.29,.2]]],dtype=torch.float64)
        state = (torch.tensor([[.1,.2,.8]],dtype=torch.float64),)
        fn = lambda pos, rot: task.touch(state,boxes,fk=(pos,rot))
        self.assertTrue(torch.autograd.gradcheck(fn,(p,rotation),eps=1e-6,atol=2e-5,rtol=2e-4))
        self.assertTrue(torch.isfinite(fn(p,rotation)).all())


if __name__ == '__main__':
    unittest.main()
