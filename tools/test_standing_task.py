"""Focused standing-task contracts; uses the prepared synthetic task data."""
import unittest
import torch

from supermimic.acp.robots import get_robot
from supermimic.acp.kinematics import RobotKinematics
from supermimic.acp.motion import MotionLib
from supermimic.acp.mdp import MDP
from supermimic.acp.task import ContactCfg, ContactTask
from supermimic.acp.standing import StandingTask, freeze_normalizers
from supermimic.acp.networks import RunningNorm


class StandingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        robot = get_robot('tienkung3_29dof')
        kin = RobotKinematics('cpu', robot)
        root = 'data/tienkung3_29dof/standing_aiming'
        cls.motion = MotionLib.from_dir(root + '/motions', kin, 'cpu')
        cls.mdp = MDP(kin, cls.motion, torch.tensor(robot.default_q))
        ctc = ContactTask(ContactCfg(contacts_dir=root + '/contacts', contact_geometry='spheres224'), cls.mdp, root + '/motions', 'cpu')
        cls.task = StandingTask(cls.mdp, ctc)

    def test_composed_pd_targets_and_gradients(self):
        t = self.task
        a = torch.randn(3, 29, requires_grad=True)
        xi = torch.randn(3, self.mdp.xi_dim) * .01
        composed = t.compose(a, xi)
        target = self.mdp.default_q + xi[:, 1:30] + t.alpha * composed
        torch.testing.assert_close(target[:, t.arm_ids], t.target_q[t.arm_ids].expand(3, -1))
        torch.testing.assert_close(composed[:, t.free], a[:, t.free])
        composed.sum().backward()
        self.assertTrue((a.grad[:, t.arm_ids] == 0).all())
        self.assertTrue((a.grad[:, t.free] == 1).all())
        self.assertEqual(t.noise_scale(3000, .05), 0.)

    def test_static_reference_translation_has_restoring_gradient(self):
        f = torch.zeros(1, dtype=torch.long)
        S = self.motion.ref_state(f)
        self.assertTrue((self.motion.root_v == 0).all())
        origin = self.task.terms(S, f)
        shifted = list(S)
        shifted[0] = (S[0] + torch.tensor([[.2, .1, 0.]])).requires_grad_()
        displaced = self.task.terms(tuple(shifted), f)
        for key in ('root', 'feet'):
            self.assertLess(float(displaced[key].detach()), float(origin[key]))
        (displaced['root'] + displaced['feet']).sum().backward()
        self.assertTrue((shifted[0].grad[:, :2] < 0).all())

    def test_freezing_preserves_saved_statistics(self):
        norm = RunningNorm(4)
        norm.update(torch.randn(16, 4))
        before = {k: v.clone() for k, v in norm.state_dict().items()}
        freeze_normalizers(norm)
        norm.update(torch.randn(20, 4) + 10)
        for k, v in norm.state_dict().items():
            torch.testing.assert_close(v, before[k], rtol=0, atol=0)


if __name__ == '__main__':
    torch.set_num_threads(4)
    unittest.main()
