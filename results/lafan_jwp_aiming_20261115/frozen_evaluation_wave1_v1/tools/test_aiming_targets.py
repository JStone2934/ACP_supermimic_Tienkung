import unittest
import math
import torch
from supermimic.acp.aiming_targets import spherical_targets, fixed_target_grid


class TargetTests(unittest.TestCase):
    def test_full_grid_has_true_range_and_angles(self):
        p = fixed_target_grid()
        chest = torch.tensor([1., 2., 1.4]).expand(len(p), -1)
        yaw = torch.full((len(p),), .4)
        target, normal = spherical_targets(chest, yaw, p)
        delta = target - chest
        torch.testing.assert_close(delta.norm(dim=-1), p[:, 0])
        torch.testing.assert_close(torch.atan2(delta[:, 1], delta[:, 0]) - yaw, p[:, 1], atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(torch.asin(delta[:, 2]/p[:, 0]), p[:, 2], atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(normal.norm(dim=-1), torch.ones(len(p)))
        self.assertEqual(len(p), 75)

    def test_world_rotation_equivariance_and_no_alias(self):
        p = torch.tensor([[3., 0., 0., .05]])
        chest = torch.zeros(1, 3)
        target, _ = spherical_targets(chest, torch.tensor([math.pi/2]), p)
        torch.testing.assert_close(target, torch.tensor([[0., 3., 0.]]), atol=1e-6, rtol=1e-6)
        chest.add_(100.)
        torch.testing.assert_close(target, torch.tensor([[0., 3., 0.]]), atol=1e-6, rtol=1e-6)


if __name__ == '__main__':
    unittest.main()
