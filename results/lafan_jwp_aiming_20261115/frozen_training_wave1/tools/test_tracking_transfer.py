import unittest
import torch
from supermimic.acp.networks import Policy
from supermimic.acp.robots import get_robot
from supermimic.acp.tracking_transfer import transfer_tracker


class TransferTests(unittest.TestCase):
    def test_all_joint_outputs_and_normalization_preserved(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        robot = get_robot('tienkung3_29dof')
        old = Policy(99, 93, hist=10, robot=robot)
        with torch.no_grad():
            old.goal_norm.mean.copy_(torch.randn(93))
            old.goal_norm.var.copy_(torch.rand(93) + .1)
            old.obs_norm.mean.copy_(torch.randn(99))
            old.obs_norm.var.copy_(torch.rand(99) + .1)
            old.dec[-1].bias.copy_(torch.randn(29))
        ck = dict(robot_metadata=robot.metadata(), task_config=None,
                  args=dict(hist=10, task='tracking'),
                  ctc_cfg=dict(blind=False, contact_geometry='spheres224'), pi=old.state_dict())
        new, meta = transfer_tracker(ck, robot)
        history, command = torch.randn(8, 10, 99), torch.randn(8, 93)
        with torch.no_grad():
            expected = old(history, command)
            actual = new(history, torch.cat((command, torch.randn(8, 14)), -1))
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(new.dec[-1].bias, old.dec[-1].bias, rtol=0, atol=0)
        self.assertEqual(len(meta['preserved_output_names']), 29)
        self.assertEqual(new.obs_norm.max_count, 0)
        ck['task_config'] = {'name': 'standing_aiming'}
        with self.assertRaises(ValueError):
            transfer_tracker(ck, robot)


if __name__ == '__main__':
    unittest.main()
