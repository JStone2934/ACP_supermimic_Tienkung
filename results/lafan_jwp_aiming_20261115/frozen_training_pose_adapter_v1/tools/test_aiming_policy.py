import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from supermimic.acp.aiming_policy import AimingPolicy
from supermimic.acp.aiming_fullbody import FullbodyPolicy
from supermimic.acp.aiming_wide import WideAimingTask
from supermimic.acp.aiming_moving import MovingAimingTask
from supermimic.acp.robots import get_robot


class InferenceTests(unittest.TestCase):
    def test_reference_free_runtime_matches_training_and_keeps_targets_fixed(self):
        torch.set_num_threads(2);torch.manual_seed(10)
        robot=get_robot('tienkung3_29dof')
        cfg=dict(initial_q=list(robot.default_q),initial_height=robot.stand_height,physics_substeps=16)
        for task_cls in (WideAimingTask,MovingAimingTask):
            with self.subTest(task=task_cls.__name__),tempfile.TemporaryDirectory() as folder:
                task=task_cls('cpu',cfg);policy=FullbodyPolicy(robot,freeze_norm=True)
                path=Path(folder)/'test_only.pt'
                torch.save(dict(config=cfg,task_config=task.metadata(),robot_metadata=robot.metadata(),pi=policy.state_dict()),path)
                with patch('supermimic.acp.motion.MotionLib.__init__',side_effect=AssertionError('No motion reads')):
                    runtime=AimingPolicy(path)
                    state=task.initial_state(2);runtime.reset(state,torch.tensor([[3.,0.,0.,.05]]).expand(2,-1))
                    target=runtime.context['aim_target_w'].clone();anchor=runtime.context['root_anchor_w'].clone()
                    velocity=torch.tensor([[.2,0.,0.]]).expand(2,-1)
                    if task_cls is MovingAimingTask:runtime.set_velocity(velocity)
                    else:
                        with self.assertRaises(ValueError):runtime.set_velocity(velocity)
                    obs=task.mdp.obs(state,runtime.previous_action,runtime.xi)
                    manual=policy(obs[:,None].repeat(1,10,1),task.command(state,runtime.context))
                    expected=task.compose_action(manual,runtime.xi)[1]
                    torch.testing.assert_close(runtime.step(state),expected,rtol=0,atol=0)
                    torch.testing.assert_close(runtime.context['aim_target_w'],target,rtol=0,atol=0)
                    expected_anchor=anchor+(.02*velocity if task_cls is MovingAimingTask else 0)
                    torch.testing.assert_close(runtime.context['root_anchor_w'],expected_anchor)
                    history=runtime.history.clone()
                    runtime.set_target(target+torch.tensor([0.,.2,0.]),torch.tensor([[1.,0.,0.]]).expand(2,-1))
                    torch.testing.assert_close(runtime.history,history,atol=0,rtol=0)
                    runtime.step(state)
                    with self.assertRaises(ValueError):runtime.step((*state[:4],state[4][:,:28],state[5]))


if __name__=='__main__':unittest.main()
