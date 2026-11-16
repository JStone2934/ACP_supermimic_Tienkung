"""Bootstrap ceilings follow actual curriculum rewards without changing rewards."""
import json
import unittest
from pathlib import Path
import torch
from supermimic.acp.robots import get_robot
from supermimic.acp.aiming_wide import WideAimingTask
from supermimic.acp.aiming_moving import MovingAimingTask
from supermimic.acp.aiming_fullbody import validate_checkpoint
from supermimic.acp.train_aiming_fullbody import bound_bootstrap
from supermimic.acp.wide_curriculum import STAGES

class BoundTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        robot=get_robot('tienkung3_29dof')
        patch=json.loads(Path('docs/config/standing_aiming_fullbody_v2/chest_posture_v1.json').read_text())
        self.cfg=dict(initial_q=list(robot.default_q),initial_height=robot.stand_height,
                      chest_posture=patch['chest_posture'],reward_weights=patch['reward_weights'],
                      gamma=.99,bound_value_bootstrap=True)

    def test_actual_stage_ceiling_and_legacy_metadata(self):
        for level,stage in enumerate(STAGES):
            task=WideAimingTask('cpu',dict(self.cfg,curriculum_level=level))
            expected=.154+.02*stage['aim_weight']
            self.assertAlmostEqual(task.max_step_reward,expected)
            cap=bound_bootstrap(torch.tensor([100.]),torch.tensor([0]),task,self.cfg)
            self.assertAlmostEqual(cap.item(),expected*100,places=4)
            validate_checkpoint(dict(pi={},task_config=task.metadata(),robot_metadata=task.robot.metadata()),task)
        moving=MovingAimingTask('cpu',dict(self.cfg,curriculum_level=0))
        self.assertAlmostEqual(moving.max_step_reward,.234)

    def test_unbounded_cost_cannot_be_positive(self):
        for name in ('action_rate','joint_limit','request_limit','chest_excess'):
            cfg=dict(self.cfg,reward_weights=dict(self.cfg['reward_weights'],**{name:1.}))
            with self.assertRaises(ValueError):WideAimingTask('cpu',cfg)

    def test_reward_unchanged_across_cap_levels_for_same_context(self):
        a=WideAimingTask('cpu',dict(self.cfg,curriculum_level=0))
        b=WideAimingTask('cpu',dict(self.cfg,curriculum_level=1))
        state=a.initial_state(1)
        ctx=a.context(state,torch.tensor([[3.,0.,0.,.2]]),0.)
        action=torch.zeros(1,29)
        torch.testing.assert_close(a.reward(state,action,action,ctx),b.reward(state,action,action,ctx))

if __name__=='__main__':unittest.main()
