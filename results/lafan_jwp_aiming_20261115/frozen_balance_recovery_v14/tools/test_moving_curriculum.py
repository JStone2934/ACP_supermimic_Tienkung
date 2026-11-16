import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from supermimic.acp import moving_curriculum as curriculum
from supermimic.acp.aiming_moving import MovingAimingTask
from supermimic.acp.robots import get_robot,sha256
from supermimic.rotations import so3_exp
from tools.prepare_moving_aiming_init import verify_standing_acceptance


class MovingCurriculumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2);r=get_robot('tienkung3_29dof')
        cls.task=MovingAimingTask('cpu',dict(initial_q=list(r.default_q),initial_height=r.stand_height,physics_substeps=16))

    def test_commands_cover_stops_and_full_speed_without_moving_world_targets(self):
        task=self.task;state=task.initial_state(6);ctx=task.context(state,torch.tensor([[4.,0.,0.,.05]]).expand(6,-1))
        env=SimpleNamespace(task=task,state=lambda:state,context=ctx,ep_step=torch.arange(6)*250)
        anchor=ctx['root_anchor_w'].clone();curriculum.apply_commands(env,3)
        torch.testing.assert_close(ctx['velocity_command'][:,0],torch.tensor(curriculum.SPEED_FRACTIONS)*.3)
        torch.testing.assert_close(ctx['root_anchor_w'],anchor)
        targets=ctx['aim_target_w'].clone();env.ep_step+=1;curriculum.apply_commands(env,3)
        torch.testing.assert_close(ctx['aim_target_w'],targets)
        env.ep_step.zero_();curriculum.apply_commands(env,3,validation=True)
        self.assertEqual(env.ep_step.tolist(),[0,250,500,750,1000,1250])
        self.assertAlmostEqual(float(ctx['velocity_command'][3,0]),.3,places=6)

    def test_retarget_uses_current_chest_and_original_yaw(self):
        task=self.task;initial=task.initial_state(1);ctx=task.context(initial,torch.tensor([[4.,0.,0.,.05]]))
        state=(initial[0]+torch.tensor([2.,1.,0.]),so3_exp(torch.tensor([[0.,0.,torch.pi/2]])),*initial[2:])
        probe=task.context(state,torch.tensor([[3.,0.,0.,.05]]));chest=probe['aim_target_w']-3*probe['aim_normal_w']
        curriculum.retarget(task,state,ctx,torch.tensor([[4.,0.,0.,.05]]),torch.tensor([0]))
        torch.testing.assert_close(ctx['aim_target_w'],chest+torch.tensor([[4.,0.,0.]]))
        torch.testing.assert_close(ctx['aim_normal_w'],torch.tensor([[1.,0.,0.]]))
        self.assertEqual(len(curriculum.grid(3,'cpu')),25)

    def test_moving_initialization_requires_two_full_standing_acceptances(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'test.pt';p.write_bytes(b'test checkpoint identity')
            report=dict(checkpoint_sha256=sha256(p),scope='Full standing aiming acceptance',level=None,passed=True,
                fixed_grid_all_pass=True,fixed_count=75,random_count=100,random_success_rate=.95,seconds=30,
                continuous_without_reset=True,reference_free=True,online_target_switches=True,hit_radius_m=.05)
            verify_standing_acceptance(p,[report|dict(seed=1),report|dict(seed=2)])
            for key,value in [('level',6),('hit_radius_m',.1),('random_success_rate',.94),('seconds',10)]:
                with self.assertRaises(ValueError):verify_standing_acceptance(p,[report|dict(seed=1),report|dict(seed=2)|{key:value}])
            with self.assertRaises(ValueError):verify_standing_acceptance(p,[report|dict(seed=1)]*2)


if __name__=='__main__':unittest.main()
