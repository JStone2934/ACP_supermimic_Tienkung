import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from supermimic.acp.robots import get_robot,sha256
from supermimic.acp.aiming_wide import WideAimingTask
from supermimic.acp.aiming_targets import fixed_target_grid
from supermimic.acp.tracking_transfer import fixed_support_command
from supermimic.acp.wide_curriculum import grid,STAGES
from supermimic.acp.wide_aiming_validation import episode_metrics
from supermimic.acp.wide_aiming_validation import segment_metrics
from supermimic.acp.train_wide_aiming import check_promotion,switch_targets
from supermimic.rotations import yaw_mat


class WideTrainingTests(unittest.TestCase):
    def test_initial_adjustment_can_finish_while_ray_already_hits(self):
        error=np.zeros(501);feet=np.zeros((501,2,3))
        feet[:101,:,0]=np.linspace(0,.2,101)[:,None];feet[101:,:,0]=.2
        result=segment_metrics(error,feet)
        self.assertTrue(result['passed']);self.assertLessEqual(result['capture_s'],3.)
        self.assertLessEqual(result['stable_foot_drift_m'],.05)
        # The same displacement occurring after the allowed capture window fails.
        feet[:301,:,0]=np.linspace(0,.4,301)[:,None];feet[301:,:,0]=.4
        self.assertFalse(segment_metrics(error,feet)['passed'])

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2);r=get_robot('tienkung3_29dof')
        cls.task=WideAimingTask('cpu',dict(initial_q=list(r.default_q),initial_height=r.stand_height,
                                          physics_substeps=16,switch_interval_steps=500))

    def test_teacher_support_points_are_on_ground_and_translation_invariant(self):
        task=self.task;S=task.initial_state(2);ctx=task.context(S,torch.tensor([[3.,0.,0.,.05]]).expand(2,-1))
        command=fixed_support_command(task,S,ctx);self.assertEqual(command.shape,(2,93))
        local=command[:,:27].reshape(2,9,3)
        world=(yaw_mat(S[1])[:,None]@local[...,:2,:].unsqueeze(-1)).squeeze(-1)+S[0][:,None]
        torch.testing.assert_close(world[...,2],torch.zeros(2,2),atol=1e-6,rtol=0)
        self.assertTrue((local[:,2:]==0).all())
        delta=torch.tensor([2.,-3.,0.]);ctx['root_anchor_w']+=delta
        shifted=(S[0]+delta,*S[1:])
        torch.testing.assert_close(fixed_support_command(task,shifted,ctx),command,atol=2e-6,rtol=0)

    def test_final_grid_matches_acceptance_and_switch_does_not_move_anchor(self):
        torch.testing.assert_close(grid(len(STAGES)-1,'cpu'),fixed_target_grid('cpu'),atol=1e-6,rtol=0)
        task=self.task;S=task.initial_state(2);ctx=task.context(S,torch.tensor([[3.,0.,0.,.05]]).expand(2,-1))
        anchor=ctx['root_anchor_w'].clone();switch_targets(task,ctx,torch.arange(2),len(STAGES)-1)
        torch.testing.assert_close(ctx['root_anchor_w'],anchor,atol=0,rtol=0)
        ends=task.soft_termination(S,torch.tensor([499,500]),ctx)
        self.assertEqual(float(ends[1]),1.);self.assertLess(float(ends[0]),1.)

    def test_loose_curriculum_cannot_change_default_final_tolerance(self):
        n=1501;root=np.zeros((n,3));feet=np.zeros((n,2,3));tilt=np.zeros(n);error=np.full(n,.1);finite=np.ones(n,bool)
        self.assertFalse(episode_metrics(root,feet,tilt,error,finite,False)['passed'])
        self.assertTrue(episode_metrics(root,feet,tilt,error,finite,False,radius=.2)['passed'])
        self.assertTrue(episode_metrics(root,feet,tilt,error,finite,False,aim_required=False)['passed'])
        root[-1,0]=.51
        self.assertFalse(episode_metrics(root,feet,tilt,error,finite,False,aim_required=False)['passed'])

    def test_promotion_requires_distinct_evaluations_of_exact_source(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'source.pt';p.write_bytes(b'model')
            r=dict(checkpoint_sha256=sha256(p),level=0,passed=True,fixed_grid_all_pass=True,random_success_rate=1.)
            check_promotion(p,[r|{'seed':1},r|{'seed':2}],0)
            with self.assertRaises(ValueError):check_promotion(p,[r|{'seed':1}]*2,0)
            with self.assertRaises(ValueError):check_promotion(p,[r|{'seed':1},r|{'seed':2,'random_success_rate':.94}],0)


if __name__=='__main__':unittest.main()
