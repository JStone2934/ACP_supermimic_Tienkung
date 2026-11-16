import unittest
import numpy as np
import torch
from supermimic.rotations import so3_exp
from supermimic.acp.chest_posture import ChestPosture,apply_posture_acceptance,verify_posture_reports
from supermimic.acp.aiming_fullbody import FullbodyTask


class ChestPostureTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);self.posture=ChestPosture({})

    def test_low_target_only_permits_forward_lean_and_always_prefers_upright(self):
        angles=torch.deg2rad(torch.tensor([[0.,0.,0.],[0.,20.,0.],[20.,0.,0.],[0.,-20.,0.]]))
        R=so3_exp(angles);normal=torch.tensor([[1.,0.,0.]]).expand(4,-1)
        low=torch.tensor([[.8660254,0.,-.5]]).expand(4,-1)
        regular,_=self.posture.measure(R,normal);relaxed,reward=self.posture.measure(R,low)
        self.assertEqual(regular['chest_posture_valid'].tolist(),[True,False,False,False])
        self.assertEqual(relaxed['chest_posture_valid'].tolist(),[True,True,False,False])
        self.assertGreater(float(reward['chest_upright'][0]),float(reward['chest_upright'][1]))
        torch.testing.assert_close(relaxed['chest_limit_deg'],torch.full((4,),25.))

    def test_native_chest_detects_bent_waist_with_vertical_pelvis(self):
        task=FullbodyTask('cpu',{'chest_posture':{}});state=list(task.initial_state(1))
        names=task.robot.joint_names
        waist=next(i for i,n in enumerate(names) if 'waist' in n and 'pitch' in n)
        state[4][:]=0;state[4][:,waist]=.55
        context=task.context(state,torch.tensor([[3.,0.,0.,.05]]))
        rotation=task.kin.forward(*state)[1][:,task.torso]
        measures,_=task.chest_posture.measure(rotation,context['aim_normal_w'])
        self.assertEqual(float(state[1][0,2,2]),1.)
        self.assertGreater(float(measures['chest_tilt_deg'][0]),30.)
        self.assertFalse(bool(measures['chest_posture_valid'][0]))
        _,terms=task.reward_terms(state,torch.zeros(1,29),torch.zeros(1,29),context)
        self.assertIn('chest_upright',terms);self.assertLess(float(terms['chest_excess'][0]),0)
        legacy=FullbodyTask('cpu',{})
        self.assertNotIn('chest_posture',legacy.metadata())
        self.assertNotIn('chest_upright',legacy.weights)

    def test_reward_gradient_finite_at_upright_and_bent(self):
        v=torch.tensor([[0.,0.,0.],[0.,.55,0.]],requires_grad=True)
        _,terms=self.posture.measure(so3_exp(v),torch.tensor([[1.,0.,0.]]).expand(2,-1))
        (terms['chest_upright']-terms['chest_excess']).sum().backward()
        self.assertTrue(bool(v.grad.isfinite().all()))
        self.assertLess(float(v.grad[1,1]),0)

    def test_switch_return_and_bad_initial_pose_fail_acceptance(self):
        tilt=np.full((1501,1),5.);tilt[500:1150]=22.
        limit=np.full_like(tilt,10.);limit[500:1000]=25.
        arrays=dict(chest_tilt_deg=tilt,chest_forward_deg=tilt,chest_lateral_deg=tilt*0,
                    chest_limit_deg=limit,chest_posture_valid=tilt<=limit)
        self.assertTrue(self.posture.episode(arrays,0,switching=True)['passed'])
        arrays['chest_posture_valid'][1150:]=False
        scenarios=[{'passed':True}]
        task=type('Task',(),{'chest_posture':self.posture})()
        apply_posture_acceptance(task,arrays,scenarios,lambda i:True)
        self.assertFalse(scenarios[0]['passed'])
        tilt[0]=31.
        self.assertFalse(self.posture.episode(arrays,0,switching=True)['initial_upright'])

    def test_legacy_and_relaxed_evidence_cannot_be_delivered(self):
        with self.assertRaises(ValueError):verify_posture_reports([{'passed':True}])
        spec=self.posture.metadata();spec['normal_limit_deg']=15.
        with self.assertRaises(ValueError):verify_posture_reports([{'task_config':{'chest_posture':spec}}])
        with self.assertRaises(ValueError):ChestPosture({'low_target_start_deg':-40.})


if __name__=='__main__':unittest.main()
