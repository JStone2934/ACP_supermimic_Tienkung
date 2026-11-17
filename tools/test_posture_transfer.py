import unittest
import torch
from supermimic.acp.posture_transfer import reference_reward,retention_cost


class TransferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_22_joint_reference_includes_left_arm_excludes_entire_right_arm(self):
        from supermimic.acp.robots import get_robot
        from supermimic.acp.standing import ARM_NAMES
        robot=get_robot('tienkung3_29dof')
        ids=torch.tensor([i for i,n in enumerate(robot.joint_names) if n not in ARM_NAMES])
        arm=torch.tensor([robot.joint_names.index(n) for n in ARM_NAMES])
        self.assertEqual(len(ids),22);self.assertEqual(len(arm),7)
        q=torch.ones(2,29,requires_grad=True);reference=torch.zeros(29)
        reference_reward(q,reference,.35,ids).sum().backward()
        self.assertTrue((q.grad[:,ids]<0).all())
        self.assertEqual(int(torch.count_nonzero(q.grad[:,arm])),0)
        changed=q.detach().clone();changed[:,arm]=100
        self.assertTrue(torch.equal(reference_reward(q.detach(),reference,.35,ids),reference_reward(changed,reference,.35,ids)))

    def test_original_task_rewards_unchanged(self):
        from supermimic.acp.aiming_fullbody import FullbodyTask
        from supermimic.acp.standing import ARM_NAMES
        old=FullbodyTask('cpu')
        names=[n for n in old.robot.joint_names if n not in ARM_NAMES]
        new=FullbodyTask('cpu',dict(feet_position_scale_m=old.feet_position_scale_m,
            posture_reference=dict(q=old.initial_q.tolist(),scale_rad=.35,joint_names=names),
            reward_weights=dict(posture_reference=.2)))
        state=old.initial_state(2);state[4].add_(.01)
        ctx=old.context(state,torch.tensor([[3.,0.,0.,.1]]).expand(2,-1))
        action=torch.zeros(2,29)
        before=old.reward_terms(state,action,action,ctx)[1]
        after=new.reward_terms(state,action,action,ctx)[1]
        self.assertEqual(set(after)-set(before),{'posture_reference'})
        for name,value in before.items():self.assertTrue(torch.equal(value,after[name]),name)
        self.assertFalse(new.chest_posture.enabled)
        self.assertFalse(hasattr(new,'retention_teacher'))

    def test_reference_moves_body_toward_target_without_arm_gradient(self):
        q=torch.ones(2,29,requires_grad=True);ref=torch.zeros(29)
        r=reference_reward(q,ref,.35);r.sum().backward()
        self.assertTrue((q.grad[:,:15]<0).all())
        self.assertEqual(int(torch.count_nonzero(q.grad[:,15:])),0)
        self.assertTrue((reference_reward(torch.zeros_like(q),ref,.35)==1).all())

    def test_teacher_is_detached_and_policy_pulled_back(self):
        a=torch.ones(2,29,requires_grad=True);b=torch.zeros(2,29,requires_grad=True)
        loss=retention_cost(a,b,torch.ones(29),2).sum();loss.backward()
        self.assertIsNone(b.grad);self.assertTrue((a.grad>0).all())
        self.assertEqual(float(retention_cost(b,b,torch.ones(29),2).sum().detach()),0)


if __name__=='__main__':unittest.main()
