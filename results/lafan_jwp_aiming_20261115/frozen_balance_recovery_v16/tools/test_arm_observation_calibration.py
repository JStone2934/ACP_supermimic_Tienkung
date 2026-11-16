import unittest
import torch
from supermimic.acp.aiming_fullbody import FullbodyPolicy
from tools.prepare_arm_observation_calibration import folded_observation_state


class ArmObservationFoldTests(unittest.TestCase):
    def test_matches_explicit_mask_for_every_history_slot_and_retains_other_inputs(self):
        torch.set_num_threads(2);torch.manual_seed(31);p=FullbodyPolicy()
        with torch.no_grad():
            p.obs_norm.mean.copy_(torch.randn(99));p.obs_norm.var.copy_(torch.rand(99)+.1)
        before={k:v.clone() for k,v in p.state_dict().items()}
        joints=[p.robot.joint_names.index(n) for n in ('elbow_yaw_r_joint','wrist_pitch_r_joint','wrist_roll_r_joint')]
        columns=joints+[29+i for i in joints]+[64+i for i in joints];values=torch.randn(len(columns))
        state,flat=folded_observation_state(p,columns,values);q=FullbodyPolicy();q.load_state_dict(state)
        h=torch.randn(4,10,99);g=torch.randn(4,107);masked=h.clone();masked[:,:,columns]=values
        with torch.no_grad():
            torch.testing.assert_close(p(masked,g),q(h,g),atol=1e-7,rtol=1e-5)
            changed=h.clone();changed[:,:,columns]+=torch.randn_like(changed[:,:,columns])*10
            torch.testing.assert_close(q(changed,g),q(h,g),atol=0,rtol=0)
            changed=h.clone();changed[:,:,0]+=.5
            self.assertGreater(float((q(changed,g)-q(h,g)).abs().max()),0)
        self.assertEqual(len(flat),10*len(columns))
        for k,v in p.state_dict().items():torch.testing.assert_close(v,before[k],atol=0,rtol=0)
        for k,v in state.items():
            if k not in ('dec.0.weight','dec.0.bias'):torch.testing.assert_close(v,before[k],atol=0,rtol=0)

    def test_rejects_invalid_columns_and_values(self):
        p=FullbodyPolicy()
        for columns,values in (([],[]),([0,0],[1,2]),([-1],[0]),([99],[0]),([0],[float('nan')]),([0],[1,2])):
            with self.assertRaises(ValueError):folded_observation_state(p,columns,values)


if __name__=='__main__':unittest.main()
