import unittest
import torch
from supermimic.acp.aiming_fullbody import FullbodyPolicy
from tools.prepare_support_gain_calibration import scaled_feedback_state


class GainCalibrationTests(unittest.TestCase):
    def test_preserves_original_and_zero_feedback_function(self):
        torch.set_num_threads(2);torch.manual_seed(12);source=FullbodyPolicy()
        source.goal_norm.mean[:6].zero_()
        original={k:v.clone() for k,v in source.state_dict().items()}
        scaled=scaled_feedback_state(source.state_dict(),2.)
        target=FullbodyPolicy();target.load_state_dict(scaled)
        for k,v in source.state_dict().items():
            torch.testing.assert_close(v,original[k],atol=0,rtol=0)
            if k=='enc.0.weight':
                torch.testing.assert_close(scaled[k][:,6:],v[:,6:],atol=0,rtol=0)
                torch.testing.assert_close(scaled[k][:,:6],2*v[:,:6],atol=0,rtol=0)
            else:torch.testing.assert_close(scaled[k],v,atol=0,rtol=0)
        history=torch.randn(4,10,99);command=torch.randn(4,107);command[:,:6]=0
        with torch.no_grad():
            torch.testing.assert_close(source(history,command),target(history,command),atol=0,rtol=0)
            command[:,:6]=.02
            self.assertGreater(float((source(history,command)-target(history,command)).abs().max()),0.)

    def test_invalid_gain_rejected(self):
        for gain in (0.,-1.,float('nan'),float('inf')):
            with self.assertRaises(ValueError):scaled_feedback_state({},gain)

    def test_lateral_uses_each_foot_base_y_column_only(self):
        state={'enc.0.weight':torch.arange(2*107,dtype=torch.float32).reshape(2,107),'other':torch.ones(4)}
        result=scaled_feedback_state(state,3.,'lateral')
        torch.testing.assert_close(result['enc.0.weight'][:,[1,4]],3*state['enc.0.weight'][:,[1,4]],atol=0,rtol=0)
        self.assertEqual(int(torch.count_nonzero(result['enc.0.weight'][:,[0,2,3,5]])),0)
        torch.testing.assert_close(result['enc.0.weight'][:,6:],state['enc.0.weight'][:,6:],atol=0,rtol=0)
        torch.testing.assert_close(result['other'],state['other'],atol=0,rtol=0)


if __name__=='__main__':unittest.main()
