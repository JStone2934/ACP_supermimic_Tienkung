import unittest
from unittest.mock import patch
import numpy as np
from supermimic.acp.native_pose_polish import polish
from supermimic.acp.robots import get_robot
from tools.polish_php_conversion import verify_seed

class NativePolishTests(unittest.TestCase):
    def test_clear_motion_is_exact_identity(self):
        robot=get_robot('tienkung3_29dof')
        motion=np.tile(np.r_[[0.,0.,2.],[1.,0.,0.,0.],robot.default_q],(3,1))
        changed,stats=polish(robot,motion,np.empty((0,8)),np.zeros((3,9),bool),np.zeros((3,9,3)))
        np.testing.assert_array_equal(changed,motion)
        self.assertEqual(stats['selected_frames'],[])

    def test_invalid_arrays_are_rejected(self):
        with self.assertRaises(ValueError):
            polish(None,np.zeros((2,36)),None,np.zeros((1,9),bool),np.zeros((2,9,3)))
        motion=np.zeros((2,36));motion[1,0]=np.nan
        with self.assertRaises(ValueError):
            polish(None,motion,None,np.zeros((2,9),bool),np.zeros((2,9,3)))

    def test_changed_seed_receipt_or_terrain_is_rejected(self):
        receipt=dict(parent='/not-read',parent_task_id='seed',motion_sha256='original')
        parent=dict(record={'key':'category/name','source_hash':'source'},solve_variant=2)
        task=dict(record=dict(parent['record']),solve_variant=2,seed=dict(receipt))
        with patch('tools.polish_php_conversion.seed_receipt',return_value=(receipt,parent,{})):
            task['seed']['motion_sha256']='changed'
            with self.assertRaises(ValueError):verify_seed(task)
            task['seed']=dict(receipt);task['solve_variant']=1
            with self.assertRaises(ValueError):verify_seed(task)
            task['solve_variant']=2;task['record']['source_hash']='different source'
            with self.assertRaises(ValueError):verify_seed(task)

if __name__=='__main__':unittest.main()
