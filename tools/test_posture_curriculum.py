import tempfile,unittest,json
from pathlib import Path
import torch
from supermimic.acp.posture_curriculum import mixed_reset_state,next_phase,conditions,validate_evidence
from supermimic.acp.wide_curriculum import grid
from supermimic.acp.robots import sha256
from tools.run_posture_curriculum import next_target,write_json,matches_phase

class CurriculumTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.s=(torch.zeros(100,3),torch.eye(3).repeat(100,1,1),torch.zeros(100,3),torch.zeros(100,3),torch.zeros(100,29),torch.zeros(100,29))
        self.ref=[0,0,1,1,0,0,0]+[.1]*29
    def test_reset_endpoints_and_overrides(self):
        mix=dict(probability=0.,reference_pose=self.ref)
        for a,b in zip(self.s,mixed_reset_state(self.s,mix)):self.assertTrue(torch.equal(a,b))
        r=mixed_reset_state(self.s,mix,'reference');self.assertTrue((r[0][:,2]==1).all());self.assertTrue((r[4]==.1).all())
        mix['probability']=1
        r=mixed_reset_state(self.s,mix);self.assertTrue((r[4]==.1).all())
        for a,b in zip(self.s,mixed_reset_state(self.s,mix,'old')):self.assertTrue(torch.equal(a,b))
        self.assertTrue((self.s[4]==0).all())
    def test_mixture_is_selection_not_interpolation(self):
        torch.manual_seed(17);r=mixed_reset_state(self.s,dict(probability=.5,reference_pose=self.ref))
        self.assertTrue(((r[4]==0)|(r[4]==.1)).all());self.assertGreater(int((r[4][:,0]>.05).sum()),10);self.assertLess(int((r[4][:,0]>.05).sum()),90)
    def test_invalid_pose_and_condition(self):
        with self.assertRaises(ValueError):mixed_reset_state(self.s,dict(probability=2,reference_pose=self.ref))
        with self.assertRaises(ValueError):mixed_reset_state(self.s,None,'reference')
    def test_unbounded_blocks_and_order(self):
        self.assertEqual(next_target(100,100),200);self.assertEqual(next_target(100000,100000),100100)
        with self.assertRaises(ValueError):next_target(100,99)
        self.assertEqual(next_phase(0,0),(0,1));self.assertEqual(next_phase(0,4),(1,4));self.assertIsNone(next_phase(6,4))
        self.assertEqual(conditions(0,1),['old','reference']);self.assertEqual(conditions(1,4),['reference'])
    def test_report_identity_seed_and_reset_guards(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'ck.pt';p.write_bytes(b'identity');n=len(grid(0,'cpu'))
            r=dict(seed=17,checkpoint_sha256=sha256(p),level=0,passed=True,fixed_grid_all_pass=True,random_success_rate=1.,seconds=30.,continuous_without_reset=True,reference_free=True,fixed_count=n,random_count=100,scope='Intermediate curriculum only',scenarios=[dict(passed=True)]*(n+100),reset_pose_condition='old')
            reports=[r,r|{'seed':53}];validate_evidence(p,reports,0,'old')
            for changes in [dict(seed=17),dict(checkpoint_sha256='wrong'),dict(reset_pose_condition='reference'),dict(passed=False)]:
                with self.assertRaises(ValueError):validate_evidence(p,[r,reports[1]|changes],0,'old')
    def test_atomic_state_and_phase_resume(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'state.json';s=dict(phase_id=2,level=0,reset_index=2)
            write_json(path,s);self.assertEqual(json.loads(path.read_text()),s)
            c=dict(config=dict(formal_phase_id=2,curriculum_level=0,reset_mixture=dict(probability=.25)))
            self.assertTrue(matches_phase(c,s));self.assertFalse(matches_phase(c,s|{'phase_id':3}))
    def test_target_switch_uses_actual_episode_pose(self):
        from unittest.mock import patch
        from supermimic.acp.train_wide_aiming import switch_targets
        class Task:
            alpha=torch.zeros(1)
            def initial_state(self,n):raise AssertionError('Must not resample a reset')
            def context(self,s,params,weight):
                return dict(aim_target_w=s[4][:,:3],aim_normal_w=s[1][:,:,0],aim_tolerance=params[:,:2])
            def release_support(self,ctx,ids):pass
        ctx=dict(root_anchor_w=torch.zeros(100,3),aim_target_w=torch.zeros(100,3),aim_normal_w=torch.zeros(100,3),aim_tolerance=torch.zeros(100,2))
        actual=list(self.s);actual[4]=torch.arange(100).float()[:,None].expand(100,29)
        ids=torch.tensor([3,8])
        with patch('supermimic.acp.train_wide_aiming.sample',return_value=(torch.zeros(2,4),4.)):
            switch_targets(Task(),ctx,ids,6,tuple(actual))
        self.assertTrue(torch.equal(ctx['aim_target_w'][ids,0],torch.tensor([3.,8.])))

if __name__=='__main__':unittest.main()
