"""Small, task-specific wiring checks. No performance or disturbance matrix."""
import json,unittest
from pathlib import Path
import numpy as np
import torch
from supermimic.acp.aiming import setup,DEFAULT_CONFIG,policy
from supermimic.acp.aiming_curriculum import sample_parameters,Curriculum
from supermimic.acp.buffer import ReplayBuffer

class AimingWiring(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4);cls.task=setup(json.loads(Path(DEFAULT_CONFIG).read_text()),'cpu')
    def test_geometry_action_and_gradient(self):
        t=self.task;S=t.mdp.motion.ref_state(torch.zeros(1,dtype=torch.long));h,d,_=t.hand(S)
        g=torch.cat((h+3*d,torch.zeros(1,3),torch.full((1,2),.2)),-1)
        e,angle,valid,_=t.metrics(S,g);self.assertTrue(valid.all());self.assertLess(float(e.max()),1e-5)
        reverse=g.clone();reverse[:,:3]=h-3*d
        self.assertFalse(t.metrics(S,reverse)[2].any())
        a=torch.zeros(1,29,requires_grad=True);xi=torch.zeros(1,33);prev=S[4][:,t.terminal]
        result,q=t.compose_action(a,xi,prev);target=t.mdp.default_q+t.alpha*result
        torch.testing.assert_close(target[:,t.fixed],t.target_q[t.fixed][None]);result.sum().backward()
        self.assertTrue((a.grad[:,t.fixed]==0).all());self.assertTrue((a.grad[:,t.free]!=0).all())
        large,_=t.compose_action(torch.full((1,29),10.),xi,prev)
        delta=t.mdp.default_q[t.terminal]+t.alpha[t.terminal]*large[:,t.terminal]-prev
        self.assertLessEqual(float(delta.abs().max()),.040001)
        moved=list(S);moved[0]=(S[0]+torch.tensor([[.2,.1,0]])).requires_grad_();terms=t.terms(tuple(moved),torch.zeros(1,dtype=torch.long));(terms['root']+terms['feet']).sum().backward();self.assertTrue((moved[0].grad[:,:2]<0).all())
    def test_targets_replay_and_migration(self):
        t=self.task;p=sample_parameters(dict(mode=0,depth=(3,3),angle=(0,0),rho=.2),2,np.random.default_rng(0),'cpu')
        torch.testing.assert_close(t.target(p,torch.zeros(2)),t.target(p,torch.ones(2)*29))
        p[:,0]=1;nodes=p[:,16:].reshape(2,8,6);nodes[:,1,0]=1;nodes[:,1,1]=.1
        self.assertTrue((t.target(p,torch.ones(2))[:,3:6]==0).all())
        nodes[:,1,0]=.1
        torch.testing.assert_close(t.target(p,torch.full((2,),.1)),t.target(p,torch.full((2,),5.)*.02))
        b=ReplayBuffer(3,2,'cpu',extra={'aim_params':(64,torch.float32),'terminal_prev':(3,torch.float32)},robot=t.mdp.robot)
        tr={k:torch.zeros_like(v[0]) for k,v in b.data.items()};tr['aim_params']=p;tr['terminal_prev'][:]=.123;b.add(tr)
        s,_=b.sample_starts(4,10);self.assertTrue((s['terminal_prev']==.123).all());self.assertEqual(s['aim_params'].shape,(4,64))
        ck=torch.load(t.config['source'],map_location='cpu',weights_only=False);pi=policy(t);pi.migrate(ck['pi'],t)
        torch.testing.assert_close(pi.goal_norm.mean,ck['pi']['goal_norm.mean']);self.assertTrue((pi.enc[0].weight[:,-17:]==0).all())

if __name__=='__main__':unittest.main()
