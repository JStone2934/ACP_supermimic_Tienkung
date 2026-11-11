"""CPU regression tests for reset mixtures, promotion gates, resume and two-rank synchronization."""
import copy
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from supermimic.acp.transition_curriculum import TransitionCfg, TransitionCurriculum
from tools.build_transition_curriculum import event_windows, serialize_plan


class Contacts:
    def __init__(self):
        self.labels=torch.zeros(3000,1,9,dtype=torch.bool)
        self.labels[:,:, [0,1,4]]=True
        self.distance=torch.zeros(9)
    def rv(self,v):return v
    def body_distance(self,S,boxes):return self.distance.to(S[0].device).expand(len(S[0]),-1)
    def contact_point(self,S,boxes):return S[0][:,None].expand(-1,9,-1)


def fixture(n=100, runtime=None):
    names=['climb/a','climb/b','walk/c'];length=torch.tensor([1000]*3)
    robot=SimpleNamespace(control_dt=.02, contact_bodies=dict.fromkeys(['left_foot','right_foot','left_knee','right_knee','left_hand','right_hand','left_elbow','right_elbow','pelvis']))
    m=SimpleNamespace(names=names,robot=robot,data_hash='fixture',clip_len=length,
        clip_start=torch.tensor([0,1000,2000]),clip_end=torch.tensor([999,1999,2999]),
        clip_id=torch.arange(3).repeat_interleave(1000),frame_bin=torch.arange(3000)//50,
        root_p=torch.zeros(3000,3),q=torch.zeros(3000,2),boxes=lambda f,v:torch.zeros(len(f),1,8))
    event=dict(start=85,anchor=100,end=130)
    plan=dict(version=1,data_hash='fixture',control_dt=.02,groups=['climb'],config={},clips={
        'climb/a':dict(frames=1000,events=[event,dict(start=185,anchor=200,end=230)]),
        'climb/b':dict(frames=1000,events=[event])})
    return TransitionCurriculum(plan,m,Contacts(),n,'cpu',runtime)


def prepare_local(c,n=1):
    c.kind[:]=0;c.window[:]=0;c.window_start[:]=85;c.episode_end[:]=130
    c.checked_steps[:]=10;c.good_steps[:]=10


def worker(root):
    from supermimic.acp.distributed import Runtime
    rt=Runtime(SimpleNamespace(device='cpu',log_dir=root,seed=7,num_envs=4),backend='gloo')
    c=fixture(2,rt);prepare_local(c)
    c.cfg=TransitionCfg(promotion_episodes=2)
    done=torch.full((2,),rt.rank==0)
    c.finish(torch.full((2,),130),done,torch.zeros(2,dtype=torch.bool))
    assert c.start[0]==77
    states=rt.objects(c.state_dict())
    for k in ('start','attempts','successes','promotions','totals','quality_counts'):
        torch.testing.assert_close(states[0][k],states[1][k],rtol=0,atol=0)
    # Empty-ended rank still participates; in-flight old-window successes cannot advance again.
    c.finish(torch.full((2,),130),torch.ones(2,dtype=torch.bool),torch.zeros(2,dtype=torch.bool))
    assert c.start[0]==77
    rt.close()


class Tests(unittest.TestCase):
    def test_sampling_mixture_and_uniform_clips(self):
        torch.manual_seed(1);c=fixture(20000);ids=torch.arange(20000)
        f,_=c.sample(ids,torch.tensor([.4,.4,0.]))
        rates=torch.bincount(c.kind,minlength=3)/len(ids)
        torch.testing.assert_close(rates,torch.tensor([.5,.25,.25]),atol=.012,rtol=0)
        clips=c.motion.clip_id[f];offset=f-c.motion.clip_start[clips]
        self.assertTrue((offset[c.kind==1]==0).all())
        self.assertTrue((clips[c.kind!=2]<2).all())
        local_clips=clips[c.kind==0]
        self.assertLess(abs(float((local_clips==0).float().mean())-.5),.02)
        self.assertTrue((offset[(c.kind==2)&(clips<2)]>=599).all())
        self.assertTrue((c.episode_end[c.kind!=0]==c.motion.clip_end[clips[c.kind!=0]]).all())
        self.assertTrue(((c.episode_end-f)>0).all())
        # Repeated short local episodes cannot consume full/retention environment slots.
        original_kind=c.kind.clone()
        for _ in range(3):
            c.sample((c.kind==0).nonzero().flatten(),torch.tensor([.4,.4,0.]))
            self.assertTrue(torch.equal(c.kind,original_kind))

    def test_failed_window_does_not_retreat(self):
        c=fixture();original=c.start.clone()
        c.update_windows(torch.full_like(c.start,20),torch.zeros_like(c.start))
        torch.testing.assert_close(c.start,original)
        c.update_windows(torch.full_like(c.start,20),torch.full_like(c.start,16))
        torch.testing.assert_close(c.start,original-8)
        self.assertTrue((c.attempts==0).all())

    def test_terminal_failure_never_promotes(self):
        c=fixture(20);prepare_local(c)
        c.finish(torch.full((20,),130),torch.ones(20,dtype=torch.bool),torch.ones(20,dtype=torch.bool))
        self.assertEqual(c.start[0],85);self.assertEqual(c.totals[0,2],0)

    def test_gate_checks_stability_not_just_survival(self):
        c=fixture(20);prepare_local(c);c.good_steps[:]=7
        c.finish(torch.full((20,),130),torch.ones(20,dtype=torch.bool),torch.zeros(20,dtype=torch.bool))
        self.assertEqual(c.start[0],85)
        c.good_steps[:]=8
        c.finish(torch.full((20,),130),torch.ones(20,dtype=torch.bool),torch.zeros(20,dtype=torch.bool))
        self.assertEqual(c.start[0],77)
        c.finish(torch.full((20,),130),torch.ones(20,dtype=torch.bool),torch.zeros(20,dtype=torch.bool))
        self.assertEqual(c.start[0],77)  # stale episodes

    def test_quality_missing_hand_slip_and_root_error(self):
        c=fixture(1);prepare_local(c);c.good_steps.zero_();c.checked_steps.zero_()
        S=(torch.zeros(1,3),torch.eye(3)[None],torch.zeros(1,3),torch.zeros(1,3),torch.zeros(1,2),torch.zeros(1,2))
        f=torch.tensor([125]);v=torch.tensor([0])
        c.observe(S,S,f,v);self.assertEqual(c.good_steps[0],1)
        c.contacts.distance[4]=.1
        c.observe(S,S,f,v);self.assertEqual(c.good_steps[0],1)  # feet do not hide missing hand
        c.contacts.distance[4]=0
        shifted=(S[0]+.01,*S[1:])
        c.observe(S,shifted,f,v);self.assertEqual(c.good_steps[0],1)  # slip
        shifted=(S[0]+.3,*S[1:])
        c.observe(shifted,shifted,f,v);self.assertEqual(c.good_steps[0],1)  # root error
        self.assertEqual(c.checked_steps[0],4)

    def test_resume_preserves_progress_and_rejects_identity_change(self):
        c=fixture();c.update_windows(torch.full_like(c.start,20),torch.full_like(c.start,20))
        buffer=io.BytesIO();torch.save(c.state_dict(),buffer);buffer.seek(0)
        d=fixture();d.load_state_dict(torch.load(buffer,weights_only=False));torch.testing.assert_close(c.start,d.start)
        bad=copy.deepcopy(c.state_dict());bad['plan_hash']='wrong'
        with self.assertRaisesRegex(ValueError,'different transition'):d.load_state_dict(bad)
        bad=c.state_dict();bad['start'][0]=999
        with self.assertRaisesRegex(ValueError,'past its original'):d.load_state_dict(bad)

    def test_invalid_config_and_missing_clip(self):
        for kw in [dict(local_fraction=.9),dict(promotion_rate=1.1),dict(stable_s=0),dict(advance_s=float('nan'))]:
            with self.assertRaises(ValueError):TransitionCfg(**kw)
        c=fixture();p=copy.deepcopy(c.plan);p['clips'].pop('climb/b')
        with self.assertRaisesRegex(ValueError,'every clip'):TransitionCurriculum(p,c.motion,c.contacts,2,'cpu')

    def test_events_include_failure_and_hand_switch(self):
        labels=np.zeros((300,9),dtype=bool);labels[:,:2]=True;labels[150:200,4:6]=True
        events=event_windows(labels,130,True,.02)
        anchors={e['anchor'] for e in events};self.assertIn(130,anchors);self.assertIn(150,anchors)
        for e in events:self.assertTrue(0<=e['start']<=e['anchor']<e['end']<300)
        with self.assertRaisesRegex(ValueError,'neither'):event_windows(labels[:100],99,False,.02)

    def test_generated_plan_roundtrip(self):
        c=fixture()
        self.assertEqual(json.loads(serialize_plan(c.plan)),c.plan)

    def test_two_rank_empty_peer_and_promotion(self):
        with tempfile.TemporaryDirectory() as root, socket.socket() as sock:
            sock.bind(('127.0.0.1',0));port=sock.getsockname()[1];sock.close()
            cmd=[sys.executable,'-m','torch.distributed.run','--nproc-per-node=2',f'--master-port={port}',
                 '-m','tools.test_transition_curriculum','--worker',root]
            r=subprocess.run(cmd,capture_output=True,text=True,timeout=90,env=os.environ|{'OMP_NUM_THREADS':'1'})
            self.assertEqual(r.returncode,0,r.stdout+r.stderr)

if __name__=='__main__':
    if '--worker' in sys.argv:worker(sys.argv[-1])
    else:unittest.main()
