"""Actual free-base simulation: support capture and inference/training command parity."""
import argparse,json
from pathlib import Path
from unittest.mock import patch
import torch
from supermimic.acp.aiming_wide import WideAimingTask
from supermimic.acp.aiming_fullbody import FullbodyPolicy,validate_checkpoint
from supermimic.acp.aiming_fullbody_env import FullbodySim
from supermimic.acp.aiming_policy import AimingPolicy
from supermimic.acp.robots import sha256


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args()
    torch.set_num_threads(2);torch.manual_seed(941);digest=sha256(a.checkpoint)
    ck=torch.load(a.checkpoint,map_location=a.device,weights_only=False);task=WideAimingTask(a.device,ck['config']);validate_checkpoint(ck,task)
    assert task.support_capture_steps==150
    policy=FullbodyPolicy(task.robot).to(a.device);policy.load_state_dict(ck['pi']);policy.eval()
    params=torch.tensor([[3.,0.,0.,.2]],device=a.device).expand(8,-1)
    with patch('supermimic.acp.motion.MotionLib.__init__',side_effect=AssertionError('No motion reads')):
        env=FullbodySim(task,8,lambda n:(params,0.),eval_mode=True);runtime=AimingPolicy(a.checkpoint,a.device);runtime.reset(env.state(),params)
        history=None;maximum=0.;locked=[];failed=torch.zeros(8,dtype=torch.bool,device=a.device);anchors=[]
        for step in range(326):
            if step==160:
                delta=torch.tensor([0.,.2,0.],device=a.device)
                env.context['aim_target_w']=env.context['aim_target_w']+delta;task.release_support(env.context)
                runtime.set_target(runtime.context['aim_target_w']+delta,runtime.context['aim_normal_w'],.2)
            state=env.state();obs=env.observe();history=obs[:,None].repeat(1,10,1) if history is None else torch.cat((history[:,1:],obs[:,None]),1)
            raw=policy(history,task.command(state,env.context));expected=task.compose_action(raw,env.xi)[1];actual=runtime.step(state)
            maximum=max(maximum,float((actual-expected).abs().max()));torch.testing.assert_close(actual,expected,atol=1e-6,rtol=0)
            torch.testing.assert_close(runtime.context['feet_anchor_w'],env.context['feet_anchor_w'],atol=0,rtol=0)
            active=env.context['support_age_steps']<0;assert torch.equal(active,runtime.context['support_age_steps']<0)
            locked.append(bool(active.all()))
            if step in (150,310):
                measured=task.kin.forward(*state)[0][:,task.feet]
                torch.testing.assert_close(env.context['feet_anchor_w'],measured,atol=0,rtol=0);anchors.append(step)
            tr=env.step(raw);failed|=tr['terminated']
    assert not locked[149] and locked[150] and locked[159] and not locked[160] and not locked[309] and locked[310]
    assert sha256(a.checkpoint)==digest
    report=dict(passed=True,scope=__doc__,policy_acceptance=False,checkpoint_sha256=digest,
        simulated_seconds=326*.02,environments=8,maximum_pd_target_difference_rad=maximum,
        capture_frames=anchors,target_switch_frame=160,root_supports_added=False,motion_reads=False,
        terminated_episodes=int(failed.sum()),observation='Only state-machine and command parity are assessed; this is not a30s policy gate.')
    a.out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
