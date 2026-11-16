"""Initialize error-feedback input training from an unchanged formal balance policy."""
import argparse,math
from pathlib import Path
import torch
from supermimic.acp.aiming_fullbody import FullbodyPolicy,validate_checkpoint
from supermimic.acp.aiming_wide import WideAimingTask
from supermimic.acp.aiming_fullbody_io import save_checkpoint,json_save
from supermimic.acp.robots import sha256


@torch.no_grad()
def initialize_error_normalizer(policy,scale=.05):
    if torch.count_nonzero(policy.enc[0].weight[:,:6]):
        raise ValueError('Source support-input columns must still be zero')
    if not math.isfinite(scale) or scale<=policy.goal_norm.eps:
        raise ValueError('Input scale must exceed the normalizer epsilon')
    policy.goal_norm.mean[:6].zero_()
    policy.goal_norm.var[:6].fill_(scale**2-policy.goal_norm.eps**2)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--actor-lr',type=float,default=1e-4)
    a=p.parse_args();torch.set_num_threads(2);torch.manual_seed(2026)
    if a.out.exists():raise FileExistsError(a.out)
    if not math.isfinite(a.actor_lr) or a.actor_lr<=0:raise ValueError('Positive actor LR required')
    digest=sha256(a.source);ck=torch.load(a.source,map_location='cpu',weights_only=False)
    if (ck.get('smoke') or ck.get('task')!='wide_aiming_v1' or ck['config'].get('curriculum_level')!=0
            or ck.get('distributed',{}).get('updates',{}).get('pi',0)):
        raise ValueError('Expected formal pre-actor observed-support checkpoint')
    old=WideAimingTask('cpu',ck['config']);validate_checkpoint(ck,old)
    if old.support_capture_steps is None or old.support_command_mode!='position':
        raise ValueError('Expected measured-position source with zero input columns')
    cfg=dict(ck['config'],support_command_mode='error',policy_training_scope='support_error_inputs',
             initialization_mode='support_error_adapter',distill_iters=0,wm_only_iters=100,critic_warmup_iters=100,
             actor_lr=a.actor_lr,ckpt_every=50)
    task=WideAimingTask('cpu',cfg);before=FullbodyPolicy(task.robot);before.load_state_dict(ck['pi']);before.eval()
    after=FullbodyPolicy(task.robot);after.load_state_dict(ck['pi']);initialize_error_normalizer(after);after.eval()
    with torch.no_grad():
        S=task.initial_state(32);params=torch.tensor([[3.,0.,0.,.2]]).expand(32,-1)
        ctx=old.observe_context(S,old.advance_context(old.context(S,params,0.),150))
        S=(S[0]+torch.randn_like(S[0])*.01,*S[1:])
        xi=torch.zeros(32,task.robot.xi_dim);xi[:,0]=1.
        prev=(S[4]-task.mdp.default_q)/task.alpha
        hist=task.mdp.obs(S,prev,xi)[:,None].expand(-1,10,-1).clone()
        error=float(((before(hist,old.command(S,ctx))-after(hist,task.command(S,ctx)))*task.alpha).abs().max())
        if error>1e-6:raise ValueError(f'Initialization changed PD targets: {error}')
    assert sha256(a.source)==digest
    metadata=dict(source_accepted=False,source_checkpoint_sha256=digest,
        policy_function_max_difference_rad=error,normalizer_change='Only first6 channels: zero mean,0.05m standard deviation',
        trainable_policy_elements=int(after.enc[0].out_features*6),
        frozen_policy='All existing parameters; only first6 columns of encoder input weights train',
        no_feedback_when='Before support capture, during movement, or exactly zero support error',
        physical_acceptance=False)
    artifact=dict(task='wide_aiming_v1',stage='initialization',smoke=False,config=cfg,task_config=task.metadata(),
        robot_metadata=task.robot.metadata(),pi=after.state_dict(),wm=ck['wm'],provenance=ck.get('provenance'),
        source_checkpoint=dict(path=str(a.source.resolve()),sha256=digest),initialization=metadata)
    save_checkpoint(a.out,artifact)
    json_save(a.out.with_suffix('.json'),dict(checkpoint_sha256=sha256(a.out),config=cfg,initialization=metadata))
    print(metadata)


if __name__=='__main__':main()
