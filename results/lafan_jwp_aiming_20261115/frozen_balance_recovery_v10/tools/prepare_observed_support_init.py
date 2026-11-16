"""Enable measured support goals while preserving the source policy's initial function."""
import argparse,json,math
from pathlib import Path
import torch
from supermimic.acp.aiming_fullbody import FullbodyPolicy,validate_checkpoint
from supermimic.acp.aiming_wide import WideAimingTask
from supermimic.acp.robots import sha256
from supermimic.acp.aiming_fullbody_io import save_checkpoint,json_save


@torch.no_grad()
def initialize_support_inputs(policy):
    """Fold the old constant-zero contact inputs into the bias, then release six columns."""
    zero=policy.goal_norm.mean.new_zeros(1,93)
    constant=policy.goal_norm(zero)[0,:6]
    layer=policy.enc[0]
    layer.bias.add_(layer.weight[:,:6]@constant)
    layer.weight[:,:6].zero_()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--capture-steps',type=int,default=150);p.add_argument('--foot-weight',type=float,default=4.)
    p.add_argument('--actor-lr',type=float,default=1e-6)
    a=p.parse_args();torch.set_num_threads(2);torch.manual_seed(2026)
    if a.out.exists():raise FileExistsError(a.out)
    if not math.isfinite(a.foot_weight) or a.foot_weight<=0 or not math.isfinite(a.actor_lr) or a.actor_lr<=0:
        raise ValueError('Positive finite foot weight and actor learning rate required')
    digest=sha256(a.source);ck=torch.load(a.source,map_location='cpu',weights_only=False)
    if ck.get('smoke') or ck.get('task')!='wide_aiming_v1' or ck['config'].get('curriculum_level')!=0:
        raise ValueError('Expected a formal balance-stage source')
    if ck['config'].get('support_capture_steps') is not None:raise ValueError('Support inputs are already enabled')
    old=WideAimingTask('cpu',ck['config']);validate_checkpoint(ck,old)
    cfg=dict(ck['config'],support_capture_steps=a.capture_steps,
        reward_weights=dict(ck['config']['reward_weights'],feet=a.foot_weight,heading=0.,foot_boundary=0.),
        initialization_mode='observed_support',distill_iters=0,wm_only_iters=100,critic_warmup_iters=100,
        actor_lr=a.actor_lr,wm_foot_velocity_weight=0.,ckpt_every=100,wm_validation_steps=384)
    task=WideAimingTask('cpu',cfg);before=FullbodyPolicy(task.robot);before.load_state_dict(ck['pi']);before.eval()
    after=FullbodyPolicy(task.robot);after.load_state_dict(ck['pi']);initialize_support_inputs(after);after.eval()
    with torch.no_grad():
        state=old.initial_state(32);state=(*state[:4],state[4]+torch.randn_like(state[4])*.02,state[5])
        params=torch.tensor([[3.,0.,0.,.2]]).expand(32,-1);old_ctx=old.context(state,params,0.);new_ctx=task.context(state,params,0.)
        new_ctx=task.observe_context(state,task.advance_context(new_ctx,a.capture_steps))
        xi=torch.zeros(32,33);xi[:,0]=1.;previous=(state[4]-task.mdp.default_q)/task.alpha
        history=task.mdp.obs(state,previous,xi)[:,None].expand(-1,10,-1).clone()
        raw0=before(history,old.command(state,old_ctx));raw1=after(history,task.command(state,new_ctx))
        error=float(((raw1-raw0)*task.alpha).abs().max())
        if error>1e-5:raise ValueError(f'Policy fold changed PD targets: {error}')
    assert sha256(a.source)==digest
    artifact=dict(task='wide_aiming_v1',stage='initialization',smoke=False,config=cfg,
        task_config=task.metadata(),robot_metadata=task.robot.metadata(),pi=after.state_dict(),wm=ck['wm'],
        provenance=ck.get('provenance'),source_checkpoint=dict(path=str(a.source.resolve()),sha256=digest),
        initialization=dict(mode='observed_support',source_accepted=False,critic_and_optimizers='fresh at training start',
            source_constant_contact_columns_folded=list(range(6)),numerical_pd_target_max_difference_rad=error,
            physical_acceptance=False))
    save_checkpoint(a.out,artifact)
    json_save(a.out.with_suffix('.json'),dict(source_sha256=digest,checkpoint_sha256=sha256(a.out),
        task_config=task.metadata(),initialization=artifact['initialization'],config=cfg))
    print(json.dumps(dict(output=str(a.out),pd_target_max_difference_rad=error,accepted=False)))


if __name__=='__main__':main()
