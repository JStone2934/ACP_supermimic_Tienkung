"""Paired fixed-lane physical recheck of the finite critic-isolation experiment."""
import argparse
import json
from pathlib import Path

import torch

from supermimic.acp.aiming_fullbody import FullbodyTask, FullbodyPolicy, validate_checkpoint
from supermimic.acp.aiming_fullbody_env import FullbodySim
from supermimic.acp.aiming_fullbody_io import json_save
from supermimic.acp.networks import WorldModel, Critic
from supermimic.acp.standing import freeze_normalizers
from supermimic.acp.train_aiming_fullbody import bound_bootstrap
from tools.diagnose_aiming_fullbody_update import imagined
from tools.experiment_aiming_critic_isolation import take, unpack, repeat, sampler, direction, physical_returns


def stats(x):
    return dict(mean=float(x.mean()),paired_state_se=float(x.std()/len(x)**.5),
                positive_fraction=float((x>0).float().mean()),per_state=x.tolist())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();source=Path(a.experiment);out=Path(a.out)
    if out.exists() and any(out.iterdir()):raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.manual_seed(20261114);torch.backends.cuda.matmul.allow_tf32=True
    import warp as wp
    wp.config.quiet=True
    meta=json.loads((source/'experiment.json').read_text())
    ck=torch.load(meta['source'],map_location='cpu',weights_only=False);cfg=ck['config']
    task=FullbodyTask('cuda:0',cfg);validate_checkpoint(ck,task)
    pi=FullbodyPolicy().cuda();pi.load_state_dict(ck['pi']);pi.requires_grad_(False)
    wm=WorldModel(task.mdp.scan_dim,robot=task.robot).cuda();wm.load_state_dict(ck['wm']);wm.requires_grad_(False)
    freeze_normalizers(pi,wm)
    critic=Critic(task.critic_dim).cuda();critic.requires_grad_(False)
    test=torch.load(source/'fixed_states.pt',weights_only=False)['test']
    real=torch.load(source/'true_returns.pt',weights_only=False)
    ids=((test['t']<1400)&(real['first_done'][:64]>8)).nonzero().flatten()
    probe=take(test,ids);base=real['actions'][:8,ids];n=len(ids);replicas=4
    b=repeat(probe,replicas);env=FullbodySim(task,n*replicas,sampler(task),eval_mode=True)
    def run(label,actions):
        print(f'CASE {label}: {n} states x {replicas} replicas, same physical lanes',flush=True)
        torch.manual_seed(20261114)
        return physical_returns(pi,task,cfg,b,out/f'{label}.pt',
                                first_actions=torch.cat([actions]*replicas,1),env=env,capture_tail=True)
    def avg(x):return x.reshape(replicas,n).mean(0)
    baseline=run('baseline_begin',base)
    results=[]
    for milestone in [0,100,500,2000]:
        c=torch.load(source/f'critic_{milestone:04d}.pt',weights_only=False);critic.load_state_dict(c['critic'])
        d,grad_stats=direction(pi,wm,critic,task,cfg,probe,base);critic.requires_grad_(False)
        # Intermediate milestones use the larger perturbation; endpoints use both scales.
        for epsilon in ([.005,.02] if milestone in [0,2000] else [.02]):
            physical=[];modeled=[];real_state_values=[]
            for sign in [1.,-1.]:
                actions=base+sign*epsilon*d
                label=f'{milestone}_{epsilon}_{"plus" if sign>0 else "minus"}'
                with torch.no_grad():
                    parts,_,_,_=imagined(pi,wm,critic,task,cfg,unpack(probe),actions=actions)
                    modeled.append(parts['bootstrap'])
                r=run(label,actions);physical.append(r)
                with torch.no_grad():
                    v=bound_bootstrap(critic(r['tail_features']),b['t']+8,task,cfg)
                    real_state_values.append(avg(cfg['gamma']**8*r['tail_alive']*v))
            plus,minus=physical
            total=avg(plus['total']-minus['total']);short=avg(plus['first8']-minus['first8'])
            entry=dict(extra_rounds=milestone,epsilon_rad=epsilon,gradient=grad_stats,
                       physical_total=stats(total),physical_first8=stats(short),
                       physical_discounted_tail=stats(total-short),
                       physical_standing=stats(avg(plus['standing']-minus['standing'])),
                       model_discounted_tail=stats(modeled[0]-modeled[1]),
                       critic_at_physical_tail=stats(real_state_values[0]-real_state_values[1]),
                       physical_plus_vs_baseline=stats(avg(plus['total']-baseline['total'])),
                       physical_minus_vs_baseline=stats(avg(minus['total']-baseline['total'])))
            results.append(entry)
            json_save(out/'progress.json',results)
            print('RESULT',milestone,epsilon,{k:v['mean'] for k,v in entry.items() if isinstance(v,dict) and 'mean' in v},flush=True)
    control=run('baseline_end',base)
    control_delta=avg(control['total']-baseline['total'])
    unchanged={name:all(torch.equal(v.cpu(),ck[key][k]) for k,v in model.state_dict().items())
               for name,key,model in [('actor','pi',pi),('world_model','wm',wm)]}
    assert all(unchanged.values())
    report=dict(source_experiment=str(source.resolve()),test_indices=ids.tolist(),states=n,replicas=replicas,
                same_lanes=True,no_optimizer_steps=True,unchanged=unchanged,results=results,
                baseline_control=stats(control_delta),baseline_control_mae=float(control_delta.abs().mean()),
                baseline_individual_control_mae=float((control['total']-baseline['total']).abs().mean()),
                baseline_replica_mae=float((baseline['total'].reshape(replicas,n)-avg(baseline['total'])).abs().mean()))
    json_save(out/'report.json',report)
    print('COMPLETE',json.dumps({k:v for k,v in report.items() if k not in ['results','test_indices']}),flush=True)


if __name__=='__main__':main()
