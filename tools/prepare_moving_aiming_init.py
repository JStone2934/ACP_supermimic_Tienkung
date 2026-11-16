"""Create a separate moving branch only after two full standing acceptances."""
import argparse
import json
from pathlib import Path
import torch
from supermimic.acp.robots import sha256
from supermimic.acp.chest_posture import verify_posture_reports
from supermimic.acp.aiming_wide import WideAimingTask
from supermimic.acp.aiming_moving import MovingAimingTask
from supermimic.acp.aiming_fullbody import validate_checkpoint
from supermimic.acp.aiming_fullbody_io import save_checkpoint,json_save


def verify_standing_acceptance(checkpoint,reports):
    digest=sha256(checkpoint)
    if len(reports)!=2 or len({r.get('seed') for r in reports})!=2:
        raise ValueError('Two independently seeded full standing reports required')
    for r in reports:
        if not (r.get('checkpoint_sha256')==digest and r.get('scope')=='Full standing aiming acceptance'
                and r.get('level') is None and r.get('passed') is True and r.get('fixed_grid_all_pass') is True
                and r.get('fixed_count')==75 and r.get('random_count')==100 and r.get('random_success_rate',0)>=.95
                and r.get('seconds')==30 and r.get('continuous_without_reset') is True
                and r.get('reference_free') is True and r.get('online_target_switches') is True
                and r.get('hit_radius_m')==.05):
            raise ValueError('Report does not prove full standing acceptance for this checkpoint')
    return digest


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',required=True,type=Path)
    p.add_argument('--reports',nargs=2,required=True,type=Path);p.add_argument('--out',required=True,type=Path)
    a=p.parse_args();torch.set_num_threads(2)
    if a.out.exists():raise FileExistsError(a.out)
    reports=[json.loads(x.read_text()) for x in a.reports]
    verify_posture_reports(reports)
    digest=verify_standing_acceptance(a.checkpoint,reports)
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    validate_checkpoint(ck,WideAimingTask('cpu',ck['config']))
    if ck.get('smoke') or ck.get('task')!='wide_aiming_v1':raise ValueError('Expected formal standing checkpoint')
    cfg=dict(ck['config']);source=dict(path=str(a.checkpoint.resolve()),sha256=digest)
    cfg.update(task='moving_aiming_v1',curriculum_level=0,distill_iters=0,wm_only_iters=100,
               critic_warmup_iters=200,switch_interval_steps=250,source=source,
               policy_training_scope='full')
    # Locomotion must be able to adapt the legs; standing-only adapter optimizer
    # groups do not apply to the new full-body branch. Policy weights stay exact.
    cfg.pop('support_actor_lr',None)
    task=MovingAimingTask('cpu',cfg)
    result=dict(task='moving_aiming_v1',stage='initialization',iteration=0,robot_metadata=ck['robot_metadata'],
        config=cfg,task_config=task.metadata(),pi=ck['pi'],wm=ck['wm'],source_checkpoint=source,
        provenance=dict(standing_checkpoint=source,standing_reports=[dict(path=str(x.resolve()),sha256=sha256(x)) for x in a.reports]))
    if sha256(a.checkpoint)!=digest:raise ValueError('Source checkpoint changed')
    save_checkpoint(a.out,result);json_save(a.out.with_suffix('.json'),dict(config=cfg,provenance=result['provenance'],
        scope='Separate moving initialization only; standing checkpoint unchanged, moving acceptance still required'))


if __name__=='__main__':main()
