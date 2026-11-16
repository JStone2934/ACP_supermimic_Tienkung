"""Transfer unchanged policy weights by one curriculum level after both physical gates."""
import argparse
import json
from pathlib import Path

import torch

from supermimic.acp.aiming_fullbody import validate_checkpoint
from supermimic.acp.aiming_fullbody_io import json_save, save_checkpoint
from supermimic.acp.aiming_wide import WideAimingTask
from supermimic.acp.robots import sha256
from supermimic.acp.train_wide_aiming import check_promotion
from supermimic.acp.wide_curriculum import STAGES


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--reports',type=Path,nargs=2,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(2)
    if a.out.exists():raise FileExistsError(a.out)
    ck=torch.load(a.source,map_location='cpu',weights_only=False)
    if ck.get('smoke') or ck.get('task')!='wide_aiming_v1' or ck['stage'] not in ('training','calibration'):
        raise ValueError('Requires a formal standing aiming artifact')
    previous=ck['config']['curriculum_level'];level=previous+1
    if previous not in range(1,len(STAGES)-1):
        raise ValueError('Exact-weight transfer requires a completed aiming stage before the final switching stage')
    reports=[json.loads(path.read_text()) for path in a.reports]
    digest=check_promotion(a.source,reports,previous)
    validate_checkpoint(ck,WideAimingTask('cpu',ck['config']))
    cfg=dict(ck['config'],curriculum_level=level,initialization_mode='gated_exact_weight_curriculum_transfer',
        switch_interval_steps=500 if level==len(STAGES)-1 else None,
        distill_iters=0,wm_only_iters=100,critic_warmup_iters=100)
    task=WideAimingTask('cpu',cfg)
    promotion=dict(previous_level=previous,next_level=level,source_checkpoint_sha256=digest,
        reports=[dict(path=str(path.resolve()),sha256=sha256(path),seed=report['seed']) for path,report in zip(a.reports,reports)],
        all_policy_and_model_weights_exact=True,optimizer_updates=0,previous_stage_passed=True,
        next_stage_accepted=False,final_policy_accepted=False,
        scope='Evaluate the exact transferred policy before deciding whether more training is needed. Training from this artifact uses fresh critic and100model warmup updates.')
    artifact=dict(task=ck['task'],stage='calibration',smoke=False,config=cfg,task_config=task.metadata(),
        robot_metadata=ck['robot_metadata'],pi=ck['pi'],wm=ck['wm'],provenance=ck.get('provenance'),
        source_checkpoint=dict(path=str(a.source.resolve()),sha256=digest),
        promotion=promotion,promotion_reports=[str(path.resolve()) for path in a.reports])
    validate_checkpoint(artifact,task)
    assert sha256(a.source)==digest
    save_checkpoint(a.out,artifact)
    json_save(a.out.with_suffix('.json'),dict(checkpoint_sha256=sha256(a.out),promotion=promotion))
    print(json.dumps(promotion),flush=True)


if __name__=='__main__':main()
