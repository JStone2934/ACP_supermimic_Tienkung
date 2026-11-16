"""Copy exact weights for a labeled, unaccepted target-range diagnostic."""
import argparse
from pathlib import Path
import torch
from supermimic.acp.aiming_fullbody import validate_checkpoint
from supermimic.acp.aiming_fullbody_io import save_checkpoint,json_save
from supermimic.acp.aiming_wide import WideAimingTask
from supermimic.acp.robots import sha256
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--level',type=int,choices=(3,4,5,6),required=True);a=p.parse_args();torch.set_num_threads(2)
assert not a.out.exists();s=torch.load(a.source,map_location='cpu',weights_only=False);assert not s.get('smoke');assert a.level>s['config']['curriculum_level'];validate_checkpoint(s,WideAimingTask('cpu',s['config']))
cfg=dict(s['config'],curriculum_level=a.level,switch_interval_steps=500 if a.level==6 else None,initialization_mode='unaccepted_range_diagnostic')
c=dict(s,stage='diagnostic',config=cfg,task_config=WideAimingTask('cpu',cfg).metadata(),source_checkpoint=dict(path=str(a.source.resolve()),sha256=sha256(a.source)),range_diagnostic=dict(diagnostic_only=True,source_stage_passed=False,automatic_promotion=False,physical_acceptance=False,all_weights_and_initial_pose_exact=True))
validate_checkpoint(c,WideAimingTask('cpu',cfg));save_checkpoint(a.out,c);json_save(a.out.with_suffix('.json'),dict(checkpoint_sha256=sha256(a.out),source_checkpoint=c['source_checkpoint'],range_diagnostic=c['range_diagnostic']));print(str(a.out))
