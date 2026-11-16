"""Change only direction-feedback gain around an unchanged learned distal mapping."""
import argparse
from pathlib import Path
import torch
from supermimic.acp.aiming_wide import WideAimingTask
from supermimic.acp.aiming_fullbody import validate_checkpoint
from supermimic.acp.aiming_fullbody_io import save_checkpoint,json_save
from supermimic.acp.robots import sha256
p=argparse.ArgumentParser(description=__doc__)
for k in ('source','feedback-reference','out'):p.add_argument('--'+k,type=Path,required=True)
p.add_argument('--gains',nargs='+',type=float,default=[4.,7.]);a=p.parse_args();torch.set_num_threads(2);assert all(0<g<=9 for g in a.gains)
ck=torch.load(a.source,map_location='cpu',weights_only=False);ref=torch.load(a.feedback_reference,map_location='cpu',weights_only=False);assert ck['config']['distal_kinematic_head']
for key in ref['pi']:
 if key.startswith(('distal_command_skip.','distal_kinematic_head.')):assert torch.equal(ref['pi'][key],ck['pi'][key]),key
oldgain=ref['calibration']['feedback_gain'];W=ck['pi']['distal_command_skip.weight'];assert torch.allclose(W[:,:3],-W[:,3:6]);assert not W[:,6:].count_nonzero();a.out.mkdir(parents=True,exist_ok=True)
for n,gain in enumerate(a.gains,1):
 out=a.out/f'ckpt_{n:06d}.pt';assert not out.exists();pi={k:v.clone() for k,v in ck['pi'].items()};pi['distal_command_skip.weight']*=gain/oldgain
 for k in pi:
  if k!='distal_command_skip.weight':assert torch.equal(pi[k],ck['pi'][k])
 cal=dict(method=__doc__,feedback_gain=gain,reference_sha256=sha256(a.feedback_reference),all_other_weights_and_config_exact=True,physical_acceptance=False)
 c=dict(ck,pi=pi,source_checkpoint=dict(path=str(a.source.resolve()),sha256=sha256(a.source)),calibration=cal);validate_checkpoint(c,WideAimingTask('cpu',c['config']));save_checkpoint(out,c);json_save(out.with_suffix('.json'),dict(checkpoint_sha256=sha256(out),calibration=cal));print(str(out),flush=True)
