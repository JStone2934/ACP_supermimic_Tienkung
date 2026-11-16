from pathlib import Path
import json,torch
from supermimic.acp.aiming_fullbody_io import save_checkpoint,json_save
from supermimic.acp.aiming_wide import WideAimingTask
from supermimic.acp.aiming_fullbody import validate_checkpoint
from supermimic.acp.robots import sha256
b=Path(__file__).resolve().parent;torch.set_num_threads(2);src=b/'medium_grid_fit_candidates/ckpt_000002.pt';ck=torch.load(src,map_location='cpu',weights_only=False);t=WideAimingTask('cpu',ck['config']);out=b/'medium_grid_feedback_candidates';out.mkdir(exist_ok=True)
# Existing current-direction coefficient is exactly -K * inverse(J) / action_scale.
W=-ck['pi']['distal_command_skip.weight'][:,3:6]/ck['calibration']['feedback_gain']
for n,gain in enumerate((7.,9.),1):
 p=out/f'ckpt_{n:06d}.pt';assert not p.exists();pi={k:v.clone() for k,v in ck['pi'].items()};delta=gain-ck['calibration']['feedback_gain'];pi['distal_command_skip.weight'][:,:3]+=delta*W;pi['distal_command_skip.weight'][:,3:6]-=delta*W
 # The feedforward sum is unchanged whenever observed and requested directions agree.
 torch.testing.assert_close(pi['distal_command_skip.weight'][:,:3]+pi['distal_command_skip.weight'][:,3:6],ck['pi']['distal_command_skip.weight'][:,:3]+ck['pi']['distal_command_skip.weight'][:,3:6],atol=2e-5,rtol=2e-5)
 for k in pi:
  if k!='distal_command_skip.weight':assert torch.equal(pi[k],ck['pi'][k])
 c=dict(ck,pi=pi,source_checkpoint=dict(path=str(src.resolve()),sha256=sha256(src)),calibration=dict(ck['calibration'],feedback_gain=gain,method='Bounded stronger feedback sweep; existing full-grid affine feedforward unchanged',physical_acceptance=False))
 validate_checkpoint(c,t);save_checkpoint(p,c);json_save(p.with_suffix('.json'),dict(checkpoint_sha256=sha256(p),calibration=c['calibration']));print(str(p),flush=True)
