import json
from pathlib import Path
import torch
from supermimic.acp.aiming_fullbody import FullbodyPolicy
from supermimic.acp.aiming_wide import WideAimingTask
from supermimic.acp.robots import sha256
b=Path(__file__).resolve().parent;torch.set_num_threads(2)
source=b/'medium_feedback_candidates/ckpt_000002.pt';ck=torch.load(source,map_location='cpu',weights_only=False)
task=WideAimingTask('cpu',ck['config']);arm=task.arm_ids.tolist();prox=arm[:4];body=[i for i in range(29) if i not in arm]
old=FullbodyPolicy(task.robot);old.load_state_dict(ck['pi']);old.eval()
torch.manual_seed(381);history=old.obs_norm.mean+torch.randn(16,10,99)*old.obs_norm.std();command=torch.randn(16,107)
rows=[]
for i in (1,2):
 path=b/f'medium_shared_proximal_candidates/ckpt_{i:06d}.pt';c=torch.load(path,map_location='cpu',weights_only=False)
 assert c['source_checkpoint']['sha256']==sha256(source)
 model=FullbodyPolicy(task.robot);model.load_state_dict(c['pi']);model.eval()
 masked=history.clone()
 if c['calibration']['proximal_observation_folded']:
  cols=prox+[29+j for j in prox]+[64+j for j in prox]
  masked[:,:,cols]=torch.tensor(c['calibration']['fold_values'])
 with torch.no_grad():
  expected=old(masked,command);actual=model(history,command)
 err=float(((expected[:,body]-actual[:,body])*task.alpha[body]).abs().max());assert err<1e-5
 assert torch.equal(c['pi']['dec.8.weight'][body],ck['pi']['dec.8.weight'][body])
 assert torch.equal(c['pi']['dec.8.bias'][body],ck['pi']['dec.8.bias'][body])
 assert not c['pi']['dec.8.weight'][arm].count_nonzero()
 for key in ck['pi']:
  if key.startswith(('enc.','obs_norm.','cmd_norm.')):assert torch.equal(c['pi'][key],ck['pi'][key]),key
 for key in ck['config']:
  if key not in ('initial_q','initialization_mode','distill_iters','wm_only_iters','critic_warmup_iters'):assert c['config'][key]==ck['config'][key],key
 assert torch.equal(torch.tensor(c['config']['initial_q'])[body],torch.tensor(ck['config']['initial_q'])[body])
 rows.append(dict(checkpoint_sha256=sha256(path),body_output_max_error_rad=err,body_weights_and_normalizers_unchanged=True,reset_body_and_noise_unchanged=True,proximal_observation_folded=c['calibration']['proximal_observation_folded']))
(b/'shared_proximal_candidates_verification.json').write_text(json.dumps(dict(passed=True,rows=rows,scope='Artifact identity and functional invariants, not physical acceptance'),indent=2))
print(json.dumps(rows))
