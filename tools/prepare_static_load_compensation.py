"""Offline leg-output bias compensation for a measured change in static contact load."""
import argparse,json
from pathlib import Path
import torch,numpy as np
from supermimic.acp.aiming_fullbody import FullbodyPolicy,validate_checkpoint
from supermimic.acp.aiming_wide import WideAimingTask
from supermimic.acp.aiming_fullbody_io import save_checkpoint,json_save
from supermimic.acp.robots import sha256
p=argparse.ArgumentParser(description=__doc__)
for k in ('source','load-diagnostic','out'):p.add_argument('--'+k,type=Path,required=True)
p.add_argument('--scales',type=float,nargs='+',default=[.5,1.]);p.add_argument('--translate-action-history',action='store_true');a=p.parse_args();torch.set_num_threads(2);assert all(0<x<=1 for x in a.scales)
ck=torch.load(a.source,map_location='cpu',weights_only=False);r=json.loads(a.load_diagnostic.read_text());assert ck['source_checkpoint']['sha256']==r['candidate_sha256'];T=WideAimingTask('cpu',ck['config']);validate_checkpoint(ck,T);assert list(T.robot.joint_names)==r['joint_names']
legs=list(range(12));assert all(any(s in T.robot.joint_names[i] for s in ('hip','knee','ankle')) for i in legs)
dq=torch.tensor(r['delta_pd_target_rad']);assert dq.isfinite().all() and float(dq[legs].abs().max())<.06
for key in ('old_contact_wrench','new_contact_wrench'):
 w=np.array(r[key]).reshape(2,6);assert (w[:,2]>0).all();assert (np.linalg.norm(w[:,:2],axis=1)<.5*w[:,2]).all();assert (np.abs(w[:,3:5])/w[:,2,None]<.08).all()
old=FullbodyPolicy(T.robot);old.load_state_dict(ck['pi']);torch.manual_seed(215);h=torch.randn(16,10,99);c=torch.randn(16,107);a.out.mkdir(parents=True,exist_ok=True)
for n,scale in enumerate(a.scales,1):
 out=a.out/f'ckpt_{n:06d}.pt';assert not out.exists();pi={k:v.clone() for k,v in ck['pi'].items()};pi['dec.8.bias'][legs]+=scale*dq[legs]/T.alpha[legs];masked=h.clone()
 if a.translate_action_history:
  delta=scale*dq[legs]/T.alpha[legs];columns=[64+i for i in legs];flat=[32+t*99+column for t in range(10) for column in columns]
  pi['dec.0.bias']-=pi['dec.0.weight'][:,flat]@(delta/old.obs_norm.std()[columns]).repeat(10)
  masked[:,:,columns]-=delta
 new=FullbodyPolicy(T.robot);new.load_state_dict(pi)
 with torch.no_grad():
  difference=(new(h,c)-old(masked,c))*T.alpha;expected=torch.zeros_like(difference);expected[:,legs]=scale*dq[legs];torch.testing.assert_close(difference,expected,atol=1e-6,rtol=1e-5)
 for key in pi:
  if key not in (('dec.8.bias','dec.0.bias') if a.translate_action_history else ('dec.8.bias',)):assert torch.equal(pi[key],ck['pi'][key])
 cfg=dict(ck['config'],initialization_mode='static_contact_load_bias_calibration')
 cal=dict(method=__doc__,scale=scale,previous_action_history_translated=a.translate_action_history,explicit_translated_history_equivalence=True,load_diagnostic_sha256=sha256(a.load_diagnostic),delta_leg_pd_rad=(scale*dq[legs]).tolist(),arm_outputs_exact=True,normalizers_and_reset_exact=True,extra_runtime_controller=False,physical_acceptance=False,limitation='Static minimum-norm load allocation is an approximation; full physical evaluation required')
 artifact=dict(task=ck['task'],stage='calibration',smoke=False,config=cfg,task_config=WideAimingTask('cpu',cfg).metadata(),robot_metadata=ck['robot_metadata'],pi=pi,wm=ck['wm'],provenance=ck.get('provenance'),source_checkpoint=dict(path=str(a.source.resolve()),sha256=sha256(a.source)),calibration=cal)
 validate_checkpoint(artifact,WideAimingTask('cpu',cfg));save_checkpoint(out,artifact);json_save(out.with_suffix('.json'),dict(checkpoint_sha256=sha256(out),calibration=cal));print(str(out),flush=True)
