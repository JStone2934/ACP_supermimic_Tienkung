"""Single bounded warmup/update/evaluation experiment; never auto-promotes."""
import json,subprocess,sys,os,time
from pathlib import Path
ROOT=Path('/home/js/robot/locomotion/standing-aiming');OUT=ROOT/'results/upright_continuation_v2';BASE=ROOT/'results/upright_continuation_v1';RUN=OUT/'runner';PY='/home/js/miniconda3/envs/acp_supermimic/bin/python'
def save(name,value):(OUT/name).write_text(json.dumps(value,indent=2)+'\n')
def call(command,log):
 with (OUT/log).open('w') as f:subprocess.run(command,cwd=RUN,stdout=f,stderr=subprocess.STDOUT,check=True)
def main():
 cfg=json.loads((OUT/'init.json').read_text())['config']
 for seed in [17,53]:
  baseline=json.loads((BASE/f'baseline/seed{seed}.json').read_text())
  if baseline['checkpoint_sha256']!=json.loads((OUT/'init.json').read_text())['source_checkpoint']['sha256']:raise ValueError('Baseline identity mismatch')
 save('experiment_status.json',dict(phase='model_warmup_then_max1_actor_update',started=time.time(),pid=os.getpid()))
 call([PY,'-m','torch.distributed.run','--nproc_per_node=2','--master_port=29786','-m','supermimic.acp.train_wide_aiming','--init',str(OUT/'init.pt'),'--log-dir',str(OUT/'train'),'--iterations','101','--checkpoint-every','100','--compile-training'],'train.log')
 save('experiment_status.json',dict(phase='post_update_model_validation',checkpoint='train/ckpt_000101.pt'))
 call([PY,str(OUT/'post_update_gate.py')],'post_update_gate.log')
 save('experiment_status.json',dict(phase='physical_evaluation',checkpoint='train/ckpt_000101.pt'))
 processes=[]
 for device,seed in enumerate([17,53]):
  dest=OUT/'evaluation';dest.mkdir(exist_ok=True)
  log=(dest/f'seed{seed}.log').open('w')
  cmd=[PY,'-m','tools.evaluate_wide_stage','--checkpoint',str(OUT/'train/ckpt_000101.pt'),'--out',str(dest/f'seed{seed}.json'),'--level','4','--seed',str(seed),'--device',f'cuda:{device}']
  processes.append((subprocess.Popen(cmd,cwd=RUN,stdout=log,stderr=subprocess.STDOUT),log))
 for p,f in processes:
  code=p.wait();f.close()
  if code:raise RuntimeError('Evaluation failed')
 sys.path.insert(0,str(RUN))
 import numpy as np,torch
 from supermimic.acp.aiming_wide import WideAimingTask
 from supermimic.rotations import quat_to_mat
 from supermimic.acp.aiming_statistics import error_summary
 torch.set_num_threads(2)
 cfg=json.loads((OUT/'init.json').read_text())['config'];task=WideAimingTask('cpu',cfg)
 def chest(z):
  if 'chest_tilt_deg' in z:return float(np.median(z['chest_tilt_deg'][150::10]))
  q=torch.tensor(z['q'][150::10].reshape(-1,29));r=torch.tensor(z['root_quat'][150::10].reshape(-1,4));vals=[]
  for start in range(0,len(q),512):
   qq=q[start:start+512];R=quat_to_mat(r[start:start+512]);p=torch.zeros(len(qq),3);v=p.clone()
   br=task.kin.forward(p,R,v,v,qq,torch.zeros_like(qq))[1][:,task.torso]
   vals.append(torch.rad2deg(torch.acos(br[:,2,2].clamp(-1,1))).numpy())
  return float(np.median(np.concatenate(vals)))
 rows=[]
 for seed in [17,53]:
  bd=BASE/f'baseline/seed{seed}.json';cd=OUT/f'evaluation/seed{seed}.json';b=json.loads(bd.read_text());c=json.loads(cd.read_text())
  bz=np.load(bd.with_suffix('.npz'));cz=np.load(cd.with_suffix('.npz'))
  def old_pass(s):return s['standing_pass'] and all(x['passed'] for x in s['segments'])
  before=np.array([old_pass(s) for s in b['scenarios']]);after=np.array([old_pass(s) for s in c['scenarios']])
  bs=error_summary(bz['error'][150:]);cs=error_summary(cz['error'][150:])
  bc,cc=chest(bz),chest(cz)
  row=dict(seed=seed,baseline_passes=int(before.sum()),candidate_passes=int(after.sum()),lost_scenarios=np.where(before&~after)[0].tolist(),baseline_chest_median_deg=bc,candidate_chest_median_deg=cc,
           baseline_error_median=bs['median_m'],candidate_error_median=cs['median_m'],baseline_error_p95=bs['p95_m'],candidate_error_p95=cs['p95_m'],original_task_acceptance=c['passed'])
  row['baseline_error_summary']=bs;row['candidate_error_summary']=cs
  row['retention_pass']=all(row[k] is not None for k in ['candidate_error_median','baseline_error_median','candidate_error_p95','baseline_error_p95']) and not row['lost_scenarios'] and after.sum()>=before.sum() and row['candidate_error_median']<=row['baseline_error_median']+1e-4 and row['candidate_error_p95']<=row['baseline_error_p95']+1e-4
  joint_ids=task.posture_reference_ids.tolist();reference=np.array(cfg['posture_reference']['q'])[joint_ids]
  measure=lambda z:float(np.median(np.sqrt(np.mean((z['q'][150:,:,joint_ids]-reference)**2,axis=-1)))*180/np.pi)
  row['baseline_reference22_rmse_median_deg']=measure(bz);row['candidate_reference22_rmse_median_deg']=measure(cz)
  row['posture_improved']=row['candidate_reference22_rmse_median_deg']<=row['baseline_reference22_rmse_median_deg']-.1
  rows.append(row)
 status=json.loads((OUT/'train/status.json').read_text())
 save('comparison.json',dict(rows=rows,training=status,eligible_for_next_review=all(x['retention_pass'] and x['posture_improved'] for x in rows) and status['updates'].get('pi',0)>0 and json.loads((OUT/'post_update_gate.json').read_text())['passed'],final_acceptance=False,scope='Paired stage4 diagnostic only; 22-joint reference RMS excludes right arm; chest medians diagnostic only sampled0.2s; no final posture or switching certification; no automatic stage advance'))
 save('experiment_status.json',dict(phase='completed_pending_review',eligible_for_next_review=all(x['retention_pass'] and x['posture_improved'] for x in rows) and status['updates'].get('pi',0)>0 and json.loads((OUT/'post_update_gate.json').read_text())['passed'],final_acceptance=False))
if __name__=='__main__':
 try:main()
 except Exception as e:
  save('experiment_status.json',dict(phase='failed',error=repr(e)));raise
