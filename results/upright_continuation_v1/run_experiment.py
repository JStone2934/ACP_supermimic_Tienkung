"""Single bounded warmup/update/evaluation experiment; never auto-promotes."""
import json,subprocess,sys,os,time
from pathlib import Path
ROOT=Path('/home/js/robot/locomotion/standing-aiming');OUT=ROOT/'results/upright_continuation_v1';RUN=OUT/'runner';PY='/home/js/miniconda3/envs/acp_supermimic/bin/python'
def save(name,value):(OUT/name).write_text(json.dumps(value,indent=2)+'\n')
def call(command,log):
 with (OUT/log).open('w') as f:subprocess.run(command,cwd=RUN,stdout=f,stderr=subprocess.STDOUT,check=True)
def main():
 save('experiment_status.json',dict(phase='model_warmup_then_max5_actor_updates',started=time.time(),pid=os.getpid()))
 call([PY,'-m','torch.distributed.run','--nproc_per_node=2','--master_port=29786','-m','supermimic.acp.train_wide_aiming','--init',str(OUT/'init.pt'),'--log-dir',str(OUT/'train'),'--iterations','105','--checkpoint-every','5','--compile-training'],'train.log')
 save('experiment_status.json',dict(phase='physical_evaluation',checkpoint='train/ckpt_000105.pt'))
 processes=[]
 for device,seed in enumerate([17,53]):
  dest=OUT/'evaluation';dest.mkdir(exist_ok=True)
  log=(dest/f'seed{seed}.log').open('w')
  cmd=[PY,'-m','tools.evaluate_wide_stage','--checkpoint',str(OUT/'train/ckpt_000105.pt'),'--out',str(dest/f'seed{seed}.json'),'--level','4','--seed',str(seed),'--device',f'cuda:{device}']
  processes.append((subprocess.Popen(cmd,cwd=RUN,stdout=log,stderr=subprocess.STDOUT),log))
 for p,f in processes:
  code=p.wait();f.close()
  if code:raise RuntimeError('Evaluation failed')
 sys.path.insert(0,str(RUN))
 import numpy as np,torch
 from supermimic.acp.aiming_wide import WideAimingTask
 from supermimic.rotations import quat_to_mat
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
  bd=OUT/f'baseline/seed{seed}.json';cd=OUT/f'evaluation/seed{seed}.json';b=json.loads(bd.read_text());c=json.loads(cd.read_text())
  bz=np.load(bd.with_suffix('.npz'));cz=np.load(cd.with_suffix('.npz'))
  def old_pass(s):return s['standing_pass'] and all(x['passed'] for x in s['segments'])
  before=np.array([old_pass(s) for s in b['scenarios']]);after=np.array([old_pass(s) for s in c['scenarios']])
  berr=np.nan_to_num(bz['error'][150:],nan=np.inf);cerr=np.nan_to_num(cz['error'][150:],nan=np.inf)
  bc,cc=chest(bz),chest(cz)
  row=dict(seed=seed,baseline_passes=int(before.sum()),candidate_passes=int(after.sum()),lost_scenarios=np.where(before&~after)[0].tolist(),baseline_chest_median_deg=bc,candidate_chest_median_deg=cc,
           baseline_error_median=float(np.median(berr)),candidate_error_median=float(np.median(cerr)),baseline_error_p95=float(np.quantile(berr,.95)),candidate_error_p95=float(np.quantile(cerr,.95)),strict_posture_acceptance=c['passed'])
  row['retention_pass']=not row['lost_scenarios'] and after.sum()>=before.sum() and row['candidate_error_median']<=row['baseline_error_median']+1e-4 and row['candidate_error_p95']<=row['baseline_error_p95']+1e-4
  row['posture_improved']=cc<=bc-.5
  rows.append(row)
 status=json.loads((OUT/'train/status.json').read_text())
 save('comparison.json',dict(rows=rows,training=status,eligible_for_next_review=all(x['retention_pass'] and x['posture_improved'] for x in rows),final_acceptance=False,scope='Paired stage4 diagnostic only; chest medians subsampled every0.2s; unchanged final chest criteria still apply; no automatic stage advance'))
 save('experiment_status.json',dict(phase='completed_pending_review',eligible_for_next_review=all(x['retention_pass'] and x['posture_improved'] for x in rows),final_acceptance=False))
if __name__=='__main__':
 try:main()
 except Exception as e:
  save('experiment_status.json',dict(phase='failed',error=repr(e)));raise
