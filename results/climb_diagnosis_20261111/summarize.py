import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
s=json.load(open(OUT/'rollouts.json'));groups=sorted({n.split('/')[0] for n in s['policy_0.0']})
summary={'cohort':'100 fixed training-set clips, 20 per category; seed 20261111; strict termination, no noise; 100k checkpoint','rollouts':{},'wm':{},'failure_height':{},'full_evaluation_timing':{}}
for case,rs in s.items():
 summary['rollouts'][case]={}
 for g in groups:
  a=[r for n,r in rs.items() if n.startswith(g+'/')];lab=np.sum([r['label_frames'] for r in a],0);tp=np.sum([r['true_touch_frames'] for r in a],0)
  summary['rollouts'][case][g]=dict(n=len(a),completed=sum(not r['failed'] for r in a),mean_alive_seconds=float(np.mean([r['alive_steps'] for r in a])*.02),mean_saturation=float(np.mean([r['saturation'] for r in a])),mean_terminal_joint_rmse=float(np.mean([r['terminal']['joint_rmse'] for r in a])),mean_terminal_hand_distance=np.mean([r['terminal']['hand_distance'] for r in a],0).tolist(),hand_label_frames=lab[4:6].tolist(),hand_recall=(tp[4:6]/np.maximum(lab[4:6],1)).tolist(),height_failures=sum(abs(r['terminal']['height_error'])>.25 for r in a),tilt_failures=sum(r['terminal']['tilt_error']>.8 for r in a))
w=json.load(open(OUT/'wm_replay.json'))
for g in groups:
 summary['wm'][g]={};mask0=np.array([n.startswith(g+'/') for n in w['names']])
 for h in [1,8,32]:
  vals=[];bias=[]
  for r in w['rows']:
   if r['horizon']!=h:continue
   mask=mask0&np.array(r['alive']);vals.extend(np.array(r['root_error'])[mask]);bias.extend(np.array(r['height_bias'])[mask])
  summary['wm'][g][h]=dict(n=len(vals),mean_root_error=float(np.mean(vals)),p95_root_error=float(np.quantile(vals,.95)),mean_height_bias=float(np.mean(bias)))
data=ROOT/'data/tienkung3_29dof/stage1/php_all_variants_20261110/motions'
c=json.load(open(ROOT/'docs/report/20261111_stage1_stage2_analysis/evidence/reference_label_census.json'))['groups'];ev=json.load(open(ROOT/'logs/tienkung3_29dof/stage1_continue_80k_20261111/eval_100000/strict_eval.json'))['per_clip']
climbs=['motions-high-climb-76','motions-low-climb-76']
for g in climbs:
 values=[]
 for n,r in s['policy_0.0'].items():
  if not n.startswith(g+'/'):continue
  q=np.load(data/(n+'_motion.npz'))['joint_pos'];t=r['alive_steps'];ref=float(q[t,2]-q[0,2]);values.append([ref,ref+r['terminal']['height_error']])
 summary['failure_height'][g]=dict(reference_rise=float(np.mean(values,0)[0]),actual_rise=float(np.mean(values,0)[1]))
 rows=c[g]['details'];dt=[(r['first_hand_step']-ev[r['name']]['alive_steps'])*.02 for r in rows]
 summary['full_evaluation_timing'][g]=dict(before_first_hand=sum(x>0 for x in dt),n=len(rows),median_seconds_to_first_hand=float(np.median(dt)))
latest=OUT/'latest110500/rollouts.json'
if latest.exists():
 summary['latest110500']={k:{g:dict(n=sum(n.startswith(g+'/') for n in rs),completed=sum(not r['failed'] for n,r in rs.items() if n.startswith(g+'/'))) for g in groups} for k,rs in json.load(open(latest)).items()}
summary['support_screen_caveat']='Instantaneous reference LP failed frequently in all five categories, including locomotion and steps that the policy can complete. Sparse native contact manifolds, 1cm candidate threshold, friction-ray discretization and exact reference acceleration make this a limited screen. It cannot attribute climb-specific failures to impossible support, and passing cannot establish trajectory feasibility.'
summary['invalid_experiment']='invalid_reset_rollouts.json is excluded: done_eval flags were not cleared between cases. v2 cleared both done_eval and failed_eval before each case; only rollouts.json is used.'
(OUT/'summary.json').write_text(json.dumps(summary,indent=2))
fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained');colors=['#2878a7','#d06b37']
for g,label,color in zip(climbs,['High climb','Low climb'],colors):
 axs[0,0].plot([0,40,60,80],[summary['rollouts'][f'policy_{p}'][g]['completed']/20*100 for p in [0.,.4,.6,.8]],'o-',label=label,color=color)
axs[0,0].set(xlabel='Reference initialization phase (%)',ylabel='Survived to clip end (%)',title='A. Policy can execute much of the suffix');axs[0,0].legend();axs[0,0].set_ylim(-5,105)
x=np.arange(2)
axs[0,1].bar(x-.18,[summary['failure_height'][g]['reference_rise']*100 for g in climbs],.36,label='Reference',color='#8998a5')
axs[0,1].bar(x+.18,[summary['failure_height'][g]['actual_rise']*100 for g in climbs],.36,label='Policy rollout',color='#d06b37')
axs[0,1].set(xticks=x,xticklabels=['High climb','Low climb'],ylabel='Root height change from start (cm)',title='B. Failure involves actual downward motion');axs[0,1].legend();axs[0,1].axhline(0,color='black',lw=.7)
for j,g in enumerate(climbs):axs[1,0].bar(np.arange(2)+j*.35,summary['rollouts']['policy_0.0'][g]['mean_terminal_hand_distance'],.35,color=colors[j],label=['High climb','Low climb'][j])
axs[1,0].set(xticks=np.arange(2)+.175,xticklabels=['Left hand','Right hand'],ylabel='Sphere-to-terrain distance (m)',title='C. Hands remain far from contact at failure');axs[1,0].axhline(.01,color='black',ls='--',lw=1,label='Touch threshold');axs[1,0].legend()
for g,label,color in zip(climbs+['motions-locomotion'],['High climb','Low climb','Locomotion'],colors+['#71934c']):
 axs[1,1].plot([.02,.16,.64],[summary['wm'][g][h]['p95_root_error']*100 for h in [1,8,32]],'o-',label=label,color=color)
axs[1,1].set(xlabel='Open-loop prediction horizon (s)',ylabel='Root prediction error P95 (cm)',title='D. Long-horizon model error is worse in low climb');axs[1,1].legend()
for ax in axs.flat:ax.grid(axis='y',alpha=.2)
fig.suptitle('Stage1 100k: fixed 100-clip diagnostic cohort (20 per class)\nSuffix survival is not full-task success; contact distance is not support force',fontsize=12)
fig.savefig(OUT/'diagnosis.png',dpi=160);fig.savefig(OUT/'diagnosis.pdf')
print(json.dumps(summary.get('latest110500'),indent=2))
