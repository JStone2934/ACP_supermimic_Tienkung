"""Summarize saved diagnostics and render static figures without GPU work."""
from pathlib import Path
import json,hashlib,datetime,statistics
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path('/home/js/robot/locomotion/standing-aiming')
OUT=Path(__file__).resolve().parent
DOC=ROOT/'docs/report/20261115_h16_standing_failure'
FIG=DOC/'figures';FIG.mkdir(parents=True,exist_ok=True)
read=lambda n:json.loads((OUT/n).read_text())
rows=read('timeline.json');stop=read('stop_record.json');phy=read('physical_audit.json')
updates=read('policy_update/report.json');directions=read('h16_directions.json');sampling=read('sampling_audit.json')
cs=[r for r in rows if r['phase'] in ['candidate_accepted','candidate_rejected']]
cal=[r for r in rows if r['phase']=='critic_calibration']
accepted=[r for r in cs if r['phase']=='candidate_accepted']
recoveries=[r for r in rows if r['phase']=='model_recovery']
last_accept=accepted[-1]
last_accept_index=cs.index(last_accept)
g=sampling['by_time'];N=sum(x['train_count'] for x in g)
summary=dict(stopped=stop,h16_rounds=stop['iteration']-4795,h16_wm_updates=(stop['iteration']-4795)*10,
  attempted_blocks=len(cs),accepted_blocks=len(accepted),rejected_blocks=len(cs)-len(accepted),
  attempted_actor_updates=len(cs)*5,retained_actor_updates=len(accepted)*5,
  last_accept_iteration=last_accept['iteration'],trailing_rejected_blocks=len(cs)-last_accept_index-1,
  hours_without_accept=(cs[-1]['elapsed_s']-last_accept['elapsed_s'])/3600,
  recovery_count=len(recoveries),full_recovery_rounds=len(recoveries)*100,
  partial_final_recovery_rounds=stop['iteration']-cs[-1]['iteration'],
  actor_lr_values=sorted(set(r['actor_lr'] for r in cs)),
  all_candidate_mean_survival_under_two_seconds=all(r['candidate']['survival_s']<2 for r in cs),
  last_wm_validation_iteration=[r for r in rows if r['phase']=='wm_validation'][-1]['iteration'],
  unchanged_critic_calibrations=sum(r['calibration']['best_mae']==r['calibration']['before_mae'] for r in cal),
  unchanged_last500=sum(r['calibration']['best_mae']==r['calibration']['before_mae'] for r in cal[-500:]),
  actor_start_fraction_before_step16=sum(x['train_count'] for x in g if x['step']<16)/N,
  actor_start_fraction_after_step32=sum(x['train_count'] for x in g if x['step']>=32)/N,
  actor_start_mean_step=sum(x['step']*x['train_count'] for x in g)/N,
  gradient_tail_to_reward=updates['gradients']['objectives']['bootstrap']['gradient_norm']/updates['gradients']['objectives']['standing']['gradient_norm'])
summary['recovery_fraction']=(summary['full_recovery_rounds']+summary['partial_final_recovery_rounds'])/summary['h16_rounds']
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
x=np.array([r['elapsed_s']/3600 for r in cs]);tlast=last_accept['elapsed_s']/3600
fig,axes=plt.subplots(3,1,figsize=(10,8),sharex=True,layout='constrained')
axes[0].plot(x,[r['baseline']['survival_s'] for r in cs],lw=.8,color='#176b87')
axes[0].set_ylabel('Survival (s)');axes[0].set_ylim(1.25,1.45)
axes[0].set_title('H16 + 4096 critic environments: policy stalls despite continued model training')
axes[1].plot(x,[r['accepted_blocks']-35 for r in cs],label='Accepted blocks',color='#208860')
axes[1].plot(x,[r['rejected_blocks']-103 for r in cs],label='Rejected blocks',color='#bd453a')
axes[1].set_ylabel('New policy blocks');axes[1].legend(loc='upper left')
axes[2].plot([r['elapsed_s']/3600 for r in cal],[r['calibration']['best_mae'] for r in cal],lw=.8,color='#8055a5')
axes[2].set_ylabel('Critic validation MAE');axes[2].set_xlabel('Hours since H16 training began')
for ax in axes:
 ax.axvspan(tlast,x[-1],color='#bd453a',alpha=.07);ax.axvline(tlast,color='#bd453a',ls='--',lw=1);ax.grid(alpha=.18)
axes[0].text(tlast+.2,1.415,'No accepted update for 9.35 hours',color='#a62c25')
fig.savefig(FIG/'training_timeline.png',dpi=170);fig.savefig(FIG/'training_timeline.svg');plt.close(fig)

z=np.load(OUT/'fall_trace.npz');i=0;rot=z['rotation'][:,i]
pitch=np.degrees(np.arctan2(-rot[:,2,0],np.sqrt(rot[:,2,1]**2+rot[:,2,2]**2)))
initial=np.array(json.loads((ROOT/'data/tienkung3_29dof/manifests/robot.json').read_text())['default_q'])
arm=['shoulder_pitch_r_joint','shoulder_roll_r_joint','shoulder_yaw_r_joint','elbow_pitch_r_joint','elbow_yaw_r_joint','wrist_pitch_r_joint','wrist_roll_r_joint']
for n,v in zip(arm,[.2,-.1,0.,-np.pi/2,0.,0.,0.]):initial[phy['joint_names'].index(n)]=v
change=np.degrees(abs(z['q_des'][:,i]-initial).max(1))
fig,ax=plt.subplots(1,2,figsize=(10,3.6),layout='constrained')
ax[0].plot(z['time'],z['root'][:,i,0]*100,label='Forward root drift (cm)',color='#176b87')
ax[0].plot(z['time'],pitch,label='Body pitch (degrees)',color='#bd453a')
ax[0].set_xlabel('Time (s)');ax[0].legend();ax[0].set_title('One representative final-policy episode')
ax[1].plot(z['time'],change,color='#208860');ax[1].set_ylabel('Max target correction (degrees)')
ax[1].set_xlabel('Time (s)');ax[1].set_title('Joint targets remain near the initial pose')
for a in ax:a.grid(alpha=.2)
fig.savefig(FIG/'fall_and_actions.png',dpi=170);fig.savefig(FIG/'fall_and_actions.svg');plt.close(fig)

manifest={'created_at':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
 'checkpoint_sha256':stop['checkpoint_sha256'],'production_source':'frozen H16 runner, unchanged',
 'files':{str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.rglob('*') if p.is_file() and p.suffix in ['.py','.json','.npz'] and p.name!='manifest.json'}}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(summary,indent=2))
