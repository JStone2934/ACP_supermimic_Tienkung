import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path.cwd()
OUT = ROOT/'results/stage_training_analysis_20261111'
S1 = ROOT/'logs/tienkung3_29dof/stage1_php_all_spheres224_20261110'
S2 = ROOT/'logs/tienkung3_29dof/stage2_php_all_spheres224_20261110'
def read(path):
    with path.open() as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}
def smooth(x,n=40):
    return np.convolve(x,np.ones(n)/n,'valid')
s1 = read(S1/'train/train_log.csv')
s2 = read(S2/'joint/train_log.csv')
pv = read(S2/'joint/planner_val.csv')
groups = ['locomotion','low-step','high-step','low-climb-76','high-climb-76']
colors = ['#2563eb','#08916b','#e29a25','#ca4c6c','#8f4eac']
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes = plt.subplots(3,2,figsize=(13,12),layout='constrained')
ax=axes[0,0]
for g,c in zip(groups,colors):
    ax.plot(s1['iteration']/1000,s1['phase_motions-'+g],label=g,color=c,lw=1.4)
ax.set(title='Stage1: curriculum start phase (0 = full clip)',xlabel='Stage1 iterations (k)',ylabel='Mean starting phase',ylim=(-.03,1))
ax.legend(fontsize=8)
ax=axes[0,1]
for k,l,c in [('ep_ret_norm','Episode return (surviving portion)','#2563eb'),('term_frac','Early termination fraction','#ca4c6c')]:
    ax.plot(s1['iteration'][39:]/1000,smooth(s1[k]),label=l,color=c)
ax.set(title='Stage1: 1,000-iteration rolling averages',xlabel='Stage1 iterations (k)')
ax.legend(fontsize=8)
ax=axes[1,0]
for k,l,c in [('ep_ret_norm','Episode return','#2563eb'),('term_frac','Early termination fraction','#ca4c6c'),('admit_rate_window','Hindsight admission rate','#08916b')]:
    ax.plot(s2['iteration'][39:]/1000,smooth(s2[k]),label=l,color=c)
ax.set(title='Stage2: tracking degrades before the window change',xlabel='Stage2 iterations (k)')
ax.legend(fontsize=8)
ax=axes[1,1]
for k,l,c in [('eta_obs','Planner target probability','#2563eb'),('eta_rew','Reward mixing coefficient','#ca4c6c')]:
    ax.plot(s2['iteration']/1000,s2[k],label=l,color=c)
ax.set(title='Stage2: handover schedules',xlabel='Stage2 iterations (k)',ylim=(-.03,1.05))
ax.legend(fontsize=8)
ax=axes[2,0]
for k,l,c in [('f1_left_foot','Left foot','#2563eb'),('f1_right_foot','Right foot','#08916b')]:
    ax.plot(pv['iteration']/1000,pv[k],label=l,color=c)
ax.set(title='Planner F1 on a changing validation buffer',xlabel='Stage2 iterations (k)',ylim=(0,1))
ax.legend(fontsize=8)
ax=axes[2,1]
for k,l,c in [('touch_precision','Contact precision','#ca4c6c'),('touch_recall','Contact recall','#2563eb'),('wm_err8','Normalized WM error at 8 steps','#8f4eac')]:
    ax.plot(s2['iteration'][39:]/1000,smooth(s2[k]),label=l,color=c)
ax.set(title='Stage2: reference contact alignment and model error',xlabel='Stage2 iterations (k)',ylim=(0,1.05))
ax.legend(fontsize=8)
for ax in axes[1:].flat:
    ax.axvline(12,ls=':',c='#888',lw=1)
    ax.axvline(24,ls='--',c='#555',lw=1)
for ax in axes.flat:
    ax.grid(alpha=.2)
fig.suptitle('TienKung Stage1 / Stage2 diagnosis\nDotted: reward ramp starts at 12k; dashed: termination window changes at 24k',fontsize=14)
fig.savefig(OUT/'training_diagnosis.png',dpi=170)
fig.savefig(OUT/'training_diagnosis.pdf')
plt.close(fig)

evaluations={name:json.loads(path.read_text()) for name,path in [
    ('s1_strict',S1/'stage1_final_eval.json'),('s1_matched',S1/'stage1_matched_eval.json'),
    ('s2_matched',S2/'stage2_final_eval.json'),('s2_strict',S2/'stage2_strict_eval.json')]}
summary={}
for name,d in evaluations.items():
    per_group={}
    for group in d['per_folder']:
        rows=[v for n,v in d['per_clip'].items() if n.startswith(group+'/')]
        per_group[group]={'count':len(rows),'n_failed':sum(v['failed'] for v in rows),
                          **{k:float(np.median([v[k] for v in rows])) for k in
                             ['root_rmse_m','joint_rmse_rad','torque_saturation_fraction','velocity_exceed_fraction']}}
    summary[name]=dict(norm_return=d['norm_return'],n_failed=d['n_failed'],groups=per_group,contacts=d['contacts'])
windows={}
for tag,data,bounds in [('s1',s1,[(5000,10000),(20000,30000),(40000,50000),(55000,60000)]),
                       ('s2',s2,[(0,2000),(6000,10000),(12000,16000),(20000,24000),(24000,25000),(35000,40000)])]:
    windows[tag]=[]
    for lo,hi in bounds:
        select=(data['iteration']>lo)&(data['iteration']<=hi)
        row={'start':lo,'end':hi}
        for k,v in data.items():
            finite=v[select & np.isfinite(v)]
            if len(finite):row[k]=float(finite.mean())
        windows[tag].append(row)
(OUT/'summary.json').write_text(json.dumps(dict(full_evaluations=summary,training_windows=windows),indent=2))
print('Wrote training plots and summary.json')
