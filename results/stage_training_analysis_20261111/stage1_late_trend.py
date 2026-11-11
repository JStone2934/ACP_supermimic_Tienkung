"""Assess late Stage1 trends from existing logs; does not launch training."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path.cwd()
SOURCE = ROOT/'logs/tienkung3_29dof/stage1_php_all_spheres224_20261110/train/train_log.csv'
OUT = ROOT/'docs/report/20261111_stage1_stage2_analysis'
with SOURCE.open() as f:
    rows = list(csv.DictReader(f))
d = {k: np.array([float(row[k]) for row in rows]) for k in rows[0]}
metrics = ['ep_ret_norm','sim_rew','term_frac','touch_precision','touch_recall','wm_err8',
           'phase_zero_frac','phase_motions-high-climb-76','phase_motions-low-climb-76',
           'phase_motions-high-step','phase_motions-low-step','phase_motions-locomotion']
def window(lo,hi):
    select=(d['iteration']>lo)&(d['iteration']<=hi)
    return dict(start=lo,end=hi,logged_windows=int(select.sum()),
                **{k:float(d[k][select].mean()) for k in metrics})
late = [window(lo,lo+5000) for lo in range(40000,60000,5000)]
fine = [window(lo,lo+2000) for lo in range(50000,60000,2000)]
snapshots=[]
for it in [30000,35000,40000,45000,50000,55000,60000]:
    i=np.flatnonzero(d['iteration']==it)[0]
    snapshots.append(dict(iteration=it,**{k:float(d[k][i]) for k in metrics if k.startswith('phase')}))
last = d['iteration']>55000
wall_hours=float(d['wall_h'][-1]-d['wall_h'][np.flatnonzero(d['iteration']==55000)[0]])
rate=5000/(wall_hours*3600)
result=dict(source=str(SOURCE.relative_to(ROOT)),sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            aggregation='Arithmetic mean of equal 25-iteration log windows, not reweighted by completed episode counts.',
            last_iteration=int(d['iteration'][-1]),windows_5000=late,windows_2000=fine,snapshots=snapshots,
            last_lr=float(d['lr'][-1]),last_5000_wall_hours=wall_hours,last_5000_iterations_per_second=rate,
            estimated_additional_20000_hours_excluding_restart_and_evaluation=20000/rate/3600,
            limitation='Changing curriculum and on-policy data; these are training trends, not a fixed-distribution evaluation curve.')
(OUT/'evidence/stage1_late_statistics.json').write_text(json.dumps(result,indent=2)+'\n')

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(3,2,figsize=(12,11),layout='constrained')
x=d['iteration']/1000
def rolling(ax,key,label,color,scale=1,late_only=True):
    select=d['iteration']>=40000 if late_only else np.ones(len(x),bool)
    ax.plot(x[select],d[key][select]*scale,color=color,alpha=.16,lw=.6)
    n=40
    sx=x[n-1:];sy=np.convolve(d[key],np.ones(n)/n,'valid')*scale
    select=sx>=40 if late_only else np.ones(len(sx),bool)
    ax.plot(sx[select],sy[select],color=color,lw=2,label=label)
    ax.set_xlabel('Stage1 iterations (k)')
    ax.grid(alpha=.2)
rolling(axs[0,0],'ep_ret_norm','Episode return','#2563eb')
axs[0,0].set(title='Late training return: still increasing',ylabel='Normalized episode return')
rolling(axs[0,1],'term_frac','Early termination','#ca4c6c',100)
axs[0,1].set(title='Late training termination: gradual reduction',ylabel='Episode termination (%)')
rolling(axs[1,0],'touch_precision','Contact precision','#08916b',100)
axs[1,0].set(title='Contact precision: continued improvement',ylabel='Precision (%)')
rolling(axs[1,1],'phase_zero_frac','Starts at frame 0','#8f4eac',100)
axs[1,1].set(title='More clips reach the full-length curriculum',ylabel='Clips with phase = 0 (%)')
for g,c in [('high-step','#e29a25'),('low-step','#08916b'),('high-climb-76','#8f4eac'),('low-climb-76','#ca4c6c')]:
    axs[2,0].plot(x,d['phase_motions-'+g]*100,label=g,color=c,lw=1.5)
axs[2,0].set(title='Full curriculum: previous plateaus were followed by progress',xlabel='Stage1 iterations (k)',ylabel='Mean starting phase (%)',ylim=(0,100))
axs[2,0].axvspan(40,60,color='#777',alpha=.06)
axs[2,0].legend(fontsize=8)
for g,c in [('high-climb-76','#8f4eac'),('low-climb-76','#ca4c6c')]:
    rolling(axs[2,1],'phase_motions-'+g,g,c,100)
axs[2,1].set(title='Climbing: slow progress, far from phase 0',ylabel='Mean starting phase (%)')
axs[2,1].legend(fontsize=8)
axs[2,0].grid(alpha=.2)
fig.suptitle('Stage1 continuation assessment at 60k\nFaint lines: logged values; bold lines: trailing 1,000-iteration average',fontsize=14)
fig.savefig(OUT/'figures/stage1_late_training.png',dpi=160)
fig.savefig(OUT/'figures/stage1_late_training.pdf')
print(json.dumps(result,indent=2))
