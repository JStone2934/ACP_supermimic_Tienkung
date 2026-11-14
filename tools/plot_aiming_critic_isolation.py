"""Render calibration and matched-lane direction results from saved diagnostics."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--experiment',required=True)
    a=p.parse_args();out=Path(a.experiment)
    data=json.loads((out/'report.json').read_text())
    recheck=json.loads((out/'direction_recheck'/'report.json').read_text())
    c=data['calibration'];rounds=np.array([x['extra_rounds'] for x in c]);x=rounds
    plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False,'font.size':10})
    fig,axes=plt.subplots(2,2,figsize=(12,8.5),layout='constrained')
    ax=axes[0,0]
    for metric,label in [('mae','平均绝对误差'),('rmse','均方根误差')]:
        y=[v['bounded'][metric] for v in c];ax.plot(x,y,'o-',label=label)
        if metric=='mae':
            for xx,yy in zip(x,y):ax.annotate(f'{yy:.2f}',(xx,yy),xytext=(0,7),textcoords='offset points',ha='center')
    ax.set(title='真实物理累计回报：估计误差下降',xlabel='额外 critic 训练轮数',ylabel='回报误差',xticks=x,xticklabels=rounds)
    ax.legend();ax.grid(alpha=.2)
    ax=axes[0,1]
    for v,color in [(c[0],'#d47239'),(c[-1],'#177b9e')]:
        ax.scatter(v['true_returns'],v['predictions_bounded'],s=20,alpha=.7,color=color,label=f"额外 {v['extra_rounds']} 轮")
    lower=min(c[0]['true_returns'])-1;upper=max(max(v['predictions_bounded']) for v in c)+1
    ax.plot([lower,upper],[lower,upper],'--',color='gray',linewidth=1)
    ax.set(title='固定 64 个评估状态，每状态 2 次物理复现',xlabel='真实折扣累计回报',ylabel='critic 估计（实际使用的上界裁剪后）')
    ax.legend();ax.grid(alpha=.2)
    ax=axes[1,0];ax.semilogy(x,[v['fixed_bank_td_mse'] for v in c],'o-',color='#7951a8')
    ax.set(title='固定想象样本的 TD 误差下降',xlabel='额外 critic 训练轮数',ylabel='TD 均方误差（对数坐标）',xticks=x,xticklabels=rounds)
    ax.grid(alpha=.2)
    ax=axes[1,1];rows=[v for v in recheck['results'] if v['epsilon_rad']==.02]
    ax.plot(x,[v['model_discounted_tail']['mean'] for v in rows],'s--',label='世界模型 + critic 预测的尾段收益',color='#d47239')
    ax.errorbar(x,[v['physical_discounted_tail']['mean'] for v in rows],
                yerr=[v['physical_discounted_tail']['paired_state_se'] for v in rows],
                fmt='o-',capsize=4,label='真实物理尾段收益（±1 SE）',color='#177b9e')
    ax.plot(x,[v['critic_at_physical_tail']['mean'] for v in rows],'x:',label='critic 在真实第 8 步状态上的估计',color='#718b31')
    ax.axhline(0,color='gray',linewidth=1)
    ax.set(title='动作方向复核：59 个状态 × 4 次复现',xlabel='额外 critic 训练轮数',ylabel='沿梯度 − 逆梯度的折扣尾段回报差',xticks=x,xticklabels=rounds)
    ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.suptitle('Critic 隔离实验：固定第 500 轮 actor / 世界模型，H = 8',fontsize=15)
    control=recheck['baseline_control']
    fig.supxlabel('每轮 20 次 critic 更新；方向扰动上限 ±0.02 rad；误差线为状态均值标准误。\n'
                  f"不改动作的重复对照：平均回报差 {control['mean']:+.3f}，SE {control['paired_state_se']:.3f}；未确认梯度改善。",fontsize=9)
    fig.savefig(out/'critic_isolation.png',dpi=170);fig.savefig(out/'critic_isolation.pdf')
    print(out/'critic_isolation.png')


if __name__=='__main__':main()
