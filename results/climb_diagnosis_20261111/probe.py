"""Bounded diagnostic rollouts; reuses the evaluator initialization, no training."""
import sys,json,hashlib
from pathlib import Path
ROOT=Path.cwd(); OUT=ROOT/'results/climb_diagnosis_20261111'
sys.path.insert(0,str(ROOT))
checkpoint=ROOT/'logs/tienkung3_29dof/stage1_continue_80k_20261111/train/ckpt_100000.pt'
sys.argv=['probe','--ckpt',str(checkpoint),'--motion_dir',str(ROOT/'results/stage_training_analysis_20261111/eval_motions'),'--contacts_dir',str(ROOT/'data/tienkung3_29dof/stage1/php_all_variants_20261110/contacts'),'--term_window','0']
source=(ROOT/'supermimic/acp/evaluate.py').read_text()
exec(compile(source.split('@torch.no_grad()')[0],str(ROOT/'supermimic/acp/evaluate.py'),'exec'))
from supermimic.acp.networks import WorldModel
from supermimic.acp.mdp import normalize_xi
wm=WorldModel(mdp.scan_dim,robot=robot).to(dev);wm.load_state_dict(ck['wm']);wm.eval()
eff=torch.tensor(robot.effort,device=dev)[env.to_internal]
all_results={}
@torch.no_grad()
def probe(mode,phase):
    start=motion.clip_start+(phase*(motion.clip_len-1)).long()
    env.reset(torch.arange(C,device=dev),start)
    env.done_eval.zero_();env.failed_eval.zero_()
    gs=torch.full((C,),50,device=dev);hist=None
    counts=torch.zeros(C,device=dev); sums=torch.zeros(C,8,device=dev)
    end=torch.zeros(C,10,device=dev); touch_n=torch.zeros(C,9,device=dev); label_n=touch_n.clone();tp=touch_n.clone()
    wm_rows=[]; steps=0; predicted=None; block_alive=None
    while not bool(env.done_eval.all()):
        S=env.state();o=env.observe();f=env.f.clone();alive=~env.done_eval.clone()
        hist=o[:,None].repeat(1,Hh,1) if hist is None else torch.cat((hist[:,1:],o[:,None]),1)
        if mode=='policy':
            a=pi(hist,ctc.command(S,f,env.v,gs,noise=False))
        else:
            fn=torch.minimum(f+1,motion.clip_end)
            target=motion.q[fn]
            if mode=='pd_velocity':target=target+torch.tensor(robot.kd,device=dev)/torch.tensor(robot.kp,device=dev)*motion.qd[fn]
            a=(target-env.default_q)/env.alpha
        # Open-loop world-model replay of exactly the real actions, reset every 32 steps.
        if steps%32==0:
            predicted=tuple(x.clone() for x in S);block_alive=alive.clone()
        if mode=='policy' and phase==0:
            pred1=wm.step(S,a,normalize_xi(env.xi),mdp.scan(S,motion.boxes(f,env.v)))
            predicted=wm.step(predicted,a,normalize_xi(env.xi),mdp.scan(predicted,motion.boxes(f,env.v)))
        out=env.step(a);S1=env.state();f1=env.f.clone()
        root=S1[0]-motion.root_p[f1];joint=(S1[4]-motion.q[f1]).square().mean(-1).sqrt()
        ez=root[:,2];tilt=(S1[1][:,2,2]-motion.root_R[f1,2,2]).abs()
        sat=(env.sim.data.actuator_force.abs()>=.99*eff).float().mean(-1)
        distance=ctc.body_distance(S1,motion.boxes(f1,env.v));touch=distance<=ctc.cfg.touch_dist
        lab=ctc.labels[f1,env.v];tp+=(touch&lab&alive[:,None]);touch_n+=(touch&alive[:,None]);label_n+=(lab&alive[:,None])
        sums+=torch.stack((root.square().sum(-1),joint.square(),sat,ez.abs(),distance[:,4],distance[:,5],(ez.abs()>.25).float(),(tilt>.8).float()),-1)*alive[:,None]
        counts+=alive
        ended=alive&env.done_eval
        e=torch.cat((root,joint[:,None],distance[:,4:6],ez[:,None],tilt[:,None],sat[:,None],f1[:,None].float()),-1)
        end[ended]=e[ended]
        if mode=='policy' and phase==0:
            block_alive &= alive
            err1=(pred1[0]-S1[0]).norm(dim=-1);err=(predicted[0]-S1[0]).norm(dim=-1)
            if steps%32 in [0,7,31]:
                wm_rows.append(dict(horizon=steps%32+1,alive=block_alive.cpu().tolist(),root_error=err.cpu().tolist(),one_step=err1.cpu().tolist(),height_bias=(predicted[0][:,2]-S1[0][:,2]).cpu().tolist()))
        env.f=torch.minimum(env.f,motion.clip_end-1);steps+=1
    rows={}
    for i,name in enumerate(motion.names):
        n=max(counts[i].item(),1);vals=sums[i].cpu().tolist();last=end[i].cpu().tolist()
        rows[name]=dict(failed=bool(env.failed_eval[i]),alive_steps=int(n),start_frame=int(start[i]-motion.clip_start[i]),end_phase=float((end[i,-1]-motion.clip_start[i])/(motion.clip_len[i]-1)),root_rmse=(vals[0]/n)**.5,joint_rmse=(vals[1]/n)**.5,saturation=vals[2]/n,terminal=dict(root_error=last[:3],joint_rmse=last[3],hand_distance=last[4:6],height_error=last[6],tilt_error=last[7],saturation=last[8]),touch_frames=touch_n[i].cpu().tolist(),label_frames=label_n[i].cpu().tolist(),true_touch_frames=tp[i].cpu().tolist())
    if wm_rows:(OUT/'wm_replay.json').write_text(json.dumps(dict(names=motion.names,rows=wm_rows)))
    return rows
for mode,phase in [('policy',0.),('policy',.4),('policy',.6),('policy',.8),('pd',0.),('pd_velocity',0.)]:
    key=f'{mode}_{phase}'
    print('START',key,flush=True)
    all_results[key]=probe(mode,phase)
    (OUT/'rollouts.json').write_text(json.dumps(all_results))
    for folder in FOLDERS:
        rr=[r for n,r in all_results[key].items() if n.startswith(folder+'/')]
        print(key,folder,'complete',sum(not r['failed'] for r in rr),'/',len(rr),'mean_steps',sum(r['alive_steps'] for r in rr)/len(rr),flush=True)
(OUT/'provenance.json').write_text(json.dumps(dict(checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),evaluator_sha256=hashlib.sha256(source.encode()).hexdigest(),cohort='existing fixed 100-clip cohort; 20 per class; training data, not held out',notes='PD controls use same action target interface, gains, effort limits, free base and termination as policy. PD failure alone is not proof of infeasibility.'),indent=2))
