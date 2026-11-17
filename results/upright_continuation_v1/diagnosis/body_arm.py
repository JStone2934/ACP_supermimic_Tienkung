import sys,json
from pathlib import Path
P=Path('/home/js/robot/locomotion/standing-aiming/results/upright_continuation_v1');sys.path.insert(0,str(P/'runner'))
import torch,numpy as np
from supermimic.acp.aiming_fullbody import FullbodyPolicy
from supermimic.acp.aiming_wide import WideAimingTask
from supermimic.acp.aiming_fullbody_env import FullbodySim
from supermimic.acp.wide_aiming_validation import episode_metrics
from supermimic.rotations import mat_to_quat

torch.set_num_threads(2);torch.set_grad_enabled(False);torch.manual_seed(17);dev='cuda:0'
c=torch.load(P/'train/ckpt_000100.pt',map_location=dev,weights_only=False);n=torch.load(P/'train/ckpt_000105.pt',map_location=dev,weights_only=False);task=WideAimingTask(dev,c['config'])
old=FullbodyPolicy(task.robot).to(dev);old.load_state_dict(c['pi']);old.eval();new=FullbodyPolicy(task.robot).to(dev);new.load_state_dict(n['pi']);new.eval()
ids=[0,7,14,21,28,35,42,49,56,63,70,77,100,120,150,174];N=len(ids);params=torch.tensor(json.loads((P/'baseline/seed17.json').read_text())['target_parameters'][0],device=dev)[ids].repeat(4,1)
env=FullbodySim(task,4*N,lambda count:(params,4.),eval_mode=True);z=np.load(P/'baseline/seed17.npz');pose=torch.tensor(np.concatenate((z['root'][0,ids],z['root_quat'][0,ids]),-1),device=dev).repeat(4,1);q=torch.tensor(z['q'][0,ids],device=dev).repeat(4,1);allids=torch.arange(4*N,device=dev)
pose[:,:3]+=env.scene.env_origins;env.robot.write_root_link_pose_to_sim(pose,env_ids=allids);env.robot.write_joint_position_to_sim(q[:,env.to_internal],env_ids=allids);env.robot.write_root_link_velocity_to_sim(torch.zeros(4*N,6,device=dev),env_ids=allids);env.robot.write_joint_velocity_to_sim(torch.zeros_like(q),env_ids=allids);env.robot.set_joint_position_target(q[:,env.to_internal]);env.sim.forward();env.a_prev[:]=(q-task.mdp.default_q)/task.alpha;env.context=task.context(env.state(),params,4.)
rows=[];hist=None;failed=torch.zeros(4*N,dtype=torch.bool,device=dev)
for step in range(1501):
 S=env.state();fk=task.kin.forward(*S);err=task.metrics(S,env.context)[0];finite=torch.stack([x.flatten(1).isfinite().all(-1) for x in S]).all(0)
 rows.append(dict(root=S[0].cpu().numpy().copy(),feet=fk[0][:,task.feet].cpu().numpy().copy(),tilt=torch.rad2deg(torch.acos(S[1][:,2,2].clamp(-1,1))).cpu().numpy(),chest=torch.rad2deg(torch.acos(fk[1][:,task.torso,2,2].clamp(-1,1))).cpu().numpy(),error=err.cpu().numpy(),finite=finite.cpu().numpy()))
 if step==1500:break
 obs=env.observe();hist=obs[:,None].repeat(1,10,1) if hist is None else torch.cat((hist[:,1:],obs[:,None]),1);cmd=task.command(S,env.context);a=old(hist,cmd);b=new(hist,cmd);a[N:2*N]=b[N:2*N]
 body=[i for i in range(29) if i not in task.arm_ids.tolist()];a[2*N:3*N,body]=b[2*N:3*N,body];a[3*N:,task.arm_ids]=b[3*N:,task.arm_ids];tr=env.step(a);failed|=tr['terminated']
 if step%500==0:print('step',step,flush=True)
a={k:np.stack([r[k] for r in rows]) for k in rows[0]};reports={}
for g,name in enumerate(['old','new','new_body_old_arm','old_body_new_arm']):
 inds=range(g*N,(g+1)*N);s=[episode_metrics(*(a[k][:,i] for k in ['root','feet','tilt','error','finite']),bool(failed[i]),switching=False,radius=.1,aim_required=True) for i in inds];e=a['error'][150:,g*N:(g+1)*N]
 reports[name]=dict(full=sum(x['passed'] for x in s),standing=sum(x['standing_pass'] for x in s),terminated=int(failed[g*N:(g+1)*N].sum()),error_median=float(np.median(e)),chest_median=float(np.median(a['chest'][150:,g*N:(g+1)*N])),scenarios=s)
result=dict(scope='16 selected seed17 scenarios; exact saved initial pose; mix output actions from old/new policies on identical current observations, not parameter mixing; no training',source_ids=ids,results=reports)
(P/'diagnosis/body_arm.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:{kk:vv for kk,vv in v.items() if kk!='scenarios'} for k,v in reports.items()},indent=2))
