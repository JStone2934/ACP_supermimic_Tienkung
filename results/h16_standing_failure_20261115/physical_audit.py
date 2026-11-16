"""Finite physical controls and fresh validation; no training state is modified."""
import sys
from pathlib import Path
ROOT = Path('/home/js/robot/locomotion/standing-aiming')
RUN = ROOT/'logs/tienkung3_29dof/standing_aiming_fullbody_v2/dual_h16_critic4096_reward_v2_20261114'
sys.path.insert(0, str(RUN/'runner'))
import json
import numpy as np
import torch
from supermimic.acp.aiming_fullbody import FullbodyTask, FullbodyPolicy
from supermimic.acp.aiming_fullbody_env import FullbodySim
from supermimic.acp.aiming_physical import execution_setup, rollout
from supermimic.acp.aiming_fullbody_validation import validate_world_model
from supermimic.acp.aiming_fullbody_curriculum import Curriculum
from supermimic.acp.networks import WorldModel
from supermimic.acp.standing import freeze_normalizers

OUT = Path(__file__).resolve().parent
torch.set_num_threads(4)
torch.cuda.set_device('cuda:0')
import warp as wp
wp.config.quiet = True
ck = torch.load(OUT/'checkpoint.pt', map_location='cpu', weights_only=False)
cfg = dict(ck['config'], compile_training=False)
execution_setup(cfg)
task = FullbodyTask('cuda:0', cfg)
pi = FullbodyPolicy(task.robot).to('cuda:0')
pi.load_state_dict(ck['pi']); freeze_normalizers(pi)
sample = lambda n: (task.alpha.new_tensor([[3., 0., 0., .2]]).repeat(n, 1), 0.)
env = FullbodySim(task, 64, sample, eval_mode=True)
report = dict(checkpoint_iteration=ck['iteration'], cases={}, static_sweep=[])

class Constant:
    def __init__(self, q): self.raw = (q-task.mdp.default_q)/task.alpha
    def __call__(self, history, command): return self.raw.expand(len(history), -1)

@torch.no_grad()
def run(label, controller):
    _, result = rollout(controller, task, env, cfg, seed=cfg['seed']+50000)
    report['cases'][label] = {k:v for k,v in result.items() if k!='per_environment'}
    print(label, {k:result[k] for k in ['survival_s','undiscounted_return','fell','standing_pass']}, flush=True)
    (OUT/'physical_audit.json').write_text(json.dumps(report, indent=2)+'\n')

run('constant_initial_pose', Constant(task.initial_q))
run('constant_robot_default', Constant(task.mdp.default_q))
for label, path in [
    ('original_random_policy', RUN.parent/'dual_h8_aggressive_20261114/train/initial.pt'),
    ('h16_initial_policy', RUN/'migration/initial_h16.pt'),
    ('final_policy', OUT/'checkpoint.pt')]:
    saved = torch.load(path, map_location='cpu', weights_only=False)
    pi.load_state_dict(saved['pi']); freeze_normalizers(pi)
    run(label, pi)
    del saved

# Record the first complete episode, including the terminal state, without resets.
with torch.no_grad():
    torch.manual_seed(cfg['seed']+50000)
    env.reset(torch.arange(env.N, device=env.device))
    history = env.observe()[:,None].repeat(1,10,1)
    alive = torch.ones(env.N,dtype=torch.bool,device=env.device)
    trace=[]
    for step in range(1500):
        S=env.state(); raw=pi(history,task.command(S,env.context)); tr=env.step(raw)
        pos, rot, vel, omega, q, qd=env.state()
        feet=task.kin.forward(*env.state())[0][:,task.feet]
        trace.append(dict(time=(step+1)*task.dt, root=pos.cpu().numpy(), rotation=rot.cpu().numpy(),
            q=q.cpu().numpy(), q_des=tr['q_des'].cpu().numpy(), qd=qd.cpu().numpy(),
            feet=feet.cpu().numpy(), alive=alive.cpu().numpy(), done=tr['done'].cpu().numpy(),
            reward=tr['r'].cpu().numpy()))
        alive &= ~tr['done']; history=torch.cat((history[:,1:],env.observe()[:,None]),1)
        if not alive.any(): break
    np.savez_compressed(OUT/'fall_trace.npz', **{k:np.array([r[k] for r in trace]) for k in trace[0]})
    times=[0,9,19,29,39,49,59,len(trace)-1]
    report['fall_trace_samples']=[]
    for i in times:
        row=trace[i];R=row['rotation'][0]
        report['fall_trace_samples'].append(dict(time=row['time'],root=row['root'][0].tolist(),
          pitch_deg=float(np.degrees(np.arctan2(-R[2,0],np.sqrt(R[2,1]**2+R[2,2]**2)))),
          roll_deg=float(np.degrees(np.arctan2(R[2,1],R[2,2]))),
          target_from_initial_max_rad=float(np.max(np.abs(row['q_des'][0]-task.initial_q.cpu().numpy()))),
          tracking_error_max_rad=float(np.max(np.abs(row['q'][0]-row['q_des'][0])))))
    report['joint_names']=list(task.robot.joint_names)
    report['first_target_delta_rad']=(trace[0]['q_des'][0]-task.initial_q.cpu().numpy()).tolist()
    report['last_target_delta_rad']=(trace[-1]['q_des'][0]-task.initial_q.cpu().numpy()).tolist()

# A predefined static target sweep tests physical feasibility; it does not fit a policy.
offsets=[-.10,-.05,0.,.05,.10]
cases=[(hip, ankle) for hip in offsets for ankle in offsets]
n_per=4
big=FullbodySim(task,len(cases)*n_per,sample,eval_mode=True)
targets=task.initial_q.repeat(len(cases)*n_per,1)
for i,(hip,ankle) in enumerate(cases):
    for side in ['l','r']:
        targets[i*n_per:(i+1)*n_per,task.robot.joint_names.index(f'hip_pitch_{side}_joint')]+=hip
        targets[i*n_per:(i+1)*n_per,task.robot.joint_names.index(f'ankle_pitch_{side}_joint')]+=ankle
with torch.no_grad():
    _, sweep=rollout(Constant(targets),task,big,cfg,seed=714)
for i,(hip,ankle) in enumerate(cases):
    row=dict(hip_offset_rad=hip,ankle_offset_rad=ankle)
    for key in ['survival_s','undiscounted_return','fell','standing_pass','root_max_m','feet_max_m','tilt_max_deg']:
        row[key]=float(np.mean(sweep['per_environment'][key][i*n_per:(i+1)*n_per]))
    report['static_sweep'].append(row)
print('best_static',sorted(report['static_sweep'],key=lambda x:x['survival_s'],reverse=True)[:5],flush=True)
(OUT/'physical_audit.json').write_text(json.dumps(report,indent=2)+'\n')

wm=WorldModel(task.mdp.scan_dim,robot=task.robot).to('cuda:0');wm.load_state_dict(ck['wm']);freeze_normalizers(wm)
cur=Curriculum(task.alpha.device)
torch.manual_seed(cfg['seed']+710)
report['fresh_wm_validation']=validate_world_model(pi,wm,task,lambda n:cur.sample(n,task.weights['aim']),cfg,OUT/'fresh_wm')
(OUT/'physical_audit.json').write_text(json.dumps(report,indent=2)+'\n')
print('physical_audit_complete',flush=True)
