"""Read-only multi-horizon dynamics diagnostic on physical training-source states."""
import argparse
import json
from pathlib import Path
import torch
from supermimic.acp.robots import get_robot,sha256
from supermimic.acp.kinematics import RobotKinematics
from supermimic.acp.motion import MotionLib
from supermimic.acp.mdp import MDP,RewardCfg,normalize_xi,unflatten_state
from supermimic.acp.networks import Policy,WorldModel
from supermimic.acp.task import ContactCfg,ContactTask
from supermimic.acp.env import TrackingSim,make_sim


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--count',type=int,default=512)
    a=p.parse_args();torch.set_num_threads(2);torch.manual_seed(123)
    digest=sha256(a.checkpoint);ck=torch.load(a.checkpoint,map_location=a.device,weights_only=False);ca=ck['args']
    robot=get_robot('tienkung3_29dof');robot.validate_metadata(ck['robot_metadata'])
    kin=RobotKinematics(a.device,robot);motion=MotionLib.from_dir(ca['motion_dir'],kin,a.device)
    assert motion.data_hash==ck['data_hash']
    mdp=MDP(kin,motion,torch.tensor(robot.default_q,device=a.device),RewardCfg(term_root_distance=ca.get('tracking_root_radius')))
    ctc=ContactTask(ContactCfg(**ck['ctc_cfg']),mdp,ca['motion_dir'],a.device)
    pi=Policy(robot.obs_dim,ctc.cmd_dim,hist=10,robot=robot,pose_action_skip=ca.get('pose_action_skip',False)).to(a.device)
    wm=WorldModel(mdp.scan_dim,robot=robot).to(a.device)
    pi.load_state_dict(ck['pi']);wm.load_state_dict(ck['wm']);pi.eval();wm.eval()
    env=TrackingSim(a.count,motion,mdp,a.device,sim_cfg=make_sim(robot=robot,physics_substeps=ca['physics_substeps']))
    weights=json.loads(ca.get('dataset_weights') or '{"lafan":1.0}')
    from supermimic.acp.dataset_sampling import group_probabilities
    probabilities=group_probabilities(motion.names,weights,a.device)
    clips=torch.multinomial(probabilities,a.count,replacement=True)
    available=(motion.clip_len[clips]-65).clamp_min(1)
    frames=motion.clip_start[clips]+(torch.rand(a.count,device=a.device)*available).long()
    env.reset(torch.arange(a.count,device=a.device),frames)
    gs=torch.full_like(frames,50);history=None
    def step():
        nonlocal history
        obs=env.observe();fresh=env.ep_step==0
        h=obs[:,None].expand(-1,10,-1)
        history=h.clone() if history is None else torch.where(fresh[:,None,None],h,torch.cat((history[:,1:],obs[:,None]),1))
        action=pi(history,ctc.command(env.state(),env.f,env.v,gs,noise=True))
        action=(action+ca['action_noise']*torch.randn_like(action)).clamp(-20,20)
        return env.step(action)
    for _ in range(20):step()
    prediction=tuple(x.clone() for x in env.state());alive=torch.ones(a.count,dtype=torch.bool,device=a.device)
    floors=torch.cat((torch.full((3,),.003),torch.full((3,),.005),torch.full((3,),.02),torch.full((3,),.05),torch.full((29,),.003),torch.full((29,),.05))).to(a.device)
    std=torch.maximum(wm.out_norm.std(),floors)
    feet=[kin.tracked_names.index(robot.contact_bodies[n][0]) for n in ['left_foot','right_foot']]
    initial_groups={g:int(sum(motion.names[int(i)].startswith(g+'/') for i in motion.clip_id[env.f].tolist())) for g in weights}
    report=dict(scope=__doc__,checkpoint_sha256=digest,data_hash=motion.data_hash,states=a.count,
                physics_substeps=ca['physics_substeps'],sampling_weights=weights,initial_groups=initial_groups,
                contexts='Normal training friction/COM/reset and observation/action noise;20physical warmup steps',
                horizons={},limitation='Errors exclude transitions after physical reset. This evaluates predictions, not policy acceptance or gradient validity.')
    for t in range(32):
        tr=step();actual=unflatten_state(tr['s1'])
        prediction=wm.step(prediction,tr['a'],normalize_xi(tr['xi']),mdp.scan(prediction,motion.boxes(tr['f'],tr['v'])))
        if t+1 in (1,8,16,32):
            error=wm.state_error(prediction,actual);valid=alive&error.isfinite().all(-1)
            pk=kin.forward(*prediction)[0][:,feet];ak=kin.forward(*actual)[0][:,feet]
            report['horizons'][str(t+1)]=dict(valid_states=int(valid.sum()),nonfinite_while_alive=int((alive&~error.isfinite().all(-1)).sum()),
                normalized_mse=float((error[valid]/std).square().mean()),
                root_position_rmse_m=float((prediction[0][valid]-actual[0][valid]).square().sum(-1).mean().sqrt()),
                joint_position_rmse_rad=float((prediction[4][valid]-actual[4][valid]).square().mean().sqrt()),
                root_velocity_rmse_m_s=float((prediction[2][valid]-actual[2][valid]).square().sum(-1).mean().sqrt()),
                foot_position_rmse_m=float((pk[valid]-ak[valid]).square().sum(-1).mean().sqrt()))
        alive &= ~tr['done']
    assert sha256(a.checkpoint)==digest
    a.out.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
