"""Re-evaluate saved supervision weights without updating them."""
import argparse,json
from pathlib import Path
import torch
from supermimic.acp.aiming import setup,policy,json_save
from supermimic.acp.mdp import unflatten_state

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--ckpt',required=True);a=ap.parse_args();torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=True
    ck=torch.load(a.ckpt,map_location='cpu',weights_only=False);t=setup(ck['config']);pi=policy(t,ck['pi']);d=torch.load(ck['dataset'],map_location='cpu',weights_only=False)
    ix=(d['valid']&d['validation'][:,None]).nonzero();values=[]
    with torch.no_grad():
        for ij in ix.split(512):
            x,y=ij.T;S=unflatten_state(d['s'][x,y].cuda());g=d['goal'][x,y].cuda();f=torch.zeros(len(x),dtype=torch.long,device='cuda')
            act=pi(d['hist'][x,y].cuda(),t.command(S,f,f,f+50,g,d['previous'][x,y].cuda()));q=t.mdp.default_q[t.terminal]+t.alpha[t.terminal]*act[:,t.terminal]
            qc=q.clamp(t.mdp.soft_lo[t.terminal],t.mdp.soft_hi[t.terminal]);S=list(S);S[4]=S[4].scatter(-1,t.terminal.expand_as(qc),qc)
            values.append((t.metrics(S,g)[1].cpu(),torch.rad2deg(q.cpu()-d['q_star'][x,y]),act[:,t.body].cpu()-d['body'][x,y]))
    angle,dq,db=[torch.cat([v[i] for v in values]) for i in range(3)]
    r=dict(ck['report']);r.update(direction_p95_deg=float(torch.quantile(angle,.95)),joint_rmse_deg=dq.square().mean(0).sqrt().tolist(),body_rmse=float(db.square().mean().sqrt()),metrics_version=2)
    r['passed']=r['direction_p95_deg']<=1 and max(r['joint_rmse_deg'])<=5 and r['body_rmse']<=.05
    ck['report']=r;ck['task_config']=t.metadata();ck['robot_metadata']=t.mdp.robot.metadata()
    from supermimic.acp.robots import sha256
    ck['dataset_sha256']=sha256(ck['dataset'])
    delta=d['goal'][...,:3][d['valid']]-t.h_ref.cpu();L=(delta*t.n0.cpu()).sum(-1)
    ck['target_pool']=torch.stack((L,torch.atan2((delta*t.e1.cpu()).sum(-1),L),torch.atan2((delta*t.e2.cpu()).sum(-1),L)),-1)[::4]
    torch.save(ck,a.ckpt);json_save(Path(a.ckpt).with_suffix('.validation.json'),r);print(json.dumps(r))

if __name__=='__main__':main()
