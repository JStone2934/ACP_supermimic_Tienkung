"""Migrate an aiming checkpoint to a world-fixed chest-forward target frame."""
import argparse, json
from pathlib import Path
import numpy as np
import torch
from supermimic.acp.aiming import setup, json_save
from supermimic.acp.aiming_curriculum import Curriculum, sample_parameters, set_target_pool
from supermimic.acp.robots import sha256
from tools.prepare_aiming import Teacher

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',required=True);ap.add_argument('--out',required=True)
    args=ap.parse_args();torch.set_num_threads(2)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    ck=torch.load(args.source,map_location='cpu',weights_only=False)
    cfg=dict(ck['config'],target_frame='chest_forward')
    task=setup(cfg,'cpu');teacher=Teacher(task);teacher.geometry=teacher._geometry
    set_target_pool(None)
    ref=task.mdp.motion.ref_state(torch.zeros(1,dtype=torch.long))
    rng=np.random.default_rng(cfg['seed'])
    angles=np.r_[np.zeros((1,2)),rng.uniform(-np.deg2rad([15,10]),np.deg2rad([15,10]),(256,2))]
    params=sample_parameters(dict(mode=0,depth=(3,3),angle=(0,0),rho=.2),len(angles)*3,rng,'cpu')
    params[:,1]=torch.tensor([2.,3.,4.]).repeat(len(angles))
    params[:,4:6]=torch.tensor(angles,dtype=torch.float32).repeat_interleave(3,0)
    S=tuple(x.expand(len(params),*x.shape[1:]).clone() for x in ref)
    goal=task.target(params,torch.zeros(len(params)))
    q,valid,error,margin,_=teacher.solve(S,goal[:,:3],S[4][:,task.terminal])
    report=dict(target_frame=task.metadata()['target_frame'],central_valid=valid[:3].tolist(),
                central_error_deg=error[:3].tolist(),central_margin_deg=torch.rad2deg(margin[:3]).tolist(),
                central_terminal_q=q[:3].tolist(),central_targets=goal[:3,:3].tolist(),
                terminal_names=[task.mdp.robot.joint_names[i] for i in task.terminal.tolist()],
                terminal_soft_lo=task.mdp.soft_lo[task.terminal].tolist(),terminal_soft_hi=task.mdp.soft_hi[task.terminal].tolist(),
                candidates=256,valid_angles=int(valid.reshape(-1,3).all(-1)[1:].sum()))
    json_save(out/'geometry.json',report)
    if not bool(valid[:3].all()):
        raise RuntimeError('Chest central target is unreachable within existing distal joint limits; see geometry.json')
    keep=valid.reshape(-1,3).all(-1)
    pool=torch.cat((torch.full((int(keep.sum()),1),3.),torch.tensor(angles,dtype=torch.float32)[keep]),-1)
    torch.save(pool,out/'target_pool.pt')
    ck.update(config=cfg,task_config=task.metadata(),target_pool=pool,
              curriculum=Curriculum(cfg['seed']).state_dict(),
              recovery=dict(active=False,bad=0,good=0,release=None,last_stable=0),best=None,
              migration=dict(source_checkpoint=str(Path(args.source).resolve()),source_sha256=sha256(args.source),
                             target_pool_sha256=sha256(out/'target_pool.pt'),target_frame='chest_forward',reset_curriculum=True))
    train=out/'train';train.mkdir(exist_ok=True)
    torch.save(ck,train/'last.pt');json_save(out/'config.json',cfg);json_save(out/'migration.json',ck['migration'])
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
