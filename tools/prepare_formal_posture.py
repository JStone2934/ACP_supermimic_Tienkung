"""Explicit restart of the selected aiming policy at curriculum level0."""
import argparse,json,subprocess,sys
from pathlib import Path
import numpy as np
import torch
from supermimic.acp.aiming_wide import WideAimingTask
from supermimic.acp.aiming_fullbody import validate_checkpoint
from supermimic.acp.aiming_fullbody_io import save_checkpoint
from supermimic.acp.robots import sha256
from tools.run_posture_curriculum import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--reference',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(2)
    if (a.out/'init.pt').exists():raise FileExistsError('Formal initialization already exists')
    a.out.mkdir(parents=True,exist_ok=True)
    subprocess.run([sys.executable,'-m','tools.prepare_posture_continuation','--source',str(a.source),
        '--reference',str(a.reference),'--out',str(a.out/'source_contract.pt')],check=True)
    ck=torch.load(a.out/'source_contract.pt',map_location='cpu',weights_only=False)
    reference=np.load(a.reference,allow_pickle=False)['repaired_pose'].tolist()
    cfg=dict(ck['config'],formal_phase_id=0,curriculum_level=0,initialization_mode='formal_curriculum_restart',
             reset_mixture=dict(probability=0.,reference_pose=reference,reference_sha256=sha256(a.reference)),
             iterations=0,ckpt_every=100,log_every=10,wm_validation_every=10,
             wm_validation_steps=512,switch_interval_steps=None)
    task=WideAimingTask('cpu',cfg)
    q=task.alpha.new_tensor(reference[7:])
    if ((q<task.mdp.soft_lo)|(q>task.mdp.soft_hi)).any():raise ValueError('Repaired reset violates soft limits')
    ck.update(config=cfg,task_config=task.metadata(),provenance=dict(ck['provenance'],
        formal_curriculum_restart=True,source_stage=4,initial_stage=0,acceptance='Original task gates only; posture soft preference'))
    validate_checkpoint(ck,task);save_checkpoint(a.out/'init.pt',ck)
    write_json(a.out/'init.json',dict(config=cfg,provenance=ck['provenance'],source_checkpoint=ck['source_checkpoint'],sha256=sha256(a.out/'init.pt')))


if __name__=='__main__':main()
