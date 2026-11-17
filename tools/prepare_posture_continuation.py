"""Create a new reward contract while preserving the old policy and reset pose."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from supermimic.acp.aiming_fullbody import validate_checkpoint
from supermimic.acp.aiming_wide import WideAimingTask
from supermimic.acp.robots import sha256
from supermimic.acp.standing import ARM_NAMES


def main():
    torch.set_num_threads(2)
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--reference-weight',type=float,default=.2)
    args=p.parse_args()
    if args.out.exists():raise FileExistsError(args.out)
    ck=torch.load(args.source,map_location='cpu',weights_only=False)
    validate_checkpoint(ck,WideAimingTask('cpu',ck['config']))
    reference=np.load(args.reference,allow_pickle=False)
    robot=WideAimingTask('cpu',ck['config']).robot
    if list(reference['joint_names'])!=list(robot.joint_names):raise ValueError('Reference joint order mismatch')
    cfg=dict(ck['config']);weights=dict(cfg['reward_weights'])
    if any(k in cfg for k in ('chest_posture','posture_reference','policy_retention')):
        raise ValueError('Start from the original aiming policy, not a prior posture experiment')
    if not np.isfinite(args.reference_weight) or args.reference_weight<=0:
        raise ValueError('Reference weight must be positive and finite')
    names=[n for n in robot.joint_names if n not in ARM_NAMES]
    if len(names)!=22 or len(ARM_NAMES)!=7:raise ValueError('Expected 22 posture joints and 7 right-arm joints')
    weights['posture_reference']=args.reference_weight
    cfg.update(reward_weights=weights,
        posture_reference=dict(q=reference['repaired_pose'][7:].tolist(),scale_rad=.35,
            joint_names=names,source_sha256=sha256(args.reference)),
        policy_training_scope='full',actor_lr=1e-6,actor_value_gradient=0.,
        distill_iters=0,wm_only_iters=100,critic_warmup_iters=0,
        num_envs=512,wm_batch=256,pi_batch=128,buffer_steps=512,
        wm_window=32,initial_collect_steps=128,wm_batches=5,
        wm_validation_every=1,wm_validation_steps=512,wm_validation_envs=8,
        wm_gate_chest_deg=2.,wm_gate_pelvis_deg=2.,validation_horizons=[1,8,16,32],
        iterations=101,ckpt_every=100,log_every=1,compile_training=True)
    task=WideAimingTask('cpu',cfg)
    # Calibration stage means new critic/optimizers; do not resume a stale value function.
    artifact=dict(task=ck['task'],stage='calibration',smoke=False,config=cfg,task_config=task.metadata(),
        robot_metadata=ck['robot_metadata'],pi=ck['pi'],wm=ck['wm'],
        source_checkpoint=dict(path=str(args.source.resolve()),sha256=sha256(args.source)),
        provenance=dict(kind='22_joint_posture_transfer_v2',reference=str(args.reference.resolve()),
                        initial_pose_preserved=True,policy_weights_preserved=True,
                        original_reward_weights_preserved=True,excluded_right_arm=list(ARM_NAMES),
                        scope='Experimental initialization, not acceptance'))
    validate_checkpoint(artifact,task)
    args.out.parent.mkdir(parents=True,exist_ok=True);torch.save(artifact,args.out)
    args.out.with_suffix('.json').write_text(json.dumps(dict(config=cfg,provenance=artifact['provenance'],
        source_checkpoint=artifact['source_checkpoint'],sha256=sha256(args.out)),indent=2)+'\n')


if __name__=='__main__':main()
