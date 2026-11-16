"""Prepare wide-task initialization only from a validated JWP-finetuned tracker."""
import argparse
import json
from pathlib import Path
import torch
from supermimic.acp.robots import get_robot, sha256
from supermimic.acp.tracking_transfer import transfer_tracker
from supermimic.acp.aiming_wide import WideAimingTask
from supermimic.acp.aiming_fullbody_io import save_checkpoint, json_save


def verify_gates(checkpoint, reference_gate, standing_gate):
    digest=sha256(checkpoint)
    for name,report in [('reference',reference_gate),('standing',standing_gate)]:
        if report.get('checkpoint_sha256')!=digest or report.get('passed') is not True:
            raise ValueError(f'{name} gate is not passed for this checkpoint')
    if reference_gate.get('successful_fraction',0)<.9:
        raise ValueError('Reference completion criterion missing')
    if standing_gate.get('success_rate',0)<.95 or standing_gate.get('elapsed_sim_seconds',0)<30:
        raise ValueError('Continuous standing criterion missing')
    if not standing_gate.get('continuous_without_reset') or standing_gate.get('scenario_count',0)<100:
        raise ValueError('Standing evidence must cover 100 continuous episodes')
    return digest


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','reference-gate','standing-gate','training-seeds','out'):
        ap.add_argument('--'+name,required=True,type=Path)
    ap.add_argument('--seed-index',type=int,default=0)
    args=ap.parse_args();torch.set_num_threads(2)
    reference=json.loads(args.reference_gate.read_text());standing=json.loads(args.standing_gate.read_text())
    digest=verify_gates(args.checkpoint,reference,standing)
    if args.out.exists():raise FileExistsError(args.out)
    ck=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    weights=ck.get('args',{}).get('dataset_weights')
    if not weights or json.loads(weights)!={'jwp':.8,'lafan':.2}:
        raise ValueError('A tracker fine-tuned with 80% JWP / 20% LAFAN is required')
    if not ck.get('source_checkpoint'):
        raise ValueError('Missing LAFAN checkpoint provenance')
    seeds=json.loads(args.training_seeds.read_text())
    if seeds.get('data_hash')!=ck['data_hash'] or Path(seeds['motion_dir']).resolve()!=Path(ck['args']['motion_dir']).resolve():
        raise ValueError('Initial poses must come from this tracker training set')
    pose=seeds['seeds'][args.seed_index]
    robot=get_robot('tienkung3_29dof');policy,transfer=transfer_tracker(ck,robot,fixed_q=pose['initial_q'])
    config=dict(task='wide_aiming_v1',version=1,seed=0,hist=10,horizon=8,physics_substeps=transfer['physics_substeps'],
        initial_q=pose['initial_q'],initial_height=pose['initial_height'],initial_quat=pose['initial_quat'],
        num_envs=2048,wm_batch=1024,pi_batch=1024,buffer_steps=1000,wm_window=8,wm_batches=10,
        collect_steps=3,initial_collect_steps=128,wm_only_iters=500,critic_warmup_iters=100,
        actor_lr=2e-5,wm_lr=1e-4,critic_lr=1e-4,gamma=.99,lambda_return=.95,
        critic_epochs=5,critic_minibatches=4,ckpt_every=250,log_every=25,
        body_noise_rad=.005,arm_noise_rad=.02,noise_decay_updates=10000,noise_min_fraction=.1,
        actor_grad_clip_value=.1,term_height=.25,term_tilt_deg=30.,bound_value_bootstrap=True,
        reward_weights=dict(aim=4.,root=2.,feet=0.,height=1.,upright=1.,velocity=1.,angular_velocity=.2,
                            foot_velocity=.5,heading=0.,foot_boundary=0.,action_rate=-.1,joint_limit=-2.),
        source=dict(checkpoint=str(args.checkpoint.resolve()),sha256=digest,iteration=ck['iteration']))
    task=WideAimingTask('cpu',config)
    provenance=dict(reference_gate_sha256=sha256(args.reference_gate),standing_gate_sha256=sha256(args.standing_gate),
                    training_seeds_sha256=sha256(args.training_seeds),initial_pose=pose,transfer=transfer)
    save_checkpoint(args.out,dict(task='wide_aiming_v1',stage='initialization',robot_metadata=robot.metadata(),
        task_config=task.metadata(),config=config,pi=policy.state_dict(),wm=ck['wm'],provenance=provenance,
        source_checkpoint=config['source'],iteration=0))
    json_save(args.out.with_suffix('.json'),dict(config=config,provenance=provenance,
        scope='Initialization only; requires contact-interface distillation, ACP curriculum and independent acceptance'))


if __name__=='__main__':main()
