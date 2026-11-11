"""Reload and evaluate a held-arm standing checkpoint, or the source tracker."""
import argparse
import json
from pathlib import Path

import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ckpt', required=True)
    p.add_argument('--out', required=True, help='output stem for JSON/NPZ')
    p.add_argument('--data-root', default='data/tienkung3_29dof/standing_aiming')
    args = p.parse_args()
    from supermimic.acp.robots import get_robot, sha256
    from supermimic.acp.kinematics import RobotKinematics
    from supermimic.acp.motion import MotionLib
    from supermimic.acp.mdp import MDP
    from supermimic.acp.task import ContactCfg, ContactTask
    from supermimic.acp.standing import StandingTask
    from supermimic.acp.standing_eval import evaluate
    from supermimic.acp.env import TrackingSim
    import supermimic.acp.networks as networks
    torch.set_num_threads(8)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    ck = torch.load(args.ckpt, map_location='cpu')
    robot = get_robot('tienkung3_29dof')
    robot.validate_metadata(ck['robot_metadata'])
    dev = 'cuda:0'
    kin = RobotKinematics(dev, robot)
    root = Path(args.data_root)
    motion = MotionLib.from_dir(str(root / 'motions'), kin, dev)
    mdp = MDP(kin, motion, torch.tensor(robot.default_q, device=dev))
    cfg = dict(ck['ctc_cfg'], contacts_dir=str(root / 'contacts'), target_noise=0., scan_noise=0.)
    ctc = ContactTask(ContactCfg(**cfg), mdp, str(root / 'motions'), dev)
    task = StandingTask(mdp, ctc)
    if ck.get('task_config') is not None and ck['task_config'] != task.metadata():
        raise ValueError('standing checkpoint task configuration mismatch')
    if ck.get('task_config') and ck['data_hash'] != motion.data_hash:
        raise ValueError('standing checkpoint dataset mismatch')
    networks.AMP_DTYPE = torch.bfloat16 if ck['args']['amp'] == 'bf16' else None
    pi = networks.Policy(mdp.obs_dim, ctc.cmd_dim, hist=ck['args']['hist'], robot=robot).to(dev)
    pi.load_state_dict(ck['pi'])
    env = TrackingSim(1, motion, mdp, dev, eval_mode=True, control_task=task)
    report = evaluate(pi, env, task, args.out, iteration=ck['iteration'], checkpoint=str(Path(args.ckpt).resolve()))
    report['checkpoint_sha256'] = sha256(args.ckpt)
    Path(args.out).with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('metrics', 'task_config')}, indent=2))


if __name__ == '__main__':
    main()
