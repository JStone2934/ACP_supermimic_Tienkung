"""Diagnose held-out dynamics at 1/8/16/32 steps."""
import argparse
from pathlib import Path
import torch
from supermimic.acp.aiming_fullbody import FullbodyTask, FullbodyPolicy, validate_checkpoint
from supermimic.acp.aiming_fullbody_curriculum import Curriculum
from supermimic.acp.aiming_fullbody_validation import validate_world_model
from supermimic.acp.aiming_fullbody_io import json_save
from supermimic.acp.networks import WorldModel
from supermimic.acp.robots import sha256


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ckpt', required=True); p.add_argument('--out', required=True)
    p.add_argument('--horizon', type=int, choices=(8, 16, 32), default=32)
    args = p.parse_args(); torch.set_num_threads(8); torch.manual_seed(710)
    import warp as wp
    wp.config.quiet = True
    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    cfg = dict(ck['config'], horizon=args.horizon, wm_validation_steps=512, wm_validation_envs=16)
    task = FullbodyTask('cuda:0', cfg); validate_checkpoint(ck, task)
    pi = FullbodyPolicy(task.robot).cuda(); pi.load_state_dict(ck['pi'])
    wm = WorldModel(task.mdp.scan_dim, robot=task.robot).cuda(); wm.load_state_dict(ck['wm'])
    cur = Curriculum(task.alpha.device, ck.get('curriculum'))
    report = validate_world_model(pi, wm, task, lambda n: cur.sample(n, task.weights['aim']), cfg, args.out)
    report.update(checkpoint_sha256=sha256(args.ckpt), checkpoint=str(Path(args.ckpt).resolve()),
                  task_config=task.metadata(), curriculum_level=cur.level)
    json_save(Path(args.out).with_suffix('.json'), report)
    print(dict(passed=report['passed'], horizon=args.horizon))
