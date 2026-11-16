"""Verify every packaged input, then exercise the actual training data loaders."""
import argparse
import json
from pathlib import Path
import time

import torch

from tools.prepare_stage1_dataset import sha
from supermimic.acp.kinematics import RobotKinematics
from supermimic.acp.mdp import MDP
from supermimic.acp.motion import MotionLib
from supermimic.acp.robots import get_robot
from supermimic.acp.task import ContactCfg, ContactTask


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--contact-geometry', choices=('legacy', 'spheres224'), default='legacy')
    args = ap.parse_args()
    start = time.monotonic()
    data = json.loads((args.dataset/'manifest.json').read_text())
    for row in data['entries']:
        for field in ('motion', 'terrain', 'labels'):
            if sha(row[field]) != row[field+'_sha256']:
                raise ValueError(f'input changed: {row[field]}')
    print(f"verified {len(data['entries'])} input triplets", flush=True)
    robot = get_robot('tienkung3_29dof')
    robot.validate_metadata(data['robot_metadata'])
    kin = RobotKinematics('cuda:0', robot)
    print('loading full MotionLib', flush=True)
    motion = MotionLib.from_dir(str(args.dataset/'motions'), kin, 'cuda:0')
    expected = [e['key'] for e in data['entries']]
    if motion.names != sorted(expected) or motion.num_variants != 1 or not bool(motion.variant_ok.all()):
        raise ValueError('training loader filtered, duplicated, or lost original terrain slots')
    if motion.num_frames != data['total_frames']:
        raise ValueError('training loader changed frame coverage')
    mdp = MDP(kin, motion, torch.tensor(robot.default_q, device='cuda:0'))
    print('loading full ContactTask', flush=True)
    ctc = ContactTask(ContactCfg(contacts_dir=str(args.dataset/'contacts'), contact_geometry=args.contact_geometry), mdp, str(args.dataset/'motions'), 'cuda:0')
    for start_frame in range(0, motion.num_frames, 4096):
        f = torch.arange(start_frame, min(motion.num_frames, start_frame+4096), device='cuda:0')
        cmd = ctc.command(motion.ref_state(f), f, torch.zeros_like(f), torch.full_like(f, 50), noise=False)
        if not bool(torch.isfinite(cmd).all()):
            raise ValueError('nonfinite contact command')
    result = dict(passed=True, sources=data['source_count'], original_pairs=len(motion.names),
                  sampleable_pairs=int(motion.variant_ok.sum()), frames=motion.num_frames,
                  data_hash=motion.data_hash, dataset_manifest_sha256=sha(args.dataset/'manifest.json'),
                  robot_metadata=robot.metadata(), command_dim=ctc.cmd_dim,
                  contact_geometry=args.contact_geometry, reward_points=len(ctc.sph_pts), reward_hulls=len(ctc.hulls),
                  seconds=time.monotonic()-start, gpu_peak_allocated=torch.cuda.max_memory_allocated())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
