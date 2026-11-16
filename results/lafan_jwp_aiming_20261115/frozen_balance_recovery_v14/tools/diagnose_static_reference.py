"""Thirty-second fixed-reference PD diagnostic, never learned-policy acceptance."""
import argparse
import json
from pathlib import Path
import torch
from supermimic.acp.env import TrackingSim, make_sim
from supermimic.acp.kinematics import RobotKinematics
from supermimic.acp.motion import MotionLib
from supermimic.acp.mdp import MDP
from supermimic.acp.robots import get_robot
from supermimic.acp.task import ContactCfg, ContactTask
from tools.evaluate_tracker_standing import quiet_frames


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    robot = get_robot('tienkung3_29dof')
    kin = RobotKinematics(args.device, robot)
    motion = MotionLib.from_dir(str(args.dataset/'motions'), kin, args.device)
    mdp = MDP(kin, motion, torch.tensor(robot.default_q, device=args.device))
    ctc = ContactTask(ContactCfg(contacts_dir=str(args.dataset/'contacts'), contact_geometry='spheres224'),
                      mdp, str(args.dataset/'motions'), args.device)
    seeds = quiet_frames(motion, ctc, 100, args.seed)
    ids = torch.arange(100, device=args.device)
    frames = seeds[ids % len(seeds)]
    env = TrackingSim(100, motion, mdp, args.device, eval_mode=True,
                      sim_cfg=make_sim(robot=robot, physics_substeps=16))
    env.reset_noise = True
    env.reset(ids, frames)
    env.robot.write_root_link_velocity_to_sim(torch.zeros(100, 6, device=args.device), env_ids=ids)
    env.robot.write_joint_velocity_to_sim(torch.zeros(100, 29, device=args.device), env_ids=ids)
    env.refresh()
    anchor = env.state()[0].clone()
    feet_ids = [kin.tracked_names.index(robot.contact_bodies[k][0]) for k in ('left_foot', 'right_foot')]
    feet_anchor = kin.forward(*env.state())[0][:, feet_ids].clone()
    failure_step = torch.full((100,), 1500, device=args.device)
    max_root = torch.zeros(100, device=args.device)
    max_tilt = max_root.clone()
    max_feet = max_root.clone()
    joint_sum = max_root.clone()
    counts = max_root.clone()
    action = (motion.q[frames]-mdp.default_q-env.xi[:, 1:30])/env.alpha
    for step in range(1500):
        env.f.copy_(frames)
        transition = env.step(action)
        state = env.state()
        finite = torch.stack([x.flatten(1).isfinite().all(-1) for x in state]).all(0)
        drift = (state[0][:, :2]-anchor[:, :2]).norm(dim=-1)
        tilt = torch.rad2deg(torch.acos(state[1][:, 2, 2].clamp(-1, 1)))
        feet = kin.forward(*state)[0][:, feet_ids]
        if step == 149:
            feet_anchor = feet.clone()
        foot_drift = (feet-feet_anchor).norm(dim=-1).amax(-1) if step >= 149 else torch.zeros_like(drift)
        alive = failure_step == 1500
        joint_sum += torch.where(alive, (state[4]-motion.q[frames]).square().mean(-1), 0)
        counts += alive
        max_root = torch.maximum(max_root, torch.where(alive, drift.nan_to_num(nan=1e6), 0))
        max_tilt = torch.maximum(max_tilt, torch.where(alive, tilt.nan_to_num(nan=180.), 0))
        max_feet = torch.maximum(max_feet, torch.where(alive, foot_drift.nan_to_num(nan=1e6), 0))
        bad = ~finite | transition['term'] | (drift > .5) | (tilt > 15.) | (foot_drift > .05)
        failure_step[alive & bad] = step
        if bool((failure_step < 1500).all()):
            break
    passed = failure_step == 1500
    rows = [dict(seed_clip=motion.names[int(motion.clip_id[frames[i]])],
                 seed_frame=int(frames[i]-motion.clip_start[motion.clip_id[frames[i]]]),
                 diagnostic_passed=bool(passed[i]), duration_s=float((failure_step[i]+1).clamp_max(1500)*.02),
                 root_drift_m=float(max_root[i]), tilt_deg=float(max_tilt[i]), feet_drift_m=float(max_feet[i]),
                 joint_rmse_rad=float((joint_sum[i]/counts[i].clamp_min(1)).sqrt())) for i in range(100)]
    result = dict(scope=__doc__, learned_policy=False, policy_acceptance=False,
                  robot_metadata=robot.metadata(), data_hash=motion.data_hash, seed=args.seed,
                  physics_substeps=16, unique_poses=len(seeds), scenarios=rows,
                  diagnostic_success_rate=float(passed.float().mean()), elapsed_sim_seconds=(step+1)*.02)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ('scenarios', 'robot_metadata')}))


if __name__ == '__main__':
    main()
