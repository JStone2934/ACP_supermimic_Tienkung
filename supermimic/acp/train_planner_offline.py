"""Stage 2, offline: the contact planner (:mod:`supermimic.acp.planner`) learned from a frozen Stage 1 tracker.

The tracker (a supermimic.acp.train_stage1 checkpoint) runs in the simulator with its training randomization, action
noise and reference contact targets, from uniform start phases (the paper's Stage 2 episode starts). The replay buffer
of simulator transitions records the episodes; finished ones are labeled in hindsight
(:func:`supermimic.acp.planner.label_episodes`) into the planner's buffer, and the planner trains on it with the paper's
Table S5 settings. The planner sees the exact height scan of the episode's terrain variant (plus noise). Every 10th source motion
of each motion folder is held out, including all materialized terrain variants: its samples only go to a validation buffer, on which ``planner_val.csv`` reports
per-body occupancy precision / recall / F1, the median top-1 error and the share of top-1 points within 0.1 m.

    python -m supermimic.acp.train_planner_offline --tracker logs/acp_stage1/ckpt_060000.pt --log_dir logs/acp_planner
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import time

parser = argparse.ArgumentParser()
parser.add_argument("--robot", choices=["g1", "tienkung3_29dof"], default=None)
parser.add_argument("--tracker", type=str, default=None, help="supermimic.acp.train_stage1 checkpoint")
parser.add_argument("--num_envs", type=int, default=None, help="global environments; default: tracker/resume checkpoint")
parser.add_argument("--steps", type=int, default=20_000, help="simulator steps")
parser.add_argument("--updates_per_step", type=int, default=1, help="planner updates per simulator step")
parser.add_argument("--buffer_steps", type=int, default=1350, help="replay rows per environment: at least the longest clip")
parser.add_argument("--capacity", type=int, default=1_000_000, help="planner replay samples (validation: a fifth)")
parser.add_argument("--action_noise", type=float, default=0.1)
parser.add_argument("--val_samples", type=int, default=16384)
parser.add_argument("--goal_radius", type=float, default=0.3, help="planner goal d_xy / max(|d_xy|, r)")
parser.add_argument("--scan_noise", type=float, default=0.03, help="Gaussian noise (m) on the planner's height scan")
parser.add_argument("--start_max", type=float, default=0.1, help="success filter: episodes started within this fraction of their clip")
parser.add_argument("--reach_min", type=float, default=0.9, help="success filter: and reached at least this fraction")
parser.add_argument("--log_every", type=int, default=100)
parser.add_argument("--val_every", type=int, default=500)
parser.add_argument("--ckpt_every", type=int, default=2000)
parser.add_argument("--log_dir", type=str, required=True)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--resume", default=None)
from supermimic.acp.distributed import Runtime, add_arguments
add_arguments(parser, planner=True, reward=False)
args = parser.parse_args()
if not (args.tracker or args.resume):
    parser.error("--tracker or --resume is required")

import torch

import supermimic.acp.networks as networks
from supermimic.acp.buffer import ReplayBuffer
from supermimic.acp.env import TrackingSim
from supermimic.acp.robots import resolve_robot, sha256
from supermimic.acp.kinematics import RobotKinematics
from supermimic.acp.mdp import MDP, OBS_DIM, REF_ORI_DIMS
from supermimic.acp.motion import MotionLib
from supermimic.acp.networks import Policy
from supermimic.acp.planner import (ContactPlanner, PlannerBuffer, PlannerCfg, PlannerScan, label_episodes,
                                    distributed_planner_metrics, planner_update)
from supermimic.acp.task import ContactCfg, ContactTask
from supermimic.acp.stage2_contracts import inherit_execution_sizes, validation_split

torch.backends.cuda.matmul.allow_tf32 = True
resume_ck = torch.load(args.resume, map_location='cpu', weights_only=False) if args.resume else None
if resume_ck is not None:
    args.tracker = args.tracker or resume_ck['tracker']
    if resume_ck.get('tracker_hash') and sha256(args.tracker) != resume_ck['tracker_hash']:
        raise ValueError('resume tracker checkpoint changed')
tracker_hash = sha256(args.tracker)

ck = torch.load(args.tracker, map_location='cpu', weights_only=False)
inherit_execution_sizes(args, resume_ck if resume_ck is not None else ck)
rt = Runtime(args)
dev = rt.device
robot = resolve_robot(args.robot, ck)
NUM_J, OBS_DIM, INV_DIM, XI_DIM = robot.num_joints, robot.obs_dim, robot.inv_dim, robot.xi_dim
CONTACT_BODIES, NB = robot.contact_bodies, len(robot.contact_bodies)
ta = ck["args"]
motion_dir = ta.get("motion_dir") or robot.motion_dir
kin = RobotKinematics(dev, robot)
motion = MotionLib.from_dir(motion_dir, kin, dev)
if robot.robot_id != "g1" and ck.get("data_hash") != motion.data_hash:
    raise ValueError("checkpoint dataset mismatch")
mdp = MDP(kin, motion, torch.tensor(robot.default_q, device=dev))
ctc = ContactTask(ContactCfg(**ck["ctc_cfg"]), mdp, motion_dir, dev)
Hh = ta["hist"]
networks.AMP_DTYPE = torch.bfloat16 if ta.get("amp", "none") == "bf16" else None
pi = Policy(OBS_DIM, ctc.cmd_dim, hist=Hh, robot=robot).to(dev)
pi.load_state_dict(ck["pi"])
pi.eval()

pcfg = PlannerCfg(capacity=args.capacity, goal_radius=args.goal_radius, scan_noise=args.scan_noise, start_max=args.start_max,
                 reach_min=args.reach_min)
if resume_ck is not None:
    pcfg = PlannerCfg(**{**resume_ck['planner_cfg'], 'capacity': args.capacity})
pcfg = rt.planner_config(pcfg)
scan = PlannerScan(pcfg, dev)
planner = ContactPlanner(OBS_DIM - REF_ORI_DIMS, pcfg).to(dev)
opt = torch.optim.Adam(planner.parameters(), lr=pcfg.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
start_step, wall_offset, sim_steps = 0, 0., 0
if resume_ck is not None:
    robot.validate_metadata(resume_ck.get('robot_metadata'), legacy=True)
    if (robot.robot_id != 'g1' or resume_ck.get('data_hash') is not None) and resume_ck.get('data_hash') != motion.data_hash:
        raise ValueError('checkpoint dataset mismatch')
    planner.load_state_dict(resume_ck['planner'])
    opt.load_state_dict(resume_ck['opt'])
    start_step = resume_ck['step']
    wall_offset = resume_ck.get('wall_time', 0.)
    sim_steps = resume_ck.get('sim_steps', start_step * rt.args['num_envs'])
    rt.restore(resume_ck)
rt.register({'pi': pi, 'planner': planner}, {'opt_planner': opt}, deferred=(planner.prop_norm,))
pbuf = PlannerBuffer(rt.capacity(pcfg.capacity), pcfg.hist, planner.prop_dim, scan.num_z, dev)
vbuf = PlannerBuffer(rt.capacity(pcfg.capacity // 5), pcfg.hist, planner.prop_dim, scan.num_z, dev)
val_mask, split_record = validation_split(motion.names, motion_dir,
    saved=resume_ck.get('validation_split') if resume_ck is not None else None, continuing=resume_ck is not None)
rt.write_json('validation_split.json', split_record)
val_clip = torch.tensor(val_mask, dtype=torch.bool, device=dev)
assert args.buffer_steps > int(motion.clip_len.max()), "the replay buffer must hold a whole episode"

N = args.num_envs
env = TrackingSim(N, motion, mdp, dev, start_phase=(0.0, 0.98), runtime=rt)
buffer = ReplayBuffer(args.buffer_steps, N, dev, robot=robot)
obs_hist = torch.zeros(N, Hh, OBS_DIM, device=dev)
goal_steps = ctc.sample_goal_steps(N)

BODIES = list(CONTACT_BODIES)
new_log = not os.path.exists(os.path.join(args.log_dir, "train_log.csv"))
log_f = rt.open_log(os.path.join(args.log_dir, "train_log.csv"))
if new_log: log_f.write("step,wall_h,steps_per_s,ended,admitted,admit_rate,labeled,buffer,val_buffer,aff,occ\n")
val_f = rt.open_log(os.path.join(args.log_dir, "planner_val.csv"))
if new_log: val_f.write("step,aff,occ," + ",".join(f"{m}_{b}" for m in ("precision", "recall", "f1", "median_err", "within_0.1") for b in BODIES) + "\n")
t_log = time.time()
t0 = t_log - wall_offset
acc = {"ended": 0, "admitted": 0, "labeled": 0, "loss": []}


def save(step: int):
    ck = dict(robot_metadata=robot.metadata(), data_hash=motion.data_hash, planner=planner.state_dict(), opt=opt.state_dict(),
              planner_cfg=dataclasses.asdict(pcfg), step=step, tracker=args.tracker, tracker_hash=tracker_hash,
              ctc_cfg=dataclasses.asdict(ctc.cfg), validation_split=split_record,
              prop_dim=planner.prop_dim, args=rt.args, wall_time=time.time()-t0, sim_steps=sim_steps)
    rt.save(ck, numbered=f'planner_{step:06d}.pt', full=True)


for step in range(start_step + 1, args.steps + 1):
    with torch.no_grad():
        o = env.observe()
        fresh = env.ep_step == 0
        obs_hist = torch.where(fresh[:, None, None], o.unsqueeze(1).expand_as(obs_hist), torch.cat((obs_hist[:, 1:], o.unsqueeze(1)), 1))
        a = pi(obs_hist, ctc.command(env.state(), env.f, env.v, goal_steps, noise=True))
        out = env.step(torch.clamp(a + args.action_noise * torch.randn_like(a), -20, 20))
        out["goal_steps"] = goal_steps.clone()
        buffer.add(out)
        sim_steps += rt.args["num_envs"]
        done = out["done"]
        envs = done.nonzero(as_tuple=False).squeeze(-1)
        if len(envs) > 0:
            row = torch.full_like(envs, (buffer.ptr - 1) % buffer.T)
            d, n_adm = label_episodes(buffer, envs, row, out["ep_step"][envs] + 1, ctc, scan, pcfg)
            acc["ended"] += len(envs)
            acc["admitted"] += n_adm
            if d is not None:
                val = val_clip[d["clip"]]
                acc["labeled"] += int((~val).sum())
                if (~val).any():  # (the normalizer turns NaN on an empty batch)
                    planner.prop_norm.update(d["prop"][~val].float())
                pbuf.add({k: v[~val] for k, v in d.items()})
                vbuf.add({k: v[val] for k, v in d.items()})
            goal_steps[done] = ctc.sample_goal_steps(len(envs))
    rt.flush_norm(planner.prop_norm)
    if rt.global_size(pbuf) >= pcfg.min_entries:
        for _ in range(args.updates_per_step):
            acc["loss"].append(planner_update(planner, opt, pbuf, scan, pcfg, rt))
    rt.tick(step, sim_steps, ready=rt.updates.get('planner', 0) > 0)
    if step % args.log_every == 0:
        merged = rt.aggregate({k: [v] if k != 'loss' else v for k, v in acc.items()})
        acc = {k: sum(v) if k != 'loss' else v for k, v in merged.items()}
        sps = args.log_every / (time.time() - t_log)
        t_log = time.time()
        loss = torch.stack(acc["loss"]).mean(0).tolist() if acc["loss"] else [float("nan")] * 2
        row = [step, (time.time() - t0) / 3600, sps, acc["ended"], acc["admitted"], acc["admitted"] / max(acc["ended"], 1), acc["labeled"],
               rt.global_size(pbuf), rt.global_size(vbuf), *loss]  # fmt: skip
        log_f.write(",".join(f"{x:.6g}" if isinstance(x, float) else str(x) for x in row) + "\n")
        log_f.flush()
        rt.print(f"step {step} | {row[1]:.3f} h | {sps:.1f} steps/s | ended {acc['ended']} admitted {acc['admitted']} ({row[5]:.3f}) "
              f"labeled {acc['labeled']} | buffer {row[7]} val {row[8]} | aff {loss[0]:.3f} occ {loss[1]:.3f}", flush=True)  # fmt: skip
        acc = {"ended": 0, "admitted": 0, "labeled": 0, "loss": []}
    if step % args.val_every == 0 and rt.global_size(vbuf) > 0:
        planner.eval()
        mt = distributed_planner_metrics(planner, vbuf, scan, pcfg, rt, args.val_samples)
        planner.train()
        keys = ("precision", "recall", "f1", "median_err", "within_0.1")
        val_f.write(",".join(str(x) for x in [step, mt["aff"], mt["occ"]] + [v for k in keys for v in mt[k]]) + "\n")
        val_f.flush()
        rt.print("  val " + " | ".join(f"{b} F1 {mt['f1'][i]:.2f} err {mt['median_err'][i]:.3f} <0.1 {mt['within_0.1'][i]:.2f}"
                                    for i, b in enumerate(BODIES[:6])), flush=True)  # fmt: skip
    if step % args.ckpt_every == 0 or step == args.steps:
        save(step)

rt.close()
