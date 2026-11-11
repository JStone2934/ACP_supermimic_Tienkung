"""Stage 1: the contact-target tracker trained from scratch with SuperMimic (Algorithm A.1) on the G1 box terrain.

The policy a = dec(enc(c), o_{t-9:t}) sees proprioception and the contact command c only (next-contact targets on
the terrain, the goal command and the paper's 9 x 7 local height scan; ``--blind``: no scan), while its reward and termination grade
it on the full reference and the critic sees the reference errors and the height scan. The world model sees the height
scan of every state. The reward is the contact paper's Stage 1 reward (:class:`supermimic.acp.task.PaperReward`), or
with ``--reward supermimic`` SuperMimic's motion-tracking reward plus the asymmetric contact-label term. Episode starts
follow the paper's backward curriculum unless ``--no-backward_curriculum`` (then SuperMimic's start-bin sampling).
``--init`` fine-tunes another run (its networks, optimizers and curriculum; its learning-rate schedule continues).

Usage: python -m supermimic.acp.train_stage1 --log_dir logs/acp_stage1
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import math
import os
import time
import shutil
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--robot", choices=["g1", "tienkung3_29dof"], default="g1")
parser.add_argument("--task", choices=['tracking', 'standing_aiming'], default='tracking')
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--iterations", type=int, default=60_000, help="total iterations; -1 trains continuously until stopped")
parser.add_argument("--buffer_steps", type=int, default=1000)
parser.add_argument("--utd", type=float, default=0.4, help="updates per vectorized simulator step")
parser.add_argument("--horizon", type=int, default=32)
parser.add_argument("--hist", type=int, default=10)
parser.add_argument("--wm_window", type=int, default=8)
parser.add_argument("--wm_batches", type=int, default=10)
parser.add_argument("--wm_batch", type=int, default=2048)
parser.add_argument("--pi_batch", type=int, default=4096)
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--lam", type=float, default=0.95)
parser.add_argument("--action_noise", type=float, default=0.1)
parser.add_argument("--critic_epochs", type=int, default=5)
parser.add_argument("--critic_minibatches", type=int, default=4)
parser.add_argument("--warmup_steps", type=int, default=64, help="sim steps before any update")
parser.add_argument("--wm_only_iters", type=int, default=500, help="world-model-only iterations at the start")
parser.add_argument("--lr_decay_iters", type=int, default=250_000)
parser.add_argument("--ckpt_every", type=int, default=2000)
parser.add_argument("--save_full_checkpoints", action="store_true", help="keep optimizer/critic state in numbered checkpoints for selection and resume")
parser.add_argument("--log_every", type=int, default=50)
parser.add_argument("--motion_dir", type=str, default=None, help="default: data/php_parkour")
parser.add_argument("--contacts_dir", type=str, default=None, help="supermimic.acp.make_labels output (default: <motion_dir>_contacts)")
parser.add_argument("--contact_geometry", choices=["legacy", "spheres224"], default="legacy")
parser.add_argument("--allow_contact_geometry_change", action="store_true", help="explicitly migrate a resumed checkpoint to different reward geometry; record provenance")
parser.add_argument("--target_noise", type=float, default=0.05)
parser.add_argument("--scan_noise", type=float, default=0.05, help="--no-blind: uniform noise on the command's height scan")
parser.add_argument("--target_point", type=str, default="terrain", choices=["link", "terrain"], help="contact target: the terrain point the body touches, or its link")
parser.add_argument("--target_min_steps", type=int, default=10, help="targets only from contacts of at least this many frames")
parser.add_argument("--blind", action=argparse.BooleanOptionalAction, default=False, help="command without the paper's 9 x 7 local height scan (targets and goal only)")
parser.add_argument("--reward", type=str, default="paper", choices=["paper", "supermimic"], help="the contact paper's Stage 1 reward, or SuperMimic's tracking reward + contact labels")
parser.add_argument("--w_label", type=float, default=1.0, help="--reward supermimic: weight of the contact-label term")
parser.add_argument("--backward_curriculum", action=argparse.BooleanOptionalAction, default=True, help="episode starts: the paper's backward curriculum")
parser.add_argument("--log_dir", type=str, required=True)
parser.add_argument("--seed", type=int, default=0)
load_group = parser.add_mutually_exclusive_group()
load_group.add_argument("--resume", type=str, default=None)
load_group.add_argument("--init", type=str, default=None, help="fine-tune: start from this run's last.pt")
load_group.add_argument("--finetune", type=str, default=None, help="load policy/world model; fresh critic, optimizers and schedule")
parser.add_argument("--amp", type=str, default="bf16", choices=["none", "bf16"], help="autocast dtype for MLP matmuls")
from supermimic.acp.distributed import Runtime, add_arguments
add_arguments(parser, planner=False)
args = parser.parse_args()
if args.task == 'standing_aiming':
    if args.robot != 'tienkung3_29dof':
        parser.error('standing_aiming requires --robot tienkung3_29dof')
    if args.init:
        parser.error('standing_aiming uses --finetune or --resume, not legacy --init')
    if int(os.environ.get('WORLD_SIZE', 1)) != 1:
        parser.error('standing_aiming currently supports one local GPU')
    args.reward, args.backward_curriculum = 'supermimic', False
    args.target_noise, args.scan_noise = 0., 0.
    args.contact_geometry = 'spheres224'

import torch
import torch.nn as nn

import supermimic.acp.networks as networks
from supermimic.acp.buffer import ReplayBuffer
from supermimic.acp.memory import compile_reward, reward_batch
from supermimic.acp.env import TrackingSim
from supermimic.acp.robots import get_robot, sha256
from supermimic.acp.kinematics import RobotKinematics
from supermimic.acp.mdp import INV_DIM, MAX_STEP_REWARD, MDP, OBS_DIM, normalize_xi, unflatten_state
from supermimic.acp.motion import CONTROL_DT, MotionLib
from supermimic.acp.networks import Critic, Policy, WorldModel
from supermimic.acp.task import ContactCfg, ContactTask, PaperReward, PaperRewardCfg, check_contact_geometry_resume
from supermimic.acp.standing import StandingTask, freeze_normalizers

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
networks.AMP_DTYPE = torch.bfloat16 if args.amp == "bf16" else None
rt = Runtime(args)
dev = rt.device

robot = get_robot(args.robot)
NUM_J, OBS_DIM, INV_DIM = robot.num_joints, robot.obs_dim, robot.inv_dim
motion_dir = args.motion_dir or robot.motion_dir
kin = RobotKinematics(dev, robot)
motion = MotionLib.from_dir(motion_dir, kin, dev)
mdp = MDP(kin, motion, torch.tensor(robot.default_q, device=dev))
ctc_cfg = ContactCfg(contacts_dir=args.contacts_dir, target_noise=args.target_noise, scan_noise=args.scan_noise, w_label=args.w_label,
                     target_point=args.target_point, target_min_steps=args.target_min_steps, blind=args.blind,
                     contact_geometry=args.contact_geometry)  # fmt: skip
ctc = ContactTask(ctc_cfg, mdp, motion_dir, dev)
standing = StandingTask(mdp, ctc) if args.task == 'standing_aiming' else None
task_config = standing.metadata() if standing else None
if standing:
    if motion.num_frames < standing.episode_steps + args.horizon + 1:
        raise ValueError('standing reference must exceed episode length plus imagination horizon')
    if not torch.allclose(motion.q, standing.target_q.expand_as(motion.q), atol=1e-6):
        raise ValueError('standing reference does not match held-arm pose')
reward_cfg = PaperRewardCfg() if args.reward == "paper" else None
paper_reward = PaperReward(reward_cfg, ctc) if reward_cfg is not None else None
MAX_R = reward_cfg.max_step_reward if reward_cfg is not None else MAX_STEP_REWARD + args.w_label * CONTROL_DT
if standing:
    MAX_R = standing.max_step_reward
Hh, HZ = args.hist, args.horizon

wm = WorldModel(mdp.scan_dim, robot=robot).to(dev)
pi = Policy(OBS_DIM, ctc.cmd_dim, hist=Hh, robot=robot).to(dev)  # the contact command in place of the reference goal
critic = Critic(INV_DIM + mdp.critic_extra + ctc.cmd_dim).to(dev)
opt_wm = torch.optim.Adam(wm.parameters(), lr=args.lr)
opt_pi = torch.optim.Adam(pi.parameters(), lr=args.lr, betas=(0.9, 0.95))
opt_v = torch.optim.Adam(critic.parameters(), lr=args.lr)
start_it, wall_offset, sim_steps, lr_it0 = 0, 0.0, 0, 0
wm_only_iters = 0 if args.init else args.wm_only_iters
geometry_transition = None
source_checkpoint = None
if args.resume or args.init or args.finetune:
    load_path = args.resume or args.init or args.finetune
    ck = torch.load(load_path, map_location=dev)
    robot.validate_metadata(ck.get("robot_metadata"), legacy=True)
    if ck.get('args', {}).get('hist') != args.hist:
        raise ValueError('checkpoint history length mismatch')
    if args.resume and ck.get('task_config') != task_config:
        raise ValueError('checkpoint task configuration mismatch')
    # Resume continues the same run; --init explicitly fine-tunes on a new dataset.
    if args.resume and robot.robot_id != "g1" and ck.get("data_hash") != motion.data_hash:
        raise ValueError("checkpoint dataset mismatch")
    changed = check_contact_geometry_resume(ck.get("ctc_cfg", {}), args.contact_geometry,
                                            allow_change=args.allow_contact_geometry_change or bool(args.init or args.finetune))
    geometry_transition = ck.get("geometry_transition")
    if changed:
        geometry_transition = dict(previous=ck.get("ctc_cfg", {}).get("contact_geometry", "legacy"),
                                   current=args.contact_geometry, source_checkpoint=os.path.abspath(load_path),
                                   source_sha256=sha256(load_path), source_iteration=ck["iteration"])
        rt.print(f"explicit reward geometry migration: {geometry_transition}", flush=True)
    wm.load_state_dict(ck["wm"]), pi.load_state_dict(ck["pi"])
    if not args.finetune:
        critic.load_state_dict(ck["critic"])
        opt_wm.load_state_dict(ck["opt_wm"]), opt_pi.load_state_dict(ck["opt_pi"]), opt_v.load_state_dict(ck["opt_v"])
    source_checkpoint = (dict(path=os.path.abspath(load_path), sha256=sha256(load_path), iteration=ck['iteration'])
                         if args.finetune else ck.get('source_checkpoint'))
    if args.resume:
        start_it, wall_offset, sim_steps, lr_it0 = ck["iteration"], ck["wall_time"], ck["sim_steps"], ck.get("lr_it0", 0)
    elif args.init:  # legacy --init continues its learning-rate schedule
        lr_it0 = ck["iteration"] + ck.get("lr_it0", 0)

if standing or args.finetune or (args.resume and ck.get('freeze_normalizers', False)):
    freeze_normalizers(pi, wm)


rt.register({"wm": wm, "pi": pi, "critic": critic}, {"opt_wm": opt_wm, "opt_pi": opt_pi, "opt_v": opt_v})
if args.resume:
    rt.restore(ck)


# per-component floors on the error normalization (pos, rot, lin vel, ang vel, q, qd increments)
ERR_FLOOR = torch.cat((torch.full((3,), 3e-3), torch.full((3,), 5e-3), torch.full((3,), 2e-2), torch.full((3,), 5e-2),
                       torch.full((NUM_J,), 3e-3), torch.full((NUM_J,), 5e-2))).to(dev)  # fmt: skip


def _wm_rollout(s, a, xi, f, v):
    S = unflatten_state(s[:, 0])
    preds = []
    for j in range(args.wm_window):
        S = wm.step(S, a[:, j], normalize_xi(xi[:, j]), mdp.scan(S, motion.boxes(f[:, j], v[:, j])))
        preds.append(S)
    return tuple(torch.stack([P[i] for P in preds], 1) for i in range(6))  # (B, k, ...)


def _wm_err(S_hat, s1, mask):
    std = torch.maximum(wm.out_norm.std(), ERR_FLOOR)
    e = (wm.state_error(S_hat, unflatten_state(s1)) / std) ** 2  # (B, k, D)
    e = e.mean(-1)
    return (e * mask).sum() / s1.shape[0], e[:, -1]


def _imag_step(S, o_hist, f, v, gs, xi, xi_n, boxes):
    a = pi(o_hist, ctc.command(S, f, v, gs, noise=standing is None))
    if standing:
        a = standing.compose(a, xi)
    S1 = wm.step(S, a, xi_n, mdp.scan(S, boxes))
    o1 = mdp.obs(S1, a, xi)
    if standing is None:
        o1 = mdp.add_noise(o1)
    return S1, a, torch.cat((o_hist[:, 1:], o1.unsqueeze(1)), 1), f + 1


def _reward_term(S, S_prev, f, v, a, a_prev, a_prev2, xi):
    if standing:
        r = standing.reward(S, f, v, a, a_prev)
    elif paper_reward is not None:
        r = paper_reward(S, S_prev, f, v, a, a_prev, a_prev2, xi)
    else:
        r = mdp.reward(S, f, a, a_prev) + ctc.cfg.w_label * ctc.label_reward(S, f, v, motion.boxes(f, v)) * CONTROL_DT
    return r, mdp.soft_termination(S, f)


wm_rollout = torch.compile(_wm_rollout)
wm_err = torch.compile(_wm_err)
imag_step = torch.compile(_imag_step)
# max_fusion_size 32: Triton 3.5.0 fails to compile the fully fused backward kernel of this reward
reward_term = compile_reward(_reward_term, recompute=bool(args.reward_chunk_size))
critic_input_c = torch.compile(ctc.critic_input)


def lr_at(it: int) -> float:
    it += lr_it0
    if it >= args.lr_decay_iters:
        return args.lr * 0.03
    return args.lr * (0.03 + 0.97 * 0.5 * (1 + math.cos(math.pi * it / args.lr_decay_iters)))


def set_requires_grad(m: nn.Module, flag: bool):
    for p in m.parameters():
        p.requires_grad_(flag)


# ------------------------------------------------------------------------------------------------
def wm_update():
    w = buffer.sample_windows(args.wm_batch, args.wm_window)
    S_hat = wm_rollout(w["s"], w["a"], w["xi"], w["f"], w["v"])
    loss, e_last = wm_err(S_hat, w["s1"], w["mask"])
    opt_wm.zero_grad(set_to_none=True)
    loss.backward()
    rt.step(opt_wm, wm, "wm", max_norm=1.0)
    m = w["mask"][:, -1]
    return loss.detach() / args.wm_window, rt.ratio((e_last.detach() * m).sum(), m.sum())


def sample_starts(B: int):
    """Start points with H reference frames left in their clip, the action before their previous one (``a_prev2``)
    and the observations the policy saw before them."""
    starts, hists, n = [], [], 0
    attempts = 0
    while n < B:
        attempts += 1
        if attempts > 100:
            raise RuntimeError("no eligible imagination starts; check horizon, clip length and warmup")
        s, h = buffer.sample_starts(2 * B, Hh)
        ok = s["f"] + HZ <= motion.clip_end[motion.clip_id[s["f"]]]
        idx = ok.nonzero(as_tuple=False).squeeze(-1)[: B - n]
        starts.append({k: v[idx] for k, v in s.items()}), hists.append({k: v[idx] for k, v in h.items()})
        n += len(idx)
    cat = lambda ds: {k: torch.cat([d[k] for d in ds]) for k in ds[0]}  # noqa: E731
    s, h = cat(starts), cat(hists)
    s["a_prev2"] = h["a_prev"][:, -2]  # the history stays inside the episode: zeros right after a reset
    return s, h["o"]


def policy_update():
    s, o_hist = sample_starts(args.pi_batch)
    S = unflatten_state(s["s"])
    xi, f, v, a_prev, a_prev2, gs = s["xi"], s["f"], s["v"], s["a_prev"], s["a_prev2"], s["goal_steps"]
    xi_n = normalize_xi(xi)
    boxes = motion.boxes(f, v)  # the rollout stays inside its clip
    set_requires_grad(wm, False)
    set_requires_grad(critic, False)
    B = args.pi_batch
    S0, f0 = S, f
    Ss, Sps, As, Aps, Ap2s, F1s = [], [], [], [], [], []
    for t in range(HZ):
        Sps.append(S), Aps.append(a_prev), Ap2s.append(a_prev2)
        a_prev2 = a_prev
        S, a_prev, o_hist, f = imag_step(S, o_hist, f, v, gs, xi, xi_n, boxes)
        Ss.append(S), As.append(a_prev), F1s.append(f)
    flat = lambda xs: torch.stack(xs).flatten(0, 1)  # noqa: E731  (H*B, ...)
    S_all = tuple(flat([St[i] for St in Ss]) for i in range(6))
    S_prev = tuple(flat([St[i] for St in Sps]) for i in range(6))
    xi_all = xi.unsqueeze(0).expand(HZ, -1, -1).flatten(0, 1)
    v_all = v.repeat(HZ)
    r, d = reward_batch(reward_term, S_all, S_prev, flat(F1s), v_all, flat(As), flat(Aps), flat(Ap2s), xi_all, chunk_size=args.reward_chunk_size)
    r, d = r.view(HZ, B), d.view(HZ, B)
    alive = torch.cumprod(1 - d, 0)  # c_{t+1}
    c_t = torch.cat((torch.ones(1, B, device=dev), alive[:-1]), 0)
    disc = args.gamma ** torch.arange(HZ, device=dev, dtype=torch.float32).unsqueeze(-1)
    Jr = (disc * c_t * r).sum(0)
    c = alive[-1]
    r_sum = (c_t * r).detach().sum(0)
    rews, dones = r.detach(), d.detach()
    Jr = Jr + (args.gamma**HZ) * c * critic(ctc.critic_input(S, f, v, gs, xi_n))
    loss = -Jr.mean()
    opt_pi.zero_grad(set_to_none=True)
    loss.backward()
    gn = rt.step(opt_pi, pi, "pi", max_norm=1.0, clip_value=0.1)
    set_requires_grad(wm, True)
    set_requires_grad(critic, True)
    # ---------------- critic: GAE(lambda) targets over the imagined transitions ----------------
    with torch.no_grad():
        S_c = tuple(torch.cat((S0[i].unsqueeze(0), torch.stack([St[i].detach() for St in Ss])), 0).flatten(0, 1) for i in range(6))
        f_c = torch.cat((f0.unsqueeze(0), torch.stack(F1s)), 0).flatten()
        v_c, gs_c = v.repeat(HZ + 1), gs.repeat(HZ + 1)
        xi_c = xi_n.unsqueeze(0).expand(HZ + 1, -1, -1).flatten(0, 1)
        X = critic_input_c(S_c, f_c, v_c, gs_c, xi_c).view(HZ + 1, B, -1)
    critic.norm.update(X)
    with torch.no_grad():
        V = critic(X)
        R = V[-1]
        targets = []
        for t in range(HZ - 1, -1, -1):
            R = rews[t] + args.gamma * (1 - dones[t]) * ((1 - args.lam) * V[t + 1] + args.lam * R)
            targets.append(R)
        targets = torch.stack(targets[::-1])  # (H, B)
    Xf, Yf = X[:-1].reshape(-1, X.shape[-1]), targets.reshape(-1)
    n = Xf.shape[0]
    mb = n // args.critic_minibatches
    for _ in range(args.critic_epochs):
        perm = torch.randperm(n, device=dev)
        for i in range(args.critic_minibatches):
            idx = perm[i * mb : (i + 1) * mb]
            v_loss = ((critic(Xf[idx]) - Yf[idx]) ** 2).mean()
            opt_v.zero_grad(set_to_none=True)
            v_loss.backward()
            rt.step(opt_v, critic, "critic", max_norm=1.0)
    imag_r = (r_sum / HZ / MAX_R).mean()
    return loss.detach(), v_loss.detach(), imag_r, gn.detach(), c.mean().detach()


# ------------------------------------------------------------------------------------------------
env = TrackingSim(args.num_envs, motion, mdp, dev, gamma=args.gamma, backward_curriculum=args.backward_curriculum, runtime=rt,
                  control_task=standing)
if (args.resume or args.init) and "curriculum_phase" in ck:
    env.phase.copy_(ck["curriculum_phase"])
if args.resume:
    for key in ('bin_G', 'bin_w_avg'):
        if key in ck:
            getattr(env, key).copy_(ck[key])
if args.resume or args.init or args.finetune:
    env.reset(torch.arange(args.num_envs, device=dev))
if standing:
    from supermimic.acp.standing_eval import evaluate, selection_key
    eval_env = TrackingSim(1, motion, mdp, dev, eval_mode=True, control_task=standing)
    best_report_path = Path(args.log_dir) / 'best_eval.json'
    import json
    best_report = json.loads(best_report_path.read_text()) if args.resume and best_report_path.exists() else None
    rt.write_json('task_config.json', dict(task_config=task_config, source_checkpoint=source_checkpoint))
# clip groups for the curriculum log: the motion folders
GROUPS = sorted({n.split("/")[0] for n in motion.names})
clip_group = torch.tensor([GROUPS.index(n.split("/")[0]) for n in motion.names], device=dev)
buffer = ReplayBuffer(args.buffer_steps, args.num_envs, dev, robot=robot)
obs_hist = torch.zeros(args.num_envs, Hh, OBS_DIM, device=dev)
goal_steps = ctc.sample_goal_steps(args.num_envs)
collect_iteration = start_it


@torch.no_grad()
def collect(n_steps: int, stats: dict):
    global obs_hist, sim_steps
    for _ in range(n_steps):
        o = env.observe()
        fresh = env.ep_step == 0
        obs_hist = torch.where(fresh[:, None, None], o.unsqueeze(1).expand_as(obs_hist), torch.cat((obs_hist[:, 1:], o.unsqueeze(1)), 1))
        cmd = ctc.command(env.state(), env.f, env.v, goal_steps, noise=standing is None)
        pi.obs_norm.update(o)
        pi.goal_norm.update(cmd)
        a = pi(obs_hist, cmd)
        noise = standing.noise_scale(collect_iteration, args.action_noise) if standing else args.action_noise
        eps = torch.randn_like(a)
        if standing:
            eps = eps * standing.free
        a = torch.clamp(a + noise * eps, -20, 20)
        if standing:
            a = standing.compose(a, env.xi)
        out = env.step(a)
        out["goal_steps"] = goal_steps.clone()
        buffer.add(out)
        S0, S1 = unflatten_state(out["s"]), unflatten_state(out["s1"])
        wm.in_norm.update(wm.inputs(S0, out["a"], normalize_xi(out["xi"]), mdp.scan(S0, motion.boxes(out["f"], out["v"]))))
        wm.out_norm.update(wm.targets(S0, S1))
        sim_steps += rt.args["num_envs"]
        # contact labels in the simulator: touch per body vs the reference label
        f1 = out["f"] + 1
        touch = ctc.body_distance(S1, motion.boxes(f1, out["v"])) <= ctc.cfg.touch_dist
        label = ctc.labels[f1, out["v"]]
        stats["tp"].append((touch & label).sum()), stats["n_touch"].append(touch.sum()), stats["n_label"].append(label.sum())
        stats["r"].append(out["r"].mean() / (MAX_R if standing else MAX_STEP_REWARD))
        if "ep_len" in out:
            stats["len"].append(out["ep_len"].sum()), stats["n_ep"].append(out["ep_len"].new_tensor(len(out["ep_len"])))
            stats["ret"].append(out["ep_ret_norm"].sum()), stats["term"].append(out["term"][out["done"]].float().sum())
        done = out["done"]
        goal_steps[done] = ctc.sample_goal_steps(int(done.sum()))


stats_keys = ("r", "len", "ret", "n_ep", "term", "tp", "n_touch", "n_label")
stats = {k: [] for k in stats_keys}
collect(args.warmup_steps, stats)  # the replay buffer is not checkpointed, so a resumed run refills it too
log_f = rt.open_log(os.path.join(args.log_dir, "train_log.csv"))
if rt.main and os.path.getsize(os.path.join(args.log_dir, "train_log.csv")) == 0:
    log_f.write("iteration,wall_h,sim_steps,wm_loss,wm_err8,pi_loss,v_loss,imag_rew,pi_gradnorm,imag_survival,"
                "sim_rew,ep_len,ep_ret_norm,term_frac,touch_precision,touch_recall,it_per_s,lr,phase_zero_frac"
                + "".join(f",phase_{g}" for g in GROUPS) + "\n")  # fmt: skip
t_start = time.time() - wall_offset
t_log = time.time()
step_acc = ck.get("step_acc", 0.0) if args.resume else 0.0
agg = {"wm": [], "wm8": [], "pi": [], "v": [], "ir": [], "gn": [], "surv": []}
for it in (itertools.count(start_it) if args.iterations == -1 else range(start_it, args.iterations)):
    collect_iteration = it
    lr = lr_at(it)
    for o in (opt_wm, opt_pi, opt_v):
        for gparam in o.param_groups:
            gparam["lr"] = lr
    step_acc += 1.0 / args.utd
    n = int(step_acc)
    step_acc -= n
    collect(n, stats)
    for _ in range(args.wm_batches):
        l, e8 = wm_update()
    agg["wm"].append(l), agg["wm8"].append(e8)
    if it >= wm_only_iters:
        for k, v in zip(("pi", "v", "ir", "gn", "surv"), policy_update()):
            agg[k].append(v)
    rt.tick(it + 1, sim_steps, ready=rt.updates.get("pi", 0) > 0)
    if (it + 1) % args.log_every == 0:
        stats = rt.aggregate(stats)
        agg = rt.aggregate(agg)
        mean = lambda x: float(sum(float(v) for v in x) / max(len(x), 1))  # noqa: E731
        total = lambda k: float(sum(float(v) for v in stats[k]))  # noqa: E731
        ratio = lambda a, b: total(a) / max(total(b), 1)  # noqa: E731
        ips = args.log_every / (time.time() - t_log)
        t_log = time.time()
        wall_h = (time.time() - t_start) / 3600
        row = [it + 1, wall_h, sim_steps, mean(agg["wm"]), mean(agg["wm8"]), mean(agg["pi"]), mean(agg["v"]), mean(agg["ir"]),
               mean(agg["gn"]), mean(agg["surv"]), mean(stats["r"]), ratio("len", "n_ep"), ratio("ret", "n_ep"),
               ratio("term", "n_ep"), ratio("tp", "n_touch"), ratio("tp", "n_label"), ips, lr]  # fmt: skip
        # curriculum: fraction of clips started from their first frame, mean start phase per motion folder
        phase_g = [float(env.phase[clip_group == i].mean()) for i in range(len(GROUPS))]
        row += [float((env.phase == 0).float().mean())] + phase_g
        log_f.write(",".join(f"{x:.6g}" if isinstance(x, float) else str(x) for x in row) + "\n")
        log_f.flush()
        rt.print(f"it {it + 1} | {wall_h:.3f} h | wm {row[3]:.4f} e8 {row[4]:.4f} | pi {row[5]:.3f} v {row[6]:.4f} imag_r {row[7]:.3f} "
              f"surv {row[9]:.3f} | sim_r {row[10]:.3f} ep_len {row[11]:.1f} ep_ret {row[12]:.3f} term {row[13]:.2f} "
              f"touch P/R {row[14]:.3f}/{row[15]:.3f} | phase0 {row[18]:.2f}"
              + "".join(f" {g.removeprefix('motions-')} {p:.2f}" for g, p in zip(GROUPS, phase_g)) + f" | {ips:.2f} it/s", flush=True)  # fmt: skip
        stats = {k: [] for k in stats_keys}
        agg = {k: [] for k in agg}
    if (it + 1) % args.ckpt_every == 0 or it + 1 == args.iterations:
        ck = {"robot_metadata": robot.metadata(), "data_hash": motion.data_hash, "wm": wm.state_dict(), "pi": pi.state_dict(), "critic": critic.state_dict(), "opt_wm": opt_wm.state_dict(),
              "opt_pi": opt_pi.state_dict(), "opt_v": opt_v.state_dict(), "iteration": it + 1,
              "wall_time": time.time() - t_start, "sim_steps": sim_steps, "step_acc": step_acc, "args": rt.args,
              "ctc_cfg": dataclasses.asdict(ctc_cfg), "geometry_transition": geometry_transition,
              "curriculum_phase": env.phase.clone(), "lr_it0": lr_it0,
              "bin_G": env.bin_G.clone(), "bin_w_avg": env.bin_w_avg.clone(),
              "reward_cfg": dataclasses.asdict(reward_cfg) if reward_cfg is not None else None,
              "task_config": task_config, "source_checkpoint": source_checkpoint,
              "freeze_normalizers": bool(standing or args.finetune or (args.resume and ck.get('freeze_normalizers', False))),
              "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}  # fmt: skip
        small = {k: ck[k] for k in ("robot_metadata", "data_hash", "pi", "wm", "iteration", "wall_time", "sim_steps", "args", "ctc_cfg", "geometry_transition", "saved_at", "task_config", "source_checkpoint", "freeze_normalizers")}
        rt.save(ck, numbered=f"ckpt_{it + 1:06d}.pt", small_keys=list(small), full=args.save_full_checkpoints)
        if standing:
            output = Path(args.log_dir) / 'evaluations' / f'iter_{it + 1:06d}'
            # Evaluation neither advances the collector nor consumes its random stream.
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                report = evaluate(pi, eval_env, standing, output, iteration=it + 1,
                                  checkpoint=str(Path(args.log_dir) / f'ckpt_{it + 1:06d}.pt'))
            rt.print('standing eval: ' + json.dumps({k: v for k, v in report.items() if k not in ('metrics', 'task_config')}), flush=True)
            if best_report is None or report['standing_30s_pass'] or selection_key(report) < selection_key(best_report):
                best_report = report
                rt.write_json('best_eval.json', report)
                shutil.copyfile(Path(args.log_dir) / 'last.pt', Path(args.log_dir) / 'best.pt')
            if report['standing_30s_pass']:
                saved = torch.load(Path(args.log_dir) / 'best.pt', map_location='cpu')
                pi.load_state_dict(saved['pi'])
                with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                    confirmation = evaluate(pi, eval_env, standing, Path(args.log_dir) / 'confirmed', iteration=it + 1,
                                            checkpoint=str(Path(args.log_dir) / 'best.pt'))
                if confirmation['standing_30s_pass']:
                    rt.write_json('completion.json', dict(status='passed', iteration=it + 1, report='confirmed.json'))
                    break
    if standing and it + 1 == args.iterations:
        rt.write_json('completion.json', dict(status='not_passed', iteration=it + 1, report='best_eval.json'))

rt.close()
