"""Validate ACP assets, CUDA compilation, contact targets, and GPU simulation.

Run from the repository root: python -m tools.check_acp_setup
The first run compiles Warp and PyTorch kernels and can take a few minutes.
"""

import argparse
import importlib.metadata
import json

import torch

from supermimic.acp.env import TrackingSim
from supermimic.acp.g1 import MOTION_DIR, NUM_J, default_joint_pos_list
from supermimic.acp.kinematics import G1Kinematics
from supermimic.acp.mdp import MDP
from supermimic.acp.motion import MotionLib
from supermimic.acp.task import ContactCfg, ContactTask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=16)
    args = parser.parse_args()
    assert args.num_envs > 0 and args.steps > 0
    assert torch.cuda.is_available(), "PyTorch cannot access CUDA"
    torch.set_num_threads(8)
    torch.manual_seed(0)
    device = args.device

    # Exercise the compiler and backward pass used by training.
    print("Checking torch.compile and CUDA backward...", flush=True)
    compiled = torch.compile(lambda x: x.sin().square().sum())
    x = torch.randn(128, device=device, requires_grad=True)
    compiled(x).backward()
    torch.testing.assert_close(x.grad, 2 * x.detach().sin() * x.detach().cos())

    print("Loading all PHP motions and contact labels...", flush=True)
    kin = G1Kinematics(device)
    motion = MotionLib.from_dir(MOTION_DIR, kin, device)
    assert len(motion.files) == 296, f"Expected 296 PHP clips, found {len(motion.files)}"
    mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=device))
    contacts = ContactTask(ContactCfg(), mdp, MOTION_DIR, device)
    print("Building MuJoCo Warp simulation (first run compiles kernels)...", flush=True)
    env = TrackingSim(args.num_envs, motion, mdp, device, reset_noise=False)
    goal_steps = contacts.sample_goal_steps(args.num_envs)
    with torch.no_grad():
        for _ in range(args.steps):
            obs = env.observe()
            cmd = contacts.command(env.state(), env.f, env.v, goal_steps, noise=False)
            assert torch.isfinite(obs).all() and torch.isfinite(cmd).all()
            out = env.step(torch.zeros(args.num_envs, NUM_J, device=device))
            for name in ("s", "s1", "r"):
                assert torch.isfinite(out[name]).all(), f"Non-finite simulation {name}"
    torch.cuda.synchronize(device)
    print(json.dumps({
        "status": "PASS",
        "versions": {name: importlib.metadata.version(name) for name in
                     ("torch", "mjlab", "mujoco", "mujoco-warp", "warp-lang")},
        "gpu": torch.cuda.get_device_name(device),
        "cuda": torch.version.cuda,
        "clips": len(motion.files),
        "frames": motion.num_frames,
        "num_envs": args.num_envs,
        "control_steps": args.steps,
        "contact_command_dim": contacts.cmd_dim,
        "torch_compile_backward": "PASS",
    }, indent=2))


if __name__ == "__main__":
    main()
