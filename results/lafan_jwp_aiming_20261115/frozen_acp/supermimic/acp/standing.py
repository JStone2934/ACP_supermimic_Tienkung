"""Fixed right-arm standing task; shared by physical and imagined control."""
from dataclasses import asdict, dataclass
import math

import torch

from .mdp import MAX_STEP_REWARD

ARM_NAMES = ('shoulder_pitch_r_joint', 'shoulder_roll_r_joint', 'shoulder_yaw_r_joint',
             'elbow_pitch_r_joint', 'elbow_yaw_r_joint', 'wrist_pitch_r_joint', 'wrist_roll_r_joint')
ARM_Q = (0.2, -0.1, 0.0, -math.pi / 2, 0.0, 0.0, 0.0)


@dataclass(frozen=True)
class StandingCfg:
    name: str = 'standing_aiming'
    version: int = 1
    episode_seconds: float = 30.0
    noise_decay_iters: int = 3000
    root_weight: float = 2.0
    foot_weight: float = 2.0
    velocity_weight: float = 1.0
    foot_velocity_weight: float = 0.5
    heading_weight: float = 0.5
    root_scale: float = 0.10
    foot_scale: float = 0.05
    velocity_scale: float = 0.10
    foot_velocity_scale: float = 0.10
    heading_scale: float = math.radians(5)
    max_root_drift: float = 0.10
    max_foot_drift: float = 0.05
    max_tilt_deg: float = 15.0
    max_arm_p95_deg: float = 5.0


def held_pose(robot):
    if robot.robot_id != 'tienkung3_29dof':
        raise ValueError('standing_aiming requires tienkung3_29dof')
    q = list(robot.default_q)
    for name, angle in zip(ARM_NAMES, ARM_Q):
        q[robot.joint_names.index(name)] = angle
    return q


class StandingTask:
    def __init__(self, mdp, ctc, cfg=StandingCfg()):
        self.mdp, self.ctc, self.cfg = mdp, ctc, cfg
        robot, dev = mdp.robot, mdp.default_q.device
        self.target_q = torch.tensor(held_pose(robot), device=dev)
        self.arm_ids = torch.tensor([robot.joint_names.index(n) for n in ARM_NAMES], device=dev)
        self.free = torch.ones(robot.num_joints, dtype=torch.bool, device=dev)
        self.free[self.arm_ids] = False
        self.alpha = torch.tensor(robot.action_scales, device=dev)
        self.feet = [mdp.kin.tracked_names.index(robot.contact_bodies[n][0]) for n in ('left_foot', 'right_foot')]
        self.max_step_reward = MAX_STEP_REWARD + robot.control_dt * (
            ctc.cfg.w_label + cfg.root_weight + cfg.foot_weight + cfg.velocity_weight
            + cfg.foot_velocity_weight + cfg.heading_weight)
        self.episode_steps = round(cfg.episode_seconds / robot.control_dt)

    def metadata(self):
        return dict(config=asdict(self.cfg), arm_names=list(ARM_NAMES), arm_q=list(ARM_Q),
                    freeze_normalizers=True, nominal_physics=True, start_frame='first', observation_noise=False)

    def compose(self, action, xi):
        held = (self.target_q - self.mdp.default_q - xi[..., 1:1 + self.mdp.num_joints]) / self.alpha
        return torch.where(self.free, action, held)

    def noise_scale(self, iteration, initial):
        return initial * max(0.0, 1.0 - iteration / self.cfg.noise_decay_iters)

    def terms(self, S, f):
        """Position references stay in world coordinates, even for zero-speed motion."""
        c, m = self.cfg, self.mdp.motion
        p, R, vel = S[:3]
        bp, _, bv, _ = self.mdp.kin.forward(*S)
        kernel = lambda e2, scale: 1.0 / (1.0 + e2 / scale**2)
        yaw = torch.atan2(R[..., 1, 0], R[..., 0, 0])
        yaw_ref = torch.atan2(m.root_R[f][..., 1, 0], m.root_R[f][..., 0, 0])
        dyaw = torch.atan2(torch.sin(yaw - yaw_ref), torch.cos(yaw - yaw_ref))
        return dict(
            root=c.root_weight * kernel((p[..., :2] - m.root_p[f][..., :2]).square().sum(-1), c.root_scale),
            feet=c.foot_weight * kernel((bp[..., self.feet, :2] - m.body_p[f][..., self.feet, :2]).square().sum(-1), c.foot_scale).mean(-1),
            velocity=c.velocity_weight * kernel(vel[..., :2].square().sum(-1), c.velocity_scale),
            foot_velocity=c.foot_velocity_weight * kernel(bv[..., self.feet, :2].square().sum(-1), c.foot_velocity_scale).mean(-1),
            heading=c.heading_weight * kernel(dyaw.square(), c.heading_scale))

    def reward(self, S, f, v, a, a_prev):
        base = self.mdp.reward(S, f, a, a_prev)
        label = self.ctc.cfg.w_label * self.ctc.label_reward(S, f, v, self.mdp.motion.boxes(f, v))
        return base + self.mdp.robot.control_dt * (sum(self.terms(S, f).values()) + label)


def freeze_normalizers(*models):
    from .networks import RunningNorm
    for model in models:
        for module in model.modules():
            if isinstance(module, RunningNorm):
                module.max_count = 0  # Preserve saved mean/variance/count, also with Runtime.norm_update.
