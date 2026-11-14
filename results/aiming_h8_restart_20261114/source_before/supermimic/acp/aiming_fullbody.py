"""Reference-free standing/aiming v2. Pure task heads shared by physics and imagination."""
from __future__ import annotations

import math
import torch

from .kinematics import RobotKinematics
from .buffer import ReplayBuffer
from .mdp import MDP, invariant_features, normalize_xi
from .networks import Policy, run_mlp
from .robots import get_robot
from .standing import ARM_NAMES, held_pose, freeze_normalizers
from .task import goal_command, to_base
from .terrain import grid_offsets, height_scan
from ..rotations import yaw_mat

TASK_NAME = 'standing_aiming_fullbody_v2'
DEFAULT_CONFIG = 'docs/config/standing_aiming_fullbody_v2/default.json'
COMMAND_LAYOUT = {'contact_positions': [0, 27], 'standing_goal': [27, 30],
                  'height_scan': [30, 93], 'target_torso_div3': [93, 96],
                  'direction_torso': [96, 99], 'normal_torso': [99, 102],
                  'rho_sigma_div02': [102, 104], 'velocity_heading': [104, 107]}
CONTEXT_FIELDS = {
    'aim_target_w': (3, torch.float32), 'aim_normal_w': (3, torch.float32),
    'aim_tolerance': (2, torch.float32), 'velocity_command': (3, torch.float32),
    'root_anchor_w': (3, torch.float32), 'yaw_anchor': (1, torch.float32),
    'height_anchor': (1, torch.float32), 'feet_anchor_w': ((2, 3), torch.float32),
    'terrain_id': ((), torch.long), 'aim_weight': ((), torch.float32),
}
REPLAY_FIELDS = CONTEXT_FIELDS | {'terminated': ((), torch.bool), 'timeout': ((), torch.bool)}


class FullbodyReplay(ReplayBuffer):
    def sample_starts(self, B, hist, envs=None):
        """Require available real history; never pad from an overwritten mid-episode frame."""
        if self.size < hist:
            raise ValueError('Collect at least one full history before imagination')
        age = torch.randint(hist - 1, self.size, (B,), device=self.device)
        env = (torch.randint(0, self.N, (B,), device=self.device) if envs is None
               else envs[torch.randint(0, len(envs), (B,), device=self.device)])
        row = self._row(age)
        ep_step = self.data['ep_step'][row, env].long()
        back = torch.minimum(torch.arange(hist - 1, -1, -1, device=self.device)[None], ep_step[:, None])
        hrows = self._row(age[:, None] - back)
        start = {k: self.data[k][row, env] for k in ('s', 'a_prev', 'xi', 'f', 'v', 'goal_steps', 'goal', 'ep_step', *self.extra_keys)}
        history = {k: self.data[k][hrows, env[:, None]] for k in ('o', 'a_prev')}
        return start, history


def flat_boxes(S):
    # One unused slot preserves the existing height-scan implementation.
    return S[0].new_zeros((*S[0].shape[:-1], 1, 8))


def ray_metrics(h, d, target, normal):
    delta = target - h
    distance = delta.norm(dim=-1)
    cosine = (d * delta / distance.clamp_min(1e-6)[..., None]).sum(-1).clamp(-1, 1)
    numerator, denominator = (delta * normal).sum(-1), (d * normal).sum(-1)
    lam = numerator / torch.where(denominator.abs() > 1e-8, denominator, torch.ones_like(denominator))
    hit = h + lam[..., None] * d
    error = (hit - target).norm(dim=-1)
    valid = ((distance > 1e-6) & (numerator > 0) & (denominator > 1e-4)
             & (lam > 0) & torch.isfinite(error))
    return torch.where(valid, error, torch.full_like(error, float('inf'))), torch.rad2deg(torch.acos(cosine)), valid, hit


class FullbodyPolicy(Policy):
    def __init__(self, robot='tienkung3_29dof'):
        super().__init__(99, 93, hist=10, robot=robot)
        self.enc[0] = torch.nn.Linear(107, self.enc[0].out_features)
        freeze_normalizers(self)

    def forward(self, history, command):
        command = torch.cat((self.goal_norm(command[..., :93]), command[..., 93:]), -1)
        z = torch.tanh(run_mlp(self.enc, command))
        return run_mlp(self.dec, torch.cat((z, self.obs_norm(history).flatten(-2)), -1))


class FullbodyTask:
    def __init__(self, device='cuda:0', config=None):
        self.config = config or {}
        self.robot = robot = get_robot('tienkung3_29dof')
        self.kin = RobotKinematics(device, robot)
        self.mdp = MDP(self.kin, None, torch.tensor(robot.default_q, device=device))
        self.alpha = torch.tensor(robot.action_scales, device=device)
        self.initial_q = torch.tensor(held_pose(robot), device=device)
        self.arm_ids = torch.tensor([robot.joint_names.index(n) for n in ARM_NAMES], device=device)
        self.feet = [self.kin.tracked_names.index(robot.contact_bodies[n][0]) for n in ('left_foot', 'right_foot')]
        self.tcp = self.kin.tracked_names.index('right_tcp_link')
        self.torso = self.kin.tracked_names.index(robot.torso_body)
        self.policy_scan = grid_offsets((-.4, .4, 9), (-.3, .3, 7), device=device)
        self.episode_steps = 1500
        self.dt = robot.control_dt
        self.weights = dict(aim=2., root=2., feet=6., height=1., upright=1., velocity=1.,
                            angular_velocity=.2, foot_velocity=.5, heading=.5,
                            foot_boundary=-4., action_rate=-.1, joint_limit=-2.)
        self.weights.update(self.config.get('reward_weights', {}))
        self.max_step_reward = sum(max(0., w) for w in self.weights.values()) * self.dt
        self.term_height = self.config.get('term_height', .25)
        self.term_rzz = math.cos(math.radians(self.config.get('term_tilt_deg', 45.)))
        self.critic_dim = robot.inv_dim + 107 + 3 + 1 + 1 + 6 + 29 + robot.xi_dim + self.mdp.scan_dim + 1 + 990 + 1

    def metadata(self):
        return dict(name=TASK_NAME, version=2, command_layout=COMMAND_LAYOUT, obs_dim=99, hist=10,
                    command_dim=107, axis=[0, 0, -1], tcp='right_tcp_link', episode_steps=1500,
                    terrain='plane_z0', contact_valid=[True, True] + [False] * 7,
                    action='q_default + xi_joint_offset + action_scale * a_exec; uniform soft clip',
                    reward_weights=self.weights, term_height=self.term_height, term_rzz=self.term_rzz)

    def initial_state(self, n):
        p = self.alpha.new_zeros(n, 3)
        p[:, 2] = self.robot.stand_height
        R = torch.eye(3, device=p.device).expand(n, 3, 3).clone()
        return p, R, torch.zeros_like(p), torch.zeros_like(p), self.initial_q.expand(n, -1).clone(), self.alpha.new_zeros(n, 29)

    def context(self, S, parameters, aim_weight=2.):
        """parameters: (N,4) forward depth, alpha radians, beta radians, radius."""
        if parameters.shape != (len(S[0]), 4) or not torch.isfinite(parameters).all():
            raise ValueError('Expected finite [depth, alpha, beta, rho] per episode')
        if (parameters[:, [0, 3]] <= 0).any() or (parameters[:, 1:3].abs() >= math.pi / 2).any():
            raise ValueError('Positive depth/radius and forward angles required')
        bp, bR, _, _ = self.kin.forward(*S, all_bodies=True)
        shoulders = [self.kin.joint_idx.index(self.robot.joint_names.index(f'shoulder_pitch_{side}_joint')) for side in ('l', 'r')]
        chest = (bp[:, shoulders] + (bR[:, shoulders] @ self.kin.anchors[shoulders, :, None]).squeeze(-1)).mean(1)
        n0 = bR[:, self.kin.body_names.index(self.robot.torso_body), :, 0].clone()
        n0[:, 2] = 0
        n0 = torch.nn.functional.normalize(n0, dim=-1)
        up = torch.zeros_like(n0); up[:, 2] = 1
        left = torch.linalg.cross(up, n0)
        L, alpha, beta, rho = parameters.unbind(-1)
        target = chest + L[:, None] * (n0 + alpha.tan()[:, None] * left + beta.tan()[:, None] * up)
        return dict(aim_target_w=target, aim_normal_w=n0, aim_tolerance=rho[:, None].expand(-1, 2).clone(),
                    velocity_command=torch.zeros_like(n0), root_anchor_w=S[0].clone(),
                    yaw_anchor=torch.atan2(S[1][:, 1, 0], S[1][:, 0, 0])[:, None], height_anchor=S[0][:, 2:3].clone(),
                    feet_anchor_w=self.kin.forward(*S)[0][:, self.feet].clone(),
                    terrain_id=torch.zeros(len(L), dtype=torch.long, device=L.device),
                    aim_weight=torch.as_tensor(aim_weight, device=L.device).expand_as(L).clone())

    def compose_action(self, raw, xi):
        offset = xi[..., 1:30]
        q = (self.mdp.default_q + offset + self.alpha * raw).clamp(self.mdp.soft_lo, self.mdp.soft_hi)
        return (q - self.mdp.default_q - offset) / self.alpha, q

    def hand(self, S, fk=None):
        bp, bR, _, _ = self.kin.forward(*S) if fk is None else fk
        return bp[..., self.tcp, :], torch.nn.functional.normalize(-bR[..., self.tcp, :, 2], dim=-1), bR[..., self.torso, :, :]

    def command(self, S, ctx):
        feet = to_base(ctx['feet_anchor_w'], S[0], S[1])
        contacts = torch.cat((feet, feet.new_zeros(len(feet), 7, 3)), 1).flatten(1)
        h, d, torso = self.hand(S)
        rot = lambda x: (torso.transpose(-1, -2) @ x[..., None]).squeeze(-1)
        return torch.cat((contacts, goal_command(S, ctx['root_anchor_w'], ctx['yaw_anchor'][:, 0], .3),
                          height_scan(S[0], S[1], flat_boxes(S), self.policy_scan),
                          rot(ctx['aim_target_w'] - h) / 3., rot(d), rot(ctx['aim_normal_w']),
                          ctx['aim_tolerance'] / .2, ctx['velocity_command']), -1)

    def reward_terms(self, S, a, a_prev, ctx):
        p, R, v, w, q, _ = S
        fk = self.kin.forward(*S)
        bp, _, bv, _ = fk
        feet_error = bp[:, self.feet] - ctx['feet_anchor_w']
        yaw_error = torch.atan2(R[:, 1, 0], R[:, 0, 0]) - ctx['yaw_anchor'][:, 0]
        yaw_error = torch.atan2(yaw_error.sin(), yaw_error.cos())
        K = lambda e2, scale: 1 / (1 + e2 / scale**2)
        h, d, _ = self.hand(S, fk)
        delta = ctx['aim_target_w'] - h
        D2 = delta.square().sum(-1)
        cosine = (d * delta / D2.clamp_min(1e-12).sqrt()[:, None]).sum(-1).clamp(-1, 1)
        vh = (yaw_mat(R).transpose(-1, -2) @ v[..., None]).squeeze(-1)
        wz = torch.cat((torch.zeros_like(w[:, :2]), ctx['velocity_command'][:, 2:3]), -1)
        terms = dict(
            aim=.1 * (1 + cosine) + .8 / (1 + 2 * D2 * (1 - cosine) / ctx['aim_tolerance'][:, 1].square()),
            root=K((p[:, :2] - ctx['root_anchor_w'][:, :2]).square().sum(-1), .10),
            feet=K(feet_error.square().sum(-1), .05).mean(-1),
            height=K((p[:, 2] - ctx['height_anchor'][:, 0]).square(), .05),
            upright=1 / (1 + (1 - R[:, 2, 2]).clamp_min(0) / (1 - math.cos(math.radians(10)))),
            velocity=K((vh[:, :2] - ctx['velocity_command'][:, :2]).square().sum(-1), .10),
            angular_velocity=K((w - wz).square().sum(-1), .5),
            foot_velocity=K(bv[:, self.feet].square().sum(-1), .10).mean(-1),
            heading=K(yaw_error.square(), math.radians(5)),
            foot_boundary=(1 - torch.exp(-torch.relu((feet_error[..., :2].norm(dim=-1) - .04) / .01).square())).mean(-1),
            action_rate=(a - a_prev).square().sum(-1),
            joint_limit=(torch.relu(q - self.mdp.soft_hi) + torch.relu(self.mdp.soft_lo - q)).sum(-1))
        contributions = {k: x * (ctx['aim_weight'] if k == 'aim' else self.weights[k]) * self.dt for k, x in terms.items()}
        return terms, contributions

    def reward(self, S, a, a_prev, ctx):
        return sum(self.reward_terms(S, a, a_prev, ctx)[1].values())

    def termination(self, S, next_step, ctx):
        finite = torch.stack([x.flatten(1).isfinite().all(-1) for x in S]).all(0)
        terminated = (~finite | ((S[0][:, 2] - ctx['height_anchor'][:, 0]).abs() > self.term_height)
                      | (S[1][:, 2, 2] < self.term_rzz))
        return terminated, next_step >= self.episode_steps

    def soft_termination(self, S, next_step, ctx):
        alive = ((1 - torch.sigmoid(((S[0][:, 2] - ctx['height_anchor'][:, 0]).abs() - self.term_height) / .02))
                 * torch.sigmoid((S[1][:, 2, 2] - self.term_rzz) / .04))
        return torch.where(next_step >= self.episode_steps, torch.ones_like(alive), 1 - alive)

    def critic_features(self, S, history, a_prev, xi, step, ctx):
        yaw = torch.atan2(S[1][:, 1, 0], S[1][:, 0, 0]) - ctx['yaw_anchor'][:, 0]
        yaw = torch.atan2(yaw.sin(), yaw.cos())[:, None]
        feet = self.kin.forward(*S)[0][:, self.feet] - ctx['feet_anchor_w']
        feet = (yaw_mat(S[1]).transpose(-1, -2)[:, None] @ feet[..., None]).squeeze(-1)
        return torch.cat((invariant_features(S), self.command(S, ctx),
                          to_base(ctx['root_anchor_w'], S[0], S[1]), yaw,
                          S[0][:, 2:3] - ctx['height_anchor'], feet.flatten(1), a_prev, normalize_xi(xi),
                          self.mdp.scan(S, flat_boxes(S)), (1 - step / self.episode_steps).clamp(0, 1)[:, None],
                          history.flatten(1), ctx['aim_weight'][:, None]), -1)

    def metrics(self, S, ctx):
        h, d, _ = self.hand(S)
        return ray_metrics(h, d, ctx['aim_target_w'], ctx['aim_normal_w'])


def validate_checkpoint(ck, task):
    if ck.get('task_config') != task.metadata():
        raise ValueError('v2 task/layout/reward mismatch; old tasks cannot be resumed')
    if ck.get('robot_metadata') != task.robot.metadata():
        raise ValueError('Robot identity, joint order or control parameters differ')


def migrate_policy(ck, task):
    """Known 93/110 layouts only, with output rows released exactly once."""
    if ck.get('robot_metadata') != task.robot.metadata():
        raise ValueError('Source robot metadata does not match runtime')
    state = {k: v.detach().clone() for k, v in ck['pi'].items()}
    w = state['enc.0.weight']; old_dim = w.shape[1]
    meta = ck.get('task_config', {})
    mapped = w.new_zeros(w.shape[0], 107)
    if old_dim == 110:
        expected_features = ['target_xyz/3', 'direction_xyz', 'plane_xyz', 'velocity_xyz', 'rho/.2', 'sigma/.2', 'previous_terminal/soft_range']
        if (meta.get('name') != 'standing_aiming_curriculum' or meta.get('contact_dim') != 93
                or meta.get('features') != expected_features or meta.get('axis') != [0, 0, -1]
                or meta.get('fixed_names') != list(ARM_NAMES[:4])):
            raise ValueError('Unknown 110-column source semantics')
        mapped[:, :102] = w[:, :102]; mapped[:, 102:104] = w[:, 105:107]
        reset_names = ARM_NAMES[:4]
    elif old_dim == 93:
        if (meta.get('config', {}).get('name') != 'standing_aiming'
                or meta.get('arm_names') != list(ARM_NAMES) or ck.get('ctc_cfg', {}).get('blind', False)):
            raise ValueError('Expected the identified 93-column fixed-arm standing source')
        mapped[:, :93] = w
        reset_names = ARM_NAMES
    else:
        raise ValueError(f'Unsupported source command width: {old_dim}')
    state['enc.0.weight'] = mapped
    ids = [task.robot.joint_names.index(n) for n in reset_names]
    state['dec.8.weight'][ids] = 0
    state['dec.8.bias'][ids] = ((task.initial_q - task.mdp.default_q) / task.alpha).cpu()[ids]
    pi = FullbodyPolicy(task.robot).to(task.alpha.device)
    pi.load_state_dict(state)
    return pi, dict(source_command_dim=old_dim, destination_command_dim=107,
                    reset_output_names=list(reset_names), new_velocity_columns='zero',
                    critic='new', optimizers='new', curriculum='A', normalizers='source frozen')
