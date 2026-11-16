"""Single-actor aiming. Geometry, commands and executed actions shared by all rollouts."""
import copy
import json
from pathlib import Path
import numpy as np
import torch
from .standing import StandingCfg, StandingTask, freeze_normalizers
from .networks import Policy, run_mlp
from .mdp import unflatten_state

SOURCE = 'logs/tienkung3_29dof/standing_aiming_from100500_20261111/train/best.pt'
DATA = 'data/tienkung3_29dof/standing_aiming'
DEFAULT_CONFIG = 'docs/config/standing_aiming_curriculum/default.json'

class AimingPolicy(Policy):
    def __init__(self, obs_dim, contact_dim, hist=10, robot='tienkung3_29dof'):
        super().__init__(obs_dim, contact_dim, hist=hist, robot=robot)
        self.contact_dim = contact_dim
        self.enc[0] = torch.nn.Linear(contact_dim + 17, self.enc[0].out_features)

    def forward(self, obs_hist, goal):
        g = torch.cat((self.goal_norm(goal[..., :self.contact_dim]), goal[..., self.contact_dim:]), -1)
        z = torch.tanh(run_mlp(self.enc, g))
        return run_mlp(self.dec, torch.cat((z, self.obs_norm(obs_hist).flatten(-2)), -1))

    def migrate(self, state, task):
        state = {k: v.detach().cpu().clone() for k, v in state.items()}
        w = state['enc.0.weight']
        state['enc.0.weight'] = torch.cat((w, w.new_zeros(w.shape[0], 17)), -1)
        state['dec.8.weight'][task.terminal.cpu()] = 0
        state['dec.8.bias'][task.terminal.cpu()] = ((-task.mdp.default_q[task.terminal]) / task.alpha[task.terminal]).cpu()
        self.load_state_dict(state)
        freeze_normalizers(self)

class AimingTask(StandingTask):
    def __init__(self, mdp, ctc, config):
        super().__init__(mdp, ctc, StandingCfg(foot_weight=config.get('foot_position_weight', 2.)))
        self.config = config
        self.fixed, self.terminal = self.arm_ids[:4], self.arm_ids[4:]
        self.free[self.terminal] = True
        self.body = torch.tensor([i for i in range(29) if i not in self.arm_ids], device=mdp.default_q.device)
        self.tcp = mdp.kin.tracked_names.index('right_tcp_link')
        self.torso = mdp.kin.tracked_names.index('waist_pitch_link')
        self.track_ids = [i for i in range(len(mdp.kin.tracked_names)) if i != self.tcp]
        self.rate_weights = torch.ones_like(self.alpha)
        self.rate_weights[self.terminal] = .2
        self.half = (mdp.soft_hi[self.terminal] - mdp.soft_lo[self.terminal]) / 2
        self.mid = (mdp.soft_hi[self.terminal] + mdp.soft_lo[self.terminal]) / 2
        ref = mdp.motion.ref_state(torch.zeros(1, dtype=torch.long, device=self.alpha.device))
        self.h_ref, self.n0, _ = self.hand(ref)
        up = self.n0.new_tensor([[0., 0., 1.]])
        self.e1 = torch.linalg.cross(up, self.n0); self.e1 /= self.e1.norm(dim=-1, keepdim=True)
        self.e2 = torch.linalg.cross(self.n0, self.e1)
        if config.get('target_frame', 'hand_initial') == 'chest_forward':
            kin = mdp.kin
            p, R, _, _ = kin.forward(*ref, all_bodies=True)
            shoulders = [kin.joint_idx.index(mdp.robot.joint_names.index(f'shoulder_pitch_{side}_joint')) for side in ('l', 'r')]
            anchors = p[:, shoulders] + (R[:, shoulders] @ kin.anchors[shoulders, :, None]).squeeze(-1)
            self.h_ref = anchors.mean(1)
            forward = R[:, kin.body_names.index('waist_pitch_link'), :, 0].clone()
            forward[:, 2] = 0
            self.n0 = forward / forward.norm(dim=-1, keepdim=True)
            self.e2 = up
            self.e1 = torch.linalg.cross(up, self.n0)
        self.max_step_reward += mdp.robot.control_dt * 2.5

    def metadata(self):
        return dict(name='standing_aiming_curriculum', version=1, config=self.config,
                    features=['target_xyz/3', 'direction_xyz', 'plane_xyz', 'velocity_xyz', 'rho/.2', 'sigma/.2', 'previous_terminal/soft_range'],
                    contact_dim=self.ctc.cmd_dim, aim_dim=17, axis=[0, 0, -1], fixed_names=list(self.mdp.robot.joint_names[i] for i in self.fixed.tolist()),
                    target_frame=dict(name=self.config.get('target_frame', 'hand_initial'), version=1,
                                      origin=self.h_ref.detach().cpu().tolist(), forward=self.n0.detach().cpu().tolist(),
                                      left=self.e1.detach().cpu().tolist(), up=self.e2.detach().cpu().tolist()))

    def hand(self, S):
        p, R, _, _ = self.mdp.kin.forward(*S)
        direction = -R[..., self.tcp, :, 2]
        direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return p[..., self.tcp, :], direction, R[..., self.torso, :, :]

    def command(self, S, f, v, gs, goal, previous):
        h, d, RB = self.hand(S); Rt = RB.transpose(-1, -2)
        rot = lambda x: (Rt @ x.unsqueeze(-1)).squeeze(-1)
        extra = torch.cat((rot(goal[..., :3]-h)/3, rot(d), rot(self.n0.expand_as(h)), rot(goal[..., 3:6]),
                           goal[..., 6:8]/.2, (previous-self.mid)/self.half), -1)
        return torch.cat((self.ctc.command(S, f, v, gs, noise=False), extra), -1)

    def compose_action(self, raw, xi, previous):
        q = self.mdp.default_q[self.terminal] + self.alpha[self.terminal] * raw[..., self.terminal]
        q = q.clamp(self.mdp.soft_lo[self.terminal], self.mdp.soft_hi[self.terminal])
        limit = self.config['terminal_speed'] * self.mdp.robot.control_dt
        q = previous + (q-previous).clamp(-limit, limit)
        a = self.compose(raw, xi)  # inherited: only the four fixed rows now have free=False
        a = a.scatter(-1, self.terminal.expand_as(q), (q-self.mdp.default_q[self.terminal]-xi[..., 1:30][..., self.terminal])/self.alpha[self.terminal])
        return a, q

    def metrics(self, S, goal):
        h, d, _ = self.hand(S); delta = goal[..., :3]-h
        D = delta.norm(dim=-1); u = delta / D.clamp_min(1e-6).unsqueeze(-1)
        c = (d*u).sum(-1).clamp(-1, 1)
        a = (delta*self.n0).sum(-1); b = (d*self.n0).sum(-1)
        lam = a / torch.where(b.abs()>1e-8, b, torch.ones_like(b))
        hit = h + lam.unsqueeze(-1)*d
        e = (hit-goal[..., :3]).norm(dim=-1)
        valid = (D>1e-6)&(a>0)&(b>1e-4)&(lam>0)&torch.isfinite(e)
        e = torch.where(valid, e, torch.full_like(e, float('inf')))
        return e, torch.rad2deg(torch.acos(c)), valid, hit

    def foot_boundary_cost(self, S, f):
        bp = self.mdp.kin.forward(*S)[0][..., self.feet, :2]
        error = (bp-self.mdp.motion.body_p[f][..., self.feet, :2]).norm(dim=-1)
        excess = ((error-.04)/.01).clamp_min(0)
        return self.config.get('foot_boundary_weight', 0.)*(1-torch.exp(-excess.square())).mean(-1)

    def aim_reward(self, S, f, v, a, ap, goal):
        h, d, _ = self.hand(S); delta = goal[..., :3]-h
        D2 = delta.square().sum(-1); u = delta / D2.sqrt().clamp_min(1e-6).unsqueeze(-1)
        c = (d*u).sum(-1).clamp(-1, 1)
        aim = .2*(1+c)/2 + .8/(1+2*D2*(1-c)/goal[..., 7].square())
        fixed = (1/(1+((S[4][..., self.fixed]-self.target_q[self.fixed])/np.deg2rad(5)).square())).mean(-1)
        home = (S[4][..., self.terminal]/self.half).square().mean(-1)
        base = self.mdp.reward(S, f, a, ap, body_ids=self.track_ids, action_weights=self.rate_weights)
        label = self.ctc.cfg.w_label*self.ctc.label_reward(S, f, v, self.mdp.motion.boxes(f,v))
        boundary = self.foot_boundary_cost(S,f) if self.config.get('foot_boundary_weight', 0.) else 0.
        return base + self.mdp.robot.control_dt*(sum(self.terms(S,f).values())+label+2*aim+.5*fixed-.01*home-boundary)

    def target(self, params, t):
        """Pure target process. Packed parameters are replayed, never supplied to actor."""
        mode = params[..., 0].long(); L = params[..., 1]
        angles = params[..., 4:6]
        phase = params[..., 8:10]*t.unsqueeze(-1)+params[..., 10:12]
        sine = angles + params[..., 6:8]*torch.sin(phase)
        sine_v = params[..., 6:8]*params[..., 8:10]*torch.cos(phase)
        nodes = params[..., 16:].reshape(*params.shape[:-1], 8, 6)
        idx = (t.unsqueeze(-1)+1e-6>=nodes[..., :, 0]).sum(-1).sub(1).clamp(0,6)
        at = lambda j: nodes.gather(-2, j[..., None, None].expand(*j.shape, 1, 6)).squeeze(-2)
        x0,x1 = at(idx), at(idx+1)
        T = (x1[...,0]-x0[...,0]).clamp_min(.02); s = ((t-x0[...,0])/T).clamp(0,1); ss=s.unsqueeze(-1); tt=T.unsqueeze(-1)
        herm = (2*ss**3-3*ss**2+1)*x0[...,1:3]+(ss**3-2*ss**2+ss)*tt*x0[...,3:5]+(-2*ss**3+3*ss**2)*x1[...,1:3]+(ss**3-ss**2)*tt*x1[...,3:5]
        herm_v = ((6*ss**2-6*ss)*x0[...,1:3]+(3*ss**2-4*ss+1)*tt*x0[...,3:5]+(-6*ss**2+6*ss)*x1[...,1:3]+(3*ss**2-2*ss)*tt*x1[...,3:5])/tt
        ab = torch.where((mode==1).unsqueeze(-1), x0[...,1:3], torch.where((mode==2).unsqueeze(-1),sine,torch.where((mode==3).unsqueeze(-1),herm,angles)))
        av = torch.where((mode==2).unsqueeze(-1), sine_v, torch.where((mode==3).unsqueeze(-1),herm_v,torch.zeros_like(angles)))
        xy = L.unsqueeze(-1)*torch.tan(ab); vel=L.unsqueeze(-1)*av/torch.cos(ab).square()
        g = self.h_ref+L.unsqueeze(-1)*self.n0+xy[...,0:1]*self.e1+xy[...,1:2]*self.e2
        gv=vel[...,0:1]*self.e1+vel[...,1:2]*self.e2
        return torch.cat((g,gv,params[...,2:4]),-1)

class AimingSim:
    """Owns target and limiter state; the base simulator receives already composed actions."""
    def __init__(self, base, task, sampler):
        self.base, self.task, self.sampler = base,task,sampler
        self.params = torch.zeros(base.N,64,device=base.device)
        self.previous = torch.zeros(base.N,3,device=base.device)
        self.reset(torch.arange(base.N,device=base.device), physical=False)

    def reset(self, ids, physical=True):
        if physical: self.base.reset(ids)
        self.params[ids] = self.sampler(len(ids))
        self.previous[ids] = self.base.state()[4][ids][:,self.task.terminal]
        self.base.a_prev[ids[:,None],self.task.terminal] = (self.previous[ids]-self.task.mdp.default_q[self.task.terminal])/self.task.alpha[self.task.terminal]

    def goal(self): return self.task.target(self.params,self.base.ep_step.float()*.02)

    def step(self, raw):
        p, prev, t = self.params.clone(),self.previous.clone(),self.base.ep_step.clone()
        a,q = self.task.compose_action(raw,self.base.xi,prev)
        g1=self.task.target(p,(t+1).float()*.02)
        out=self.base.step(a, composed=True, reward_fn=lambda S,f,v,a,ap:self.task.aim_reward(S,f,v,a,ap,g1))
        out.update(aim_params=p,terminal_prev=prev,aim_goal1=g1)
        self.previous=q.detach()
        if not self.base.eval_mode:
            ids=out['done'].nonzero().flatten()
            if len(ids): self.reset(ids,physical=False)
        return out

def setup(config, device='cuda:0'):
    from .robots import get_robot
    from .kinematics import RobotKinematics
    from .motion import MotionLib
    from .mdp import MDP
    from .task import ContactTask, ContactCfg
    robot=get_robot('tienkung3_29dof'); kin=RobotKinematics(device,robot)
    motion=MotionLib.from_dir(DATA+'/motions',kin,device)
    mdp=MDP(kin,motion,torch.tensor(robot.default_q,device=device))
    ctc=ContactTask(ContactCfg(contacts_dir=DATA+'/contacts',contact_geometry='spheres224',target_noise=0,scan_noise=0),mdp,DATA+'/motions',device)
    return AimingTask(mdp,ctc,config)

def policy(task, state=None):
    p=AimingPolicy(task.mdp.obs_dim,task.ctc.cmd_dim).to(task.alpha.device)
    if state is not None: p.load_state_dict(state)
    freeze_normalizers(p)
    return p

def json_save(path, obj):
    def clean(x):
        if isinstance(x,dict): return {k:clean(v) for k,v in x.items()}
        if isinstance(x,(list,tuple)): return [clean(v) for v in x]
        if isinstance(x,(float,np.floating)) and not np.isfinite(x): return None
        return x
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(clean(obj),indent=2,ensure_ascii=False)+'\n')
