"""Reference-free wide-angle task; allows small adjustment steps and turning.

Training initialization must supply a pose selected from the learned tracker.
This task definition is not a trained controller or a physical acceptance claim.
"""
import torch
from .aiming_fullbody import FullbodyTask, flat_boxes
from .aiming_targets import spherical_targets
from .task import BodySpheres, goal_command, to_base
from .terrain import height_scan
from ..rotations import quat_to_mat


class WideAimingTask(FullbodyTask):
    def __init__(self, device='cpu', config=None):
        config = dict(config or {})
        self.support_capture_steps = config.get('support_capture_steps')
        self.support_command_mode = config.get('support_command_mode','position')
        if self.support_command_mode not in ('position','error'):
            raise ValueError('Support command must be measured position or position error')
        if self.support_command_mode=='error' and self.support_capture_steps is None:
            raise ValueError('Support error commands require observed support capture')
        if self.support_capture_steps is not None and (isinstance(self.support_capture_steps,bool)
                or not isinstance(self.support_capture_steps,int) or not 1<=self.support_capture_steps<=150):
            raise ValueError('Observed support capture must occur within the allowed first 150 control steps')
        weights = dict(feet=0., heading=0., foot_boundary=0.)
        weights.update(config.get('reward_weights', {}))
        forbidden=('heading','foot_boundary') if self.support_capture_steps is not None else ('feet','heading','foot_boundary')
        if any(weights[k] != 0 for k in forbidden):
            raise ValueError('Wide task permits stepping/turning; fixed-anchor penalties must remain zero')
        config['reward_weights'] = weights
        super().__init__(device, config)
        if 'initial_q' not in config or 'initial_height' not in config:
            raise ValueError('Supply a tracker-derived initial pose and root height')
        self.initial_q = self.alpha.new_tensor(config['initial_q'])
        if self.initial_q.shape != (29,) or not self.initial_q.isfinite().all():
            raise ValueError('Expected 29 finite initial joint values')
        if ((self.initial_q < self.mdp.soft_lo) | (self.initial_q > self.mdp.soft_hi)).any():
            raise ValueError('Initial pose exceeds soft joint limits')
        self.initial_height = float(config['initial_height'])
        if not 0.5 < self.initial_height < 1.5:
            raise ValueError('Invalid initial root height')
        self.initial_quat = self.alpha.new_tensor(config.get('initial_quat', [1., 0., 0., 0.]))
        if self.initial_quat.shape != (4,) or not self.initial_quat.isfinite().all() or abs(self.initial_quat.norm().item()-1.) > 1e-4:
            raise ValueError('Initial root quaternion must be unit wxyz')
        spheres = BodySpheres(self.robot, 'spheres224')
        mask = spheres.owner < 2
        self.foot_points = spheres.pts[mask].to(device)
        self.foot_radii = spheres.rad[mask].to(device)
        self.foot_owners = spheres.owner[mask].to(device)
        self.foot_links = torch.tensor(self.feet, device=device)[self.foot_owners]

    def bootstrap_aim_weight(self):
        if 'curriculum_level' in self.config:
            from .wide_curriculum import STAGES
            return STAGES[self.config['curriculum_level']]['aim_weight']
        return super().bootstrap_aim_weight()

    def metadata(self):
        result = super().metadata()
        result.update(name='wide_aiming_v1', version=1, contact_valid=[False]*9,
                      contact_command='zero; no future reference or fixed foot anchors',
                      target_coordinates='world-fixed, spherical range/azimuth/elevation from initial chest',
                      adjustment_steps=True, root_activity_radius_m=.5,
                      initial_q=self.initial_q.cpu().tolist(), initial_height=self.initial_height,
                      initial_quat=self.initial_quat.cpu().tolist())
        result['physics_substeps'] = self.config.get('physics_substeps', self.robot.decimation)
        result['physics_dt'] = self.dt / result['physics_substeps']
        result['imagination_command_interval_steps'] = self.config.get('switch_interval_steps')
        if self.support_capture_steps is not None:
            result.update(version=2,contact_valid=[True,True]+[False]*7,
                          contact_command='current measured support positions after adjustment; zero during adjustment or commanded movement',
                          observed_support=dict(capture_steps=self.support_capture_steps,frame='world',
                                                release='explicit target change or nonzero velocity command',
                                                imagination='truncate before the unobserved capture transition'))
            if self.support_command_mode=='error':
                result['observed_support']['command']='anchor minus current feet, in the current base frame'
                result['contact_command']='measured support position error; zero during adjustment or commanded movement'
        if any(key in self.config for key in ('reset_position_noise_m','reset_rotation_noise_rad','reset_joint_noise_rad')):
            result['reset_noise']={key:float(self.config.get(key,0.)) for key in
                ('reset_position_noise_m','reset_rotation_noise_rad','reset_joint_noise_rad')}
        return result

    def soft_termination(self,S,next_step,ctx):
        result=super().soft_termination(S,next_step,ctx)
        if self.support_capture_steps is not None:
            capture=(ctx['support_age_steps']>=self.support_capture_steps)&(ctx['velocity_command'].norm(dim=-1)<1e-6)
            result=torch.maximum(result,capture.to(result.dtype))
        interval=self.config.get('switch_interval_steps')
        if interval:
            result=torch.maximum(result,((next_step>0)&(next_step%interval==0)).to(result.dtype))
        return result

    def advance_context(self,ctx,steps):
        if self.support_capture_steps is None:return super().advance_context(ctx,steps)
        age=ctx['support_age_steps']
        # -1 denotes captured support; its anchor is fixed until explicitly released.
        steps=torch.as_tensor(steps,device=age.device,dtype=age.dtype)
        return dict(ctx,support_age_steps=torch.where(age>=0,age+steps,age))

    def observe_context(self,S,ctx):
        if self.support_capture_steps is None:return ctx
        moving=ctx['velocity_command'].norm(dim=-1)>1e-6
        age=torch.where(moving,torch.zeros_like(ctx['support_age_steps']),ctx['support_age_steps'])
        capture=age>=self.support_capture_steps
        feet=ctx['feet_anchor_w']
        if bool(capture.any()):
            measured=self.kin.forward(*S)[0][:,self.feet].detach()
            feet=torch.where(capture[:,None,None],measured,feet)
        return dict(ctx,feet_anchor_w=feet,
                    support_age_steps=torch.where(capture,-torch.ones_like(age),age))

    def release_support(self,ctx,ids=None):
        if self.support_capture_steps is not None:
            if ids is None:ctx['support_age_steps'].zero_()
            else:ctx['support_age_steps'][ids]=0

    def initial_state(self, n):
        state = super().initial_state(n)
        state[0][:, 2] = self.initial_height
        rotation = quat_to_mat(self.initial_quat[None]).expand(n, -1, -1).clone()
        return state[0], rotation, *state[2:]

    def context(self, S, parameters, aim_weight=None):
        context = super().context(S, parameters, aim_weight)
        bp, bR, _, _ = self.kin.forward(*S, all_bodies=True)
        shoulders = [self.kin.joint_idx.index(self.robot.joint_names.index(f'shoulder_pitch_{s}_joint')) for s in ('l', 'r')]
        chest = (bp[:, shoulders] + (bR[:, shoulders] @ self.kin.anchors[shoulders, :, None]).squeeze(-1)).mean(1)
        target, normal = spherical_targets(chest, context['yaw_anchor'][:, 0], parameters)
        context.update(aim_target_w=target, aim_normal_w=normal)
        return context

    def command(self, S, ctx):
        fk=self.kin.forward(*S) if self.support_command_mode=='error' else None
        h, direction, torso = self.hand(S,fk)
        rotate = lambda x: (torso.transpose(-1, -2) @ x[..., None]).squeeze(-1)
        desired_yaw = torch.atan2(ctx['aim_target_w'][:, 1]-S[0][:, 1], ctx['aim_target_w'][:, 0]-S[0][:, 0])
        contacts=S[0].new_zeros(len(S[0]),27)
        if self.support_capture_steps is not None:
            locked=(ctx['support_age_steps']<0)&(ctx['velocity_command'].norm(dim=-1)<1e-6)
            if self.support_command_mode=='error':
                delta=ctx['feet_anchor_w']-fk[0][:,self.feet]
                feet=(S[1].transpose(-1,-2)[:,None]@delta[...,None]).squeeze(-1)
            else:
                feet=to_base(ctx['feet_anchor_w'],S[0],S[1])
            feet=feet*locked[:,None,None]
            contacts=torch.cat((feet.flatten(1),contacts[:,6:]),-1)
        return torch.cat((contacts,
                          goal_command(S, ctx['root_anchor_w'], desired_yaw, .5),
                          height_scan(S[0], S[1], flat_boxes(S), self.policy_scan),
                          rotate(ctx['aim_target_w']-h)/3., rotate(direction), rotate(ctx['aim_normal_w']),
                          ctx['aim_tolerance']/.2, ctx['velocity_command']), -1)

    def reward_terms(self, S, a, a_prev, ctx, *, raw=None, xi=None):
        terms, _ = super().reward_terms(S, a, a_prev, ctx, raw=raw, xi=xi)
        if self.support_capture_steps is not None:
            locked=(ctx['support_age_steps']<0)&(ctx['velocity_command'].norm(dim=-1)<1e-6)
            terms['feet']=terms['feet']*locked.to(terms['feet'].dtype)
        bp, br, bv, bw = self.kin.forward(*S)
        offsets = (br[:, self.foot_links] @ self.foot_points[..., None]).squeeze(-1)
        centers = bp[:, self.foot_links] + offsets
        velocities = bv[:, self.foot_links] + torch.linalg.cross(bw[:, self.foot_links], offsets, dim=-1)
        near = torch.sigmoid((.005-(centers[..., 2]-self.foot_radii))/.01)
        slip = (near*velocities[..., :2].square().sum(-1)).sum(-1)/near.sum(-1).clamp_min(1.)
        terms['foot_velocity'] = 1/(1+slip/.05**2)
        if 'foot_motion' in self.weights:
            # Ground-slip alone leaves airborne stepping unpenalized. Encourage
            # quiet feet on a stop command; commanded locomotion still permits swing.
            stopped=ctx['velocity_command'].norm(dim=-1)<1e-6
            motion=bv[:,self.feet].square().sum(-1)
            terms['foot_motion']=stopped.to(motion.dtype)*(1/(1+motion/.05**2)).mean(-1)
        radius = (S[0][:, :2]-ctx['root_anchor_w'][:, :2]).norm(dim=-1)
        terms['root'] = 1/(1+(radius/.5).square())
        return terms, {k: value*(ctx['aim_weight'] if k=='aim' else self.weights[k])*self.dt
                       for k, value in terms.items()}
