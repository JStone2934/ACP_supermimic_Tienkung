"""50 Hz inference for the standing/moving policies, without motion libraries.

State is (world position, body-to-world rotation matrix, world linear velocity,
world angular velocity, native29 joint positions, native29 joint velocities).
Call reset once, set_target only when the world target changes, and step once
per 20 ms. The returned tensor contains native29 PD position targets in radians.
"""
from pathlib import Path
import torch
from .aiming_fullbody import FullbodyPolicy, validate_checkpoint
from .aiming_wide import WideAimingTask
from .aiming_moving import MovingAimingTask


class AimingPolicy:
    def __init__(self, checkpoint, device='cpu'):
        ck=torch.load(Path(checkpoint),map_location=device,weights_only=False)
        name=ck.get('task_config',{}).get('name')
        classes={'wide_aiming_v1':WideAimingTask,'moving_aiming_v1':MovingAimingTask}
        if name not in classes:raise ValueError('Expected a reference-free aiming checkpoint')
        self.task=classes[name](device,ck['config']);validate_checkpoint(ck,self.task)
        self.policy=FullbodyPolicy(self.task.robot,freeze_norm=True).to(device)
        self.policy.load_state_dict(ck['pi']);self.policy.eval().requires_grad_(False)
        self.moving=name=='moving_aiming_v1'
        self.history=None;self.context=None

    def _state(self,state):
        if len(state)!=6:raise ValueError('Expected six physical state tensors')
        state=tuple(torch.as_tensor(x,device=self.task.alpha.device,dtype=torch.float32) for x in state)
        n=len(state[0]);shapes=[(n,3),(n,3,3),(n,3),(n,3),(n,29),(n,29)]
        if any(x.shape!=shape or not bool(x.isfinite().all()) for x,shape in zip(state,shapes)):
            raise ValueError('Nonfinite state or incorrect native29 state dimensions')
        return state

    @torch.no_grad()
    def reset(self,state,parameters):
        state=self._state(state)
        parameters=torch.as_tensor(parameters,device=self.task.alpha.device,dtype=torch.float32)
        self.context=self.task.context(state,parameters)
        self.xi=state[0].new_zeros(len(state[0]),self.task.robot.xi_dim);self.xi[:,0]=1.
        self.previous_action=(state[4]-self.task.mdp.default_q)/self.task.alpha
        self.history=None

    @torch.no_grad()
    def set_target(self,position_w,normal_w,radius_m=.05):
        if self.context is None:raise RuntimeError('Call reset before setting a target')
        p=torch.as_tensor(position_w,device=self.task.alpha.device,dtype=torch.float32)
        n=torch.as_tensor(normal_w,device=p.device,dtype=p.dtype)
        if p.shape!=self.context['aim_target_w'].shape or n.shape!=p.shape or not bool(p.isfinite().all()&n.isfinite().all()):
            raise ValueError('Expected finite batched world target positions and plane normals')
        norm=n.norm(dim=-1,keepdim=True)
        radius=torch.as_tensor(radius_m,device=p.device,dtype=p.dtype)
        if (norm<1e-6).any() or not bool(radius.isfinite().all()) or (radius<=0).any():
            raise ValueError('Nonzero normal and positive radius required')
        tolerance=radius.expand(len(p))[:,None].expand(-1,2)
        self.context['aim_target_w']=p.clone();self.context['aim_normal_w']=(n/norm).clone()
        self.context['aim_tolerance']=tolerance.clone()

    @torch.no_grad()
    def set_velocity(self,velocity_w):
        if self.context is None:raise RuntimeError('Call reset before setting velocity')
        v=torch.as_tensor(velocity_w,device=self.task.alpha.device,dtype=torch.float32)
        if v.shape!=self.context['velocity_command'].shape or not bool(v.isfinite().all()):
            raise ValueError('Expected finite batched world velocity')
        if (v[:,2].abs()>1e-6).any() or (v.norm(dim=-1)>.300001).any():
            raise ValueError('Flat-ground velocity must be horizontal and at most 0.3 m/s')
        if not self.moving and (v!=0).any():raise ValueError('Standing policy does not support movement commands')
        self.context['velocity_command']=v.clone()

    @torch.no_grad()
    def step(self,state):
        if self.context is None:raise RuntimeError('Call reset before step')
        state=self._state(state)
        if len(state[0])!=len(self.previous_action):raise ValueError('Batch size changed without reset')
        observation=self.task.mdp.obs(state,self.previous_action,self.xi)
        self.history=(observation[:,None].repeat(1,10,1) if self.history is None else
                      torch.cat((self.history[:,1:],observation[:,None]),1))
        raw=self.policy(self.history,self.task.command(state,self.context))
        if not bool(raw.isfinite().all()):raise FloatingPointError('Nonfinite policy action')
        self.previous_action,targets=self.task.compose_action(raw,self.xi)
        if self.moving:self.context['root_anchor_w']+=self.task.dt*self.context['velocity_command']
        return targets
