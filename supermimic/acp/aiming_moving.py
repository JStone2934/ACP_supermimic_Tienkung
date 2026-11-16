"""Low-speed extension of the wide task, with an online velocity command.

The caller integrates the requested velocity into root_anchor_w, and changes
world-fixed aiming targets only at command-switch instants. No motion file is
used. Standing checkpoints remain separate and immutable.
"""
import torch
from .aiming_wide import WideAimingTask
from ..rotations import yaw_mat


class MovingAimingTask(WideAimingTask):
    def bootstrap_aim_weight(self):
        # Moving curricula always use aim_weight=4, including speed level0.
        return 4.

    def advance_context(self,ctx,steps):
        ctx=super().advance_context(ctx,steps)
        steps=torch.as_tensor(steps,device=self.alpha.device,dtype=self.alpha.dtype)
        if steps.ndim:steps=steps[...,None]
        return dict(ctx,root_anchor_w=ctx['root_anchor_w']+self.dt*steps*ctx['velocity_command'])

    def metadata(self):
        result=super().metadata()
        result.update(name='moving_aiming_v1',root_activity_radius_m=None,
                      root_goal='online integral of commanded world velocity',
                      velocity_context='world frame',velocity_policy_command='current yaw frame',
                      speed_range_m_s=[0.,.3],targets='world-fixed between explicit switches')
        return result

    def command(self,S,ctx):
        command=super().command(S,ctx)
        heading_velocity=(yaw_mat(S[1]).transpose(-1,-2)@ctx['velocity_command'][...,None]).squeeze(-1)
        return torch.cat((command[...,:104],heading_velocity),-1)

    def reward_terms(self,S,a,a_prev,ctx,*,raw=None,xi=None):
        terms,_=super().reward_terms(S,a,a_prev,ctx,raw=raw,xi=xi)
        error=(S[2]-ctx['velocity_command']).square().sum(-1)
        terms['velocity']=1/(1+error/.15**2)
        return terms,{k:value*(ctx['aim_weight'] if k=='aim' else self.weights[k])*self.dt
                      for k,value in terms.items()}
