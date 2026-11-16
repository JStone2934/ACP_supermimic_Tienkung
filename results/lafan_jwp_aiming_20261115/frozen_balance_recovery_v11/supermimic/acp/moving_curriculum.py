"""Low-speed stages after accepted standing aiming; targets change only every10s."""
import math
import torch
from .aiming_targets import spherical_targets,fixed_target_grid

STAGES=[dict(name=f'speed_{speed:.2f}',speed=speed) for speed in (.05,.1,.2,.3)]
SPEED_FRACTIONS=(0.,1/3,2/3,1.,0.,.5)


def sample(level,n,device,generator=None):
    u=torch.rand(n,2,device=device,generator=generator)
    return torch.stack((torch.full_like(u[:,0],4.),(2*u[:,0]-1)*math.pi/3,
                        (2*u[:,1]-1)*math.pi/6,torch.full_like(u[:,0],.05)),-1),4.


def grid(level,device):
    values=fixed_target_grid(device)
    return values[values[:,0]==4.]


def retarget(task,state,context,parameters,ids):
    """New target from actual chest and original episode yaw, then world-fixed."""
    if not len(ids):return
    selected=tuple(x[ids] for x in state)
    probe=selected[0].new_tensor([3.,0.,0.,.05]).expand(len(ids),-1)
    physical=task.context(selected,probe)
    chest=physical['aim_target_w']-3.*physical['aim_normal_w']
    target,normal=spherical_targets(chest,context['yaw_anchor'][ids,0],parameters)
    context['aim_target_w'][ids]=target;context['aim_normal_w'][ids]=normal
    context['aim_tolerance'][ids]=parameters[:,3:4].expand(-1,2)
    task.release_support(context,ids)


def apply_commands(env,level,*,validation=False):
    """Explicit command boundaries; validation covers all speed phases in128steps."""
    if validation:
        fresh=(env.ep_step==0).nonzero().flatten()
        env.ep_step[fresh]=(fresh%len(SPEED_FRACTIONS))*250
    ids=(env.ep_step%250==0).nonzero().flatten()
    if len(ids):
        phase=(env.ep_step[ids]//250).clamp_max(5)
        speed=env.ep_step.new_tensor(SPEED_FRACTIONS,dtype=torch.float32)[phase]*STAGES[level]['speed']
        yaw=env.context['yaw_anchor'][ids,0]
        env.context['velocity_command'][ids]=torch.stack((speed*yaw.cos(),speed*yaw.sin(),torch.zeros_like(speed)),-1)
    changed=((env.ep_step>0)&(env.ep_step%500==0)).nonzero().flatten()
    if len(changed):
        parameters,_=sample(level,len(changed),env.ep_step.device)
        retarget(env.task,env.state(),env.context,parameters,changed)
