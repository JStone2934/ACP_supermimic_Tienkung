"""Formal posture curriculum: reset choices and evidence-based progression."""
import math
import torch
from supermimic.rotations import quat_to_mat

RESET_PROBABILITIES=(0.,.1,.25,.5,1.)


def mixed_reset_state(state,mixture,condition=None):
    if mixture is None:
        if condition not in (None,'old'):raise ValueError('No repaired reset configured')
        return state
    probability=float(mixture['probability'])
    pose=state[0].new_tensor(mixture['reference_pose'])
    if not math.isfinite(probability) or not 0<=probability<=1 or pose.shape!=(36,) or not pose.isfinite().all():
        raise ValueError('Invalid reset mixture')
    if abs(float(pose[3:7].norm())-1)>1e-4 or not .5<float(pose[2])<1.5:
        raise ValueError('Invalid repaired root pose')
    if condition not in (None,'old','reference'):raise ValueError('Unknown reset condition')
    probability=({'old':0.,'reference':1.}[condition] if condition else probability)
    n=len(state[0]);choose=(torch.ones(n,device=pose.device,dtype=torch.bool) if probability==1 else
                           torch.zeros(n,device=pose.device,dtype=torch.bool) if probability==0 else
                           torch.rand(n,device=pose.device)<probability)
    p,R,v,w,q,qd=(x.clone() for x in state)
    # Keep the task's root XY origin; use repaired height/orientation/joint pose.
    p[choose,2]=pose[2];R[choose]=quat_to_mat(pose[3:7][None])[0];q[choose]=pose[7:]
    v[choose]=0;w[choose]=0;qd[choose]=0
    return p,R,v,w,q,qd


def conditions(level,reset_index):
    return ['old'] if level==0 and reset_index==0 else ['old','reference'] if level==0 else ['reference']


def next_phase(level,reset_index):
    if level==0 and reset_index<len(RESET_PROBABILITIES)-1:return 0,reset_index+1
    if level<6:return level+1,reset_index
    return None


def validate_evidence(checkpoint,reports,level,condition,*,final=False):
    from .train_wide_aiming import check_promotion
    from .robots import sha256
    if len(reports)!=2 or len({r.get('seed') for r in reports})!=2:
        raise ValueError('Two independent seeds required')
    for r in reports:
        if r.get('reset_pose_condition')!=condition:raise ValueError('Wrong reset evidence')
    if not final:return check_promotion(checkpoint,reports,level)
    digest=sha256(checkpoint)
    for r in reports:
        if (r.get('checkpoint_sha256')!=digest or r.get('level') is not None or
            r.get('scope')!='Full standing aiming acceptance' or not r.get('passed') or
            r.get('seconds')!=30. or not r.get('continuous_without_reset') or
            r.get('fixed_count')!=75 or r.get('random_count')!=100 or
            r.get('hit_radius_m')!=.05 or not r.get('online_target_switches')):
            raise ValueError('Incomplete final evidence')
        scenarios=r.get('scenarios',[])
        if len(scenarios)!=175 or not all(s['passed'] for s in scenarios[:75]) or sum(s['passed'] for s in scenarios[75:])<95:
            raise ValueError('Final per-scenario evidence failed')
    return digest
