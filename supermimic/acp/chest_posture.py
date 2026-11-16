"""Chest posture preference and bounded, target-conditioned forward-lean allowance."""
import math
import numpy as np
import torch

DEFAULTS = dict(normal_limit_deg=10., low_target_limit_deg=25.,
                low_target_start_deg=-15., low_target_full_deg=-30.,
                lateral_limit_deg=10., backward_limit_deg=10., hard_limit_deg=35.,
                reward_scale_deg=10., grace_seconds=3., required_fraction=.95)


class ChestPosture:
    def __init__(self, config):
        self.enabled = config is not None
        self.settings = dict(DEFAULTS)
        if not self.enabled:return
        if not isinstance(config,dict) or set(config)-set(DEFAULTS):
            raise ValueError('Unknown chest posture settings')
        self.settings.update(config)
        s=self.settings
        if not all(math.isfinite(float(v)) for v in s.values()):raise ValueError('Nonfinite chest posture setting')
        if not (0<s['normal_limit_deg']<=s['low_target_limit_deg']<=s['hard_limit_deg']<90
                and -90<s['low_target_full_deg']<s['low_target_start_deg']<=0
                and 0<s['lateral_limit_deg']<=s['normal_limit_deg']
                and 0<s['backward_limit_deg']<=s['normal_limit_deg']
                and 0<s['reward_scale_deg']<90 and 0<=s['grace_seconds']<=3
                and .95<=s['required_fraction']<=1):raise ValueError('Invalid chest posture settings')

    def metadata(self):
        return dict(version=1,**self.settings,body_axis='native torso local +Z vs world +Z',
                    allowance='fixed commanded target-normal elevation; forward lean only',
                    preference='upright remains preferred even for low targets',
                    initial_pose='normal upright limit required before every episode')

    def measure(self, rotation, target_normal):
        s=self.settings;up=rotation[..., :,2]
        heading=torch.nn.functional.normalize(rotation[..., :2,0],dim=-1,eps=1e-6)
        left=torch.stack((-heading[...,1],heading[...,0]),-1)
        tilt=torch.rad2deg(torch.atan2(torch.linalg.vector_norm(up[...,:2],dim=-1),up[...,2]))
        pitch=torch.rad2deg(torch.atan2((up[...,:2]*heading).sum(-1),up[...,2]))
        roll=torch.rad2deg(torch.atan2((up[...,:2]*left).sum(-1),up[...,2]))
        # The command is fixed between switches. Bending the torso cannot unlock a larger allowance.
        elevation=torch.rad2deg(torch.atan2(target_normal[...,2],torch.linalg.vector_norm(target_normal[...,:2],dim=-1)))
        fraction=((s['low_target_start_deg']-elevation)/(s['low_target_start_deg']-s['low_target_full_deg'])).clamp(0,1)
        limit=s['normal_limit_deg']+fraction*(s['low_target_limit_deg']-s['normal_limit_deg'])
        excess=(torch.relu(tilt-limit).square()+torch.relu(roll.abs()-s['lateral_limit_deg']).square()
                +torch.relu(-pitch-s['backward_limit_deg']).square())/s['reward_scale_deg']**2
        preference=1/(1+(1-up[...,2]).clamp_min(0)/(1-math.cos(math.radians(s['reward_scale_deg']))))
        valid=(tilt<=limit+1e-4)&(roll.abs()<=s['lateral_limit_deg']+1e-4)&(pitch>=-s['backward_limit_deg']-1e-4)
        return dict(chest_tilt_deg=tilt,chest_forward_deg=pitch,chest_lateral_deg=roll,
                    chest_limit_deg=limit,chest_posture_valid=valid), dict(chest_upright=preference,chest_excess=excess)

    def episode(self, arrays, index, *, switching, dt=.02):
        s=self.settings;tilt=arrays['chest_tilt_deg'][:,index]
        finite=all(np.isfinite(arrays[k][:,index]).all() for k in
                   ('chest_tilt_deg','chest_forward_deg','chest_lateral_deg','chest_limit_deg'))
        initial=bool(finite and tilt[0]<=s['normal_limit_deg']+1e-4)
        hard=bool(finite and np.max(tilt)<=s['hard_limit_deg']+1e-4)
        cuts=[0,500,1000,len(tilt)] if switching else [0,len(tilt)]
        rates=[float(np.mean(arrays['chest_posture_valid'][start+round(s['grace_seconds']/dt):end,index]))
               for start,end in zip(cuts[:-1],cuts[1:])]
        return dict(passed=bool(initial and hard and all(r>=s['required_fraction'] for r in rates)),
                    initial_upright=initial,hard_limit_pass=hard,segment_post_grace_fractions=rates,
                    max_chest_tilt_deg=float(np.max(tilt)) if finite else None)


def apply_posture_acceptance(task, arrays, scenarios, switching):
    if not task.chest_posture.enabled:return
    for i,scenario in enumerate(scenarios):
        posture=task.chest_posture.episode(arrays,i,switching=switching(i))
        scenario['chest_posture']=posture
        scenario['passed']=bool(scenario['passed'] and posture['passed'])


def verify_posture_reports(reports):
    """New deliveries cannot use legacy pelvis-only acceptance or relaxed curriculum limits."""
    for report in reports:
        spec=report.get('task_config',{}).get('chest_posture')
        if not spec or spec.get('version')!=1:raise ValueError('Chest posture acceptance evidence required')
        settings={k:spec[k] for k in DEFAULTS}
        ChestPosture(settings)
        for k in ('normal_limit_deg','low_target_limit_deg','lateral_limit_deg','backward_limit_deg',
                  'hard_limit_deg','grace_seconds','low_target_start_deg','low_target_full_deg'):
            if settings[k]>DEFAULTS[k]:raise ValueError('Posture curriculum limits are not final acceptance')
        scenarios=report.get('scenarios',[])
        if len(scenarios)!=report.get('fixed_count',0)+report.get('random_count',0) or not scenarios:
            raise ValueError('Missing per-scenario posture evidence')
        for scenario in scenarios:
            posture=scenario.get('chest_posture',{})
            if 'passed' not in posture:raise ValueError('Missing chest posture result')
            if scenario.get('passed') and not (posture['passed'] and posture.get('initial_upright')
                    and posture.get('hard_limit_pass') and posture.get('segment_post_grace_fractions')
                    and all(v>=settings['required_fraction'] for v in posture['segment_post_grace_fractions'])):
                raise ValueError('Scenario pass contradicts chest posture evidence')
